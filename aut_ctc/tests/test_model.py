"""Decoder / loss / greedy tests (DESIGN.md §15 T12-T18).  python -m aut_ctc.tests.test_model"""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from ..decode import ctc_greedy
from ..frontend import AUT_FRAMES_PER_CHUNK
from ..losses import ctc_feasible, total_loss
from ..model import AuTCTCDecoder, CTCDecoderConfig, load_decoder, save_decoder


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    torch.manual_seed(0)
    torch.set_num_threads(8)
    lens = torch.tensor([161, 42])
    feats = torch.randn(2, 161, 1024) * (torch.arange(161)[None, :, None] < lens[:, None, None])

    # T13: parameter counts
    p_tied, p_untied = n_params(AuTCTCDecoder(CTCDecoderConfig())), n_params(AuTCTCDecoder(CTCDecoderConfig(tie_cond=False)))
    p_up, p_small = n_params(AuTCTCDecoder(CTCDecoderConfig(upsample=2))), n_params(
        AuTCTCDecoder(CTCDecoderConfig(d_model=384, n_layers=4)))
    assert abs(p_tied - 34.43e6) < 0.01e6 and abs(p_untied - 44.62e6) < 0.01e6, (p_tied, p_untied)
    print(f"[ok] T13 params: default {p_tied / 1e6:.2f}M, untied {p_untied / 1e6:.2f}M, 2x upsample {p_up / 1e6:.2f}M, "
          f"4x384 {p_small / 1e6:.2f}M")

    # T12: shapes and lens with queries + upsample; forward/backward
    V = 19911
    dec = AuTCTCDecoder(CTCDecoderConfig(num_queries=2, upsample=2))
    qids = torch.tensor([[0, 5], [0, 6]])
    out = dec([feats], lens, query_ids=qids)
    assert torch.equal(out["lens"], (lens + 2) * 2) and torch.equal(out["inter_lens"], lens + 2)
    assert out["logits"].shape == (2, (161 + 2) * 2, V) and torch.isfinite(out["logits"]).all()
    targets = torch.cat([torch.tensor([[3, 9], [4, 9]]), torch.randint(2, V, (2, 40))], dim=1)
    target_lens = torch.tensor([42, 14])
    loss, final, inter = total_loss(out, targets, target_lens)
    loss.backward()
    assert all(p.grad is not None for p in dec.parameters())
    print(f"[ok] T12 lens=(T+Q)*u, inter_lens=T+Q; logits {tuple(out['logits'].shape)}; loss {loss.item():.2f} "
          f"(ctc {final.item():.2f}, inter {inter.item():.2f}); every parameter gets a gradient")

    # T14: streaming finality (causal conv + chunk-causal mask)
    sdec = AuTCTCDecoder(CTCDecoderConfig(vocab_size=500)).eval()
    x, l1 = feats[:1], lens[:1]
    worst = 0.0
    for w in (100, 200, 400):
        c = AUT_FRAMES_PER_CHUNK * w // 100
        for left in (2, 4, -1):
            k = max(1, min(3, (int(l1) - 1) // c))
            xb = x.clone()
            xb[:, k * c :] = torch.randn_like(xb[:, k * c :])
            ya = sdec([x], l1, chunk_frames=c, left_chunks=left)["logits"]
            yb = sdec([xb], l1, chunk_frames=c, left_chunks=left)["logits"]
            worst = max(worst, (yb[:, : k * c] - ya[:, : k * c]).abs().max().item())
    assert worst < 1e-5, worst
    print(f"[ok] T14 W in {{1,2,4}} s x left in {{2,4,inf}}: finished chunks change by at most {worst:.1e}")

    # T15: queries in streaming see only chunk 0
    qdec = AuTCTCDecoder(CTCDecoderConfig(vocab_size=500, num_queries=2, upsample=2)).eval()
    chunk = 2 * AUT_FRAMES_PER_CHUNK
    x2 = x.clone()
    x2[:, 3 * chunk :] = torch.randn_like(x2[:, 3 * chunk :])
    z0 = qdec([x], l1, chunk_frames=chunk, left_chunks=4, query_ids=qids[:1])["logits"]
    z1 = qdec([x2], l1, chunk_frames=chunk, left_chunks=4, query_ids=qids[:1])["logits"]
    cut = (2 + 3 * chunk) * 2
    dq = (z1[:, :cut] - z0[:, :cut]).abs().max().item()
    x3 = x.clone()
    x3[:, :chunk] = torch.randn_like(x3[:, :chunk])
    z2 = qdec([x3], l1, chunk_frames=chunk, left_chunks=4, query_ids=qids[:1])["logits"]
    assert dq < 1e-5 and (z2[:, :4] - z0[:, :4]).abs().max().item() > 1e-4
    print(f"[ok] T15 streaming queries ignore future chunks ({dq:.1e}) and depend on chunk 0")

    # T16: very different lengths in one batch -> no NaN
    lens16 = torch.tensor([390, 3])
    o16 = sdec([torch.randn(2, 390, 1024)], lens16)
    o16c = sdec([torch.randn(2, 390, 1024)], lens16, chunk_frames=13, left_chunks=2)
    assert torch.isfinite(o16["logits"]).all() and torch.isfinite(o16c["logits"]).all()
    print("[ok] T16 lens [390, 3] (offline and chunk-causal): no NaN")

    # T17: feasibility; blank in targets
    assert ctc_feasible([5, 5, 6], 4) and not ctc_feasible([5, 5, 6], 3)
    bad = targets.clone()
    bad[1, 3] = 0
    try:
        total_loss(out, bad, target_lens)
        raise AssertionError("blank in targets must raise")
    except ValueError:
        pass
    pad_ok = targets.clone()
    pad_ok[1, 20:] = 0  # padding beyond target_len may be 0
    total_loss(out, pad_ok, target_lens)
    print("[ok] T17 ctc_feasible; blank inside targets raises, zero padding is fine")

    # T18: greedy timestamps with queries and upsample
    fake = torch.full((1, 10, 20), -9.0)
    fake[0, :, 0] = 0.0
    fake[0, 0, 3] = 5.0  # tag at a query position
    fake[0, 4, 7] = 5.0  # first sub-frame of AuT frame 0
    fake[0, 5, 7] = 5.0  # repeated -> collapsed
    fake[0, 7, 8] = 5.0  # second sub-frame of AuT frame 1 -> 0.08 + 0.04
    hq = ctc_greedy(fake, torch.tensor([10]), upsample=2, num_queries=2)
    assert [(i, None if t is None else round(t, 3)) for i, t in hq[0]] == [(3, None), (7, 0.0), (8, 0.12)], hq
    print("[ok] T18 greedy: query tags get time None, repeats collapse, sub-frame times use +0.04 s")

    # checkpoint round trip
    path = Path(tempfile.mkdtemp()) / "decoder.pt"
    save_decoder(path, sdec, step=1)
    loaded, ckpt = load_decoder(path)
    assert ckpt["step"] == 1 and torch.equal(loaded([x], l1)["logits"], sdec([x], l1)["logits"])
    print("[ok] save_decoder / load_decoder round trip")
    print("ALL MODEL TESTS PASSED")


if __name__ == "__main__":
    main()
