"""Front-end tests (DESIGN.md §15 T1-T6) on the repo's qwen_asr AuT.

  python -m aut_ctc.tests.test_frontend           # random weights, CPU
  python -m aut_ctc.tests.test_frontend --real    # plus T3/T5 on the real Qwen3-ASR-1.7B AuT (CPU, ~1 min)
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from ..frontend import (AUT_FRAMES_PER_CHUNK, AuTFrontend, aut_frame_times, aut_output_lengths, compute_features,
                        extract_aut_features, load_feature_extractor)
from .common import MODEL_DIR, random_aut, test_audio


def official_len(L):  # qwen_asr _get_feat_extract_output_lengths, verbatim
    leave = L % 100
    feat = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (L // 100) * 13


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(8)

    # T1: frame count formula and non-uniform frame times
    L = torch.arange(1, 5001)
    assert torch.equal(aut_output_lengths(L), official_len(L))
    t = aut_frame_times(14)
    assert abs(t[12] - 0.96) < 1e-6 and abs(t[13] - 1.0) < 1e-6
    print("[ok] T1 frame counts match the official formula on L=1..5000; frame 12 at 0.96 s, next chunk at 1.00 s")

    # T2: padding="longest"
    fe = load_feature_extractor(MODEL_DIR)
    feats, mel_lens = compute_features(fe, [np.zeros(48000, np.float32), np.zeros(21920, np.float32)])
    assert tuple(feats.shape) == (2, 128, 300) and mel_lens.tolist() == [300, 137], (feats.shape, mel_lens)
    default = fe([np.zeros(48000, np.float32)], sampling_rate=16000, return_tensors="pt")["input_features"]
    assert default.shape[-1] == 3000
    _, short_lens = compute_features(fe, [np.zeros(3200, np.float32)])
    assert short_lens.tolist() == [50], short_lens
    print("[ok] T2 padding=longest -> (2,128,300), mel_lens [300,137]; the extractor default pads to 30 s (3000); "
          "a 0.2 s clip is padded to 0.5 s (50 valid frames)")

    # T4 + window-mask fix (sdpa and eager), packed batch == per utterance
    w = 8 * AUT_FRAMES_PER_CHUNK
    for impl in ("sdpa", "eager"):
        aut = random_aut(4, impl)
        x, m = torch.randn(1, 128, 1237), torch.tensor([1237])
        (base,), lens = extract_aut_features(aut, x, m)
        pert = x.clone()
        pert[..., 900:1000] += 3.0  # 9-10 s, second 8 s window
        (new,), _ = extract_aut_features(aut, pert, m)
        assert lens.tolist() == [161]
        assert (new[0, :w] - base[0, :w]).abs().max() == 0 and (new[0, w:] - base[0, w:]).abs().max() > 1e-3
        (b2,), _ = extract_aut_features(aut, x, m, n_window_infer=200)
        (n2,), _ = extract_aut_features(aut, pert, m, n_window_infer=200)
        changed = ((n2[0] - b2[0]).abs().amax(-1) > 0).nonzero().flatten()
        assert changed.min() == 104 and changed.max() == 129, changed
        lens3 = torch.tensor([1237, 320, 57])
        batch = torch.randn(3, 128, 1237) * (torch.arange(1237)[None, None] < lens3[:, None, None])
        (packed,), out_lens = extract_aut_features(aut, batch, lens3)
        worst = 0.0
        for b in range(3):
            (alone,), _ = extract_aut_features(aut, batch[b : b + 1, :, : lens3[b]], lens3[b : b + 1])
            worst = max(worst, (packed[b, : out_lens[b]] - alone[0]).abs().max().item())
        assert worst < 1e-5, worst
        print(f"[ok] T4 [{impl}] W=800: window 1 unchanged when 9-10 s is perturbed; W=200: only frames "
              f"104..129 change; packed batch vs per utterance (incl. a 0.57 s clip) max diff {worst:.1e}")

    # T5 (layer hooks) and T6 (features can feed trainable layers: no inference_mode)
    aut = random_aut(4)
    (last, l3), lens = extract_aut_features(aut, torch.randn(1, 128, 500), torch.tensor([500]), layer_ids=(-1, 3))
    assert torch.allclose(aut.ln_post(l3), last, atol=1e-5)
    lin = torch.nn.Linear(last.size(-1), 4)
    lin(last).sum().backward()
    assert lin.weight.grad is not None and all(p.grad is None for p in aut.parameters())
    print("[ok] T5 layer_ids=(i,) is the output of layer i (ln_post(last layer) == ln_post output); "
          "T6 backward through a trainable layer works, AuT gets no gradient")

    if args.real:
        fr = AuTFrontend(MODEL_DIR, device="cpu", dtype=torch.float32, layer_ids=(-1, 23))
        a, b = test_audio(12.37, seed=1), test_audio(3.2, seed=2)
        feats, mel_lens = fr.features([a, b])
        (last, l23), lens = fr.encode(feats, mel_lens)
        assert lens.tolist() == [161, 42], lens
        (alone, _), _ = fr.encode(feats[1:, :, : mel_lens[1]], mel_lens[1:])
        diff = (last[1, : lens[1]] - alone[0]).abs().max().item()
        assert diff < 1e-4, diff
        assert torch.allclose(fr.aut.ln_post(l23[0, : lens[0]]), last[0, : lens[0]], atol=1e-4)
        single, _ = fr.features([b])
        edge = ((feats[1, :, : mel_lens[1]] - single[0]).abs().amax(0) > 1e-6).nonzero().flatten().tolist()
        print(f"[ok] T3 real AuT: 12.37 s -> {lens[0].item()} frames; batch vs single max diff {diff:.1e} (fp32); "
              f"T5 layer 23 -> ln_post; batched vs single log-mel differ only at frames {edge} (STFT edge padding)")
    print("ALL FRONTEND TESTS PASSED")


if __name__ == "__main__":
    main()
