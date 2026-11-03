"""Greedy decoding of manifests + MER on the shared mer_tokens split (DESIGN.md §11).

  python -m aut_ctc.evaluate --exp_dir exp/base --manifest data/dev_zh.jsonl data/dev_en.jsonl
  # streaming accuracy (same as StreamingRecognizer up to the running log-mel max): AuT window 2 s
  python -m aut_ctc.evaluate --exp_dir exp/stream --manifest data/dev_zh.jsonl --window 200 --left_chunks 4
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .data import Collator, ManifestDataset, duration_batches
from .decode import ctc_greedy, hyps_to_text
from .frontend import AUT_FRAMES_PER_CHUNK, AUT_MEL_PER_CHUNK, AuTFrontend
from .model import load_decoder
from .tokenizer import load_tokenizer, mer_tokens


def edit_distance(ref, hyp) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def score(refs, hyps) -> dict:
    """Error rate over mer_tokens (one per Han char / English word / digit run): CER on zh-only sets, WER on
    en-only sets, MER on code-switched sets."""
    errors = tokens = 0
    for ref, hyp in zip(refs, hyps):
        r = mer_tokens(ref)
        errors += edit_distance(r, mer_tokens(hyp))
        tokens += len(r)
    return {"mer": errors / max(tokens, 1), "errors": errors, "tokens": tokens, "utts": len(refs)}


def eval_dataset(manifest, tokenizer, decoder_cfg, style="woitn"):
    """Evaluation set: no duration / <unk> / feasibility filtering, only un-normalizable references dropped."""
    return ManifestDataset([manifest], tokenizer, style, min_duration=0.0, max_duration=float("inf"),
                           drop_unk=False, upsample=decoder_cfg.upsample, num_queries=decoder_cfg.num_queries,
                           check_feasible=False)


@torch.no_grad()
def decode_dataset(frontend, decoder, tokenizer, dataset, max_batch_seconds=600, n_window_infer=800, chunk_frames=0,
                   left_chunks=-1, local_frames=0, num_workers=2, bf16=True):
    """Greedy-decode a ManifestDataset -> [{"id", "ref", "hyp"}]."""
    was_training = decoder.training
    decoder.eval()
    batches = duration_batches(range(len(dataset)), dataset.durations, max_batch_seconds)
    loader = DataLoader(dataset, batch_sampler=batches, collate_fn=Collator(frontend.fe), num_workers=num_workers)
    cfg, results = decoder.cfg, []
    for batch in loader:
        feats, lens = frontend.encode(batch["input_features"], batch["mel_lens"], n_window_infer)
        with torch.autocast(frontend.device.type, dtype=torch.bfloat16, enabled=bf16):
            out = decoder(feats, lens, chunk_frames=chunk_frames, left_chunks=left_chunks, local_frames=local_frames)
        hyps = ctc_greedy(out["logits"].float(), out["lens"], upsample=cfg.upsample, num_queries=cfg.num_queries)
        for uid, ref, hyp in zip(batch["ids"], batch["texts"], hyps_to_text(tokenizer, hyps)):
            results.append({"id": uid, "ref": ref, "hyp": hyp})
    decoder.train(was_training)
    return results


def resolve_checkpoint(exp_dir: Path, name: str) -> Path:
    if name == "best":
        return exp_dir / json.loads((exp_dir / "best.json").read_text())[0]["path"]
    if name == "last":
        return exp_dir / "last.pt"
    return Path(name)


def main():
    from .train import cap_gpu_memory

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # see train.main
    p = argparse.ArgumentParser("AuT-CTC evaluation")
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--manifest", nargs="+", required=True)
    p.add_argument("--ckpt", default="best", help="best | last | path to a .pt")
    p.add_argument("--window", type=int, default=0, help="streaming: AuT window in mel frames (100/200/400); 0 = offline")
    p.add_argument("--left_chunks", type=int, default=-1)
    p.add_argument("--max_batch_seconds", type=float, default=600)
    p.add_argument("--out", default=None, help="write per-utterance hypotheses (JSONL)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    exp = Path(args.exp_dir)
    cfg = yaml.safe_load((exp / "config.yaml").read_text())
    cap_gpu_memory(torch.device(args.device), cfg.get("optim", {}).get("gpu_memory_limit_gb"))
    tokenizer = load_tokenizer(cfg["tokenizer"]["sp_model"], exp / "vocab.json")
    decoder, _ = load_decoder(resolve_checkpoint(exp, args.ckpt))
    decoder.to(args.device)
    acfg = cfg["aut"]
    frontend = AuTFrontend(acfg["model_dir"], args.device, getattr(torch, acfg.get("dtype", "bfloat16")),
                           decoder.cfg.layer_ids, acfg.get("attn_implementation", "sdpa"))
    if args.window:
        decode_kw = dict(n_window_infer=args.window, chunk_frames=AUT_FRAMES_PER_CHUNK * args.window // AUT_MEL_PER_CHUNK,
                         left_chunks=args.left_chunks)
    else:
        decode_kw = dict(n_window_infer=800, local_frames=int(cfg.get("eval", {}).get("local_frames", 0)))
    out_f = open(args.out, "w", encoding="utf-8") if args.out else None
    for manifest in args.manifest:
        ds = eval_dataset(manifest, tokenizer, decoder.cfg, cfg["tokenizer"].get("style", "woitn"))
        res = decode_dataset(frontend, decoder, tokenizer, ds, args.max_batch_seconds,
                             bf16=args.device != "cpu", **decode_kw)
        s = score([r["ref"] for r in res], [r["hyp"] for r in res])
        print(json.dumps({"manifest": manifest, **s, "dropped": dict(ds.stats)}, ensure_ascii=False), flush=True)
        for r in res if out_f else []:
            out_f.write(json.dumps({"manifest": manifest, **r}, ensure_ascii=False) + "\n")
    if out_f:
        out_f.close()


if __name__ == "__main__":
    main()
