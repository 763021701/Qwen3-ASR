"""Contextual-biasing demo: AuT -> AuT-CTC -> ctc_rag_hw retrieval -> hotwords in Qwen3-ASR's system prompt.

  python -m aut_ctc.tools.demo_biasing --exp_dir exp/mix3000h_stream --ckpt last \
      --hotwords /root/autodl-tmp/huangpu/ctc_rag_hw/examples/test_files/hotwords_lib.txt \
      --oracle /root/autodl-tmp/huangpu/ctc_rag_hw/examples/test_files/hotwords_surgery_0133.txt \
      --audio /root/autodl-tmp/huangpu/ctc_rag_hw/examples/test_files/surgery_dialogue_0133.wav

Decodes every audio without context ("none"), with the retrieved hotwords ("rag") and, given --oracle, with the
reference hotwords as context ("oracle"); prints transcripts, retrieved hotwords, oracle hits and stage timings.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import yaml

from qwen_asr.inference.utils import SAMPLE_RATE, normalize_audios

from ..biasing import CTCRagQwen3ASR
from ..train import cap_gpu_memory


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    p = argparse.ArgumentParser("AuT-CTC contextual biasing demo")
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--ckpt", default="best", help="best | last | path to a .pt")
    p.add_argument("--model_dir", default=None, help="Qwen3-ASR checkpoint (default: the one the CTC head was trained on)")
    p.add_argument("--audio", nargs="+", required=True)
    p.add_argument("--hotwords", required=True, help="hotword library, one per line")
    p.add_argument("--oracle", default=None, help="reference hotwords (for every --audio): adds the oracle run and hits")
    p.add_argument("--language", default=None)
    p.add_argument("--retriever_device", default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--max_hotwords", type=int, default=30)
    p.add_argument("--max_segment_sec", type=float, default=30.0)
    p.add_argument("--max_batch_seconds", type=float, default=240.0)
    p.add_argument("--max_batch_size", type=int, default=8)
    p.add_argument("--gpu_memory_limit_gb", type=float, default=8.0)
    p.add_argument("--out", default=None, help="write the results as JSONL")
    args = p.parse_args()

    device = torch.device("cuda:0")
    cap_gpu_memory(device, args.gpu_memory_limit_gb)
    model_dir = args.model_dir or yaml.safe_load((Path(args.exp_dir) / "config.yaml").read_text())["aut"]["model_dir"]
    pipe = CTCRagQwen3ASR.from_pretrained(model_dir, args.exp_dir, args.ckpt, hotwords=args.hotwords,
                                          retriever_device=args.retriever_device, device=str(device),
                                          max_hotwords=args.max_hotwords, max_segment_sec=args.max_segment_sec,
                                          max_batch_seconds=args.max_batch_seconds, max_batch_size=args.max_batch_size)
    wavs = normalize_audios(args.audio)
    audio_sec = sum(len(w) for w in wavs) / SAMPLE_RATE
    oracle = [w.strip() for w in open(args.oracle, encoding="utf-8") if w.strip()] if args.oracle else []
    runs = {"none": dict(context=""), "rag": {}}
    if oracle:
        runs["oracle"] = dict(context=" ".join(oracle))
    pipe.transcribe([(wavs[0][: 5 * SAMPLE_RATE], SAMPLE_RATE)])  # warm-up: CUDA kernels, retrieval caches

    out_f = open(args.out, "w", encoding="utf-8") if args.out else None
    for mode, kw in runs.items():
        results = pipe.transcribe([(w, SAMPLE_RATE) for w in wavs], language=args.language, **kw)
        ms = {k: round(v * 1000) for k, v in pipe.timings.items()}
        print(f"\n===== {mode}: {audio_sec:.1f} s of audio, RTF {pipe.timings['total'] / audio_sec:.4f}, ms per stage {ms}")
        for path, r in zip(args.audio, results):
            row = {"mode": mode, "audio": path, "language": r.language, "text": r.text, "ctc_text": r.ctc_text,
                   "hotwords": r.hotwords, "timings_ms": ms}
            print(f"[{Path(path).name}] {r.language}: {r.text}")
            if mode == "rag":
                print(f"  ctc: {r.ctc_text}\n  retrieved ({len(r.hotwords)}): {' '.join(r.hotwords)}")
            if oracle:
                counts = {w: r.text.lower().count(w.lower()) for w in oracle}
                row["oracle_counts"] = counts
                print(f"  oracle hotword occurrences in the text: {sum(counts.values())}, "
                      f"missing {[w for w, n in counts.items() if not n]}")
                if mode == "rag":
                    found = set(r.hotwords) & set(oracle)
                    row["retrieval"] = {"recall": len(found) / len(oracle), "precision": len(found) / max(len(r.hotwords), 1)}
                    print(f"  retrieval: {len(found)}/{len(oracle)} oracle hotwords among {len(r.hotwords)} retrieved")
            if out_f:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if out_f:
        out_f.close()


if __name__ == "__main__":
    main()
