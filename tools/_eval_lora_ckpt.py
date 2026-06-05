#!/usr/bin/env python3
"""Evaluate a LoRA adapter checkpoint on a Qwen3-ASR JSONL test set.

Metrics are delegated to MASR_Eval_Pkg (compute_wer / compute_cer).
"""

import argparse
import json
import os
import sys
import time
from typing import List

import librosa
import torch
from peft import PeftModel
from qwen_asr import Qwen3ASRModel

from masr_eval_pkg import compute_wer, compute_cer


def transcribe_all(wrapper, test_items, batch_size: int, language: str):
    results = []
    for i in range(0, len(test_items), batch_size):
        batch = test_items[i : i + batch_size]
        audios = []
        refs = []
        for item in batch:
            wav, _ = librosa.load(item["audio"], sr=16000, mono=True)
            audios.append((wav, 16000))
            refs.append(item["ref"])
        preds = wrapper.transcribe(audios, language=language)
        for item, pred in zip(batch, preds):
            results.append({
                "audio": item["audio"],
                "reference": item["ref"],
                "prediction": pred.text,
            })
        if (i // batch_size) % 50 == 0:
            print(f"  {i}/{len(test_items)}", flush=True)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True)
    p.add_argument("--adapter", required=True, help="LoRA adapter dir (checkpoint-XXX)")
    p.add_argument("--base_model", default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--language", default="Uyghur")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--output_predictions", default="")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] base model: {args.base_model}")
    wrapper = Qwen3ASRModel.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    print(f"[load] LoRA adapter: {args.adapter}")
    wrapper.model.thinker = PeftModel.from_pretrained(
        wrapper.model.thinker, args.adapter
    )

    test_items = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj["text"]
            # Extract transcript body from "language Uyghur<asr_text>BODY"
            if "<asr_text>" in text:
                text = text.split("<asr_text>", 1)[1].strip()
            test_items.append({"audio": obj["audio"], "ref": text})

    if args.max_samples > 0:
        test_items = test_items[: args.max_samples]

    print(f"[eval] {len(test_items)} samples, batch_size={args.batch_size}")
    start = time.time()
    results = transcribe_all(wrapper, test_items, args.batch_size, args.language)
    elapsed = time.time() - start
    print(f"[done] {len(results)} predictions in {elapsed:.0f}s ({len(results)/elapsed:.1f} samples/s)")

    # WER / CER via MASR_Eval_Pkg
    refs = [r["reference"] for r in results]
    hyps = [r["prediction"] for r in results]

    wer_result = compute_wer(refs, hyps)
    cer_result = compute_cer(refs, hyps)

    wer = wer_result["wer"] * 100
    cer = cer_result["cer"] * 100
    total_wer = wer_result["substitutions"] + wer_result["deletions"] + wer_result["insertions"]
    total_words = wer_result["n_ref_tokens"]
    total_cer = cer_result["substitutions"] + cer_result["deletions"] + cer_result["insertions"]
    total_chars = cer_result["n_ref_chars"]

    print(f"\n  WER: {wer:.2f}%  ({total_wer:.0f} edits / {total_words} words)")
    print(f"  CER: {cer:.2f}%  ({total_cer:.0f} edits / {total_chars} chars)")

    if args.output_predictions:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_predictions)), exist_ok=True)
        with open(args.output_predictions, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[save] {args.output_predictions}")


if __name__ == "__main__":
    main()
