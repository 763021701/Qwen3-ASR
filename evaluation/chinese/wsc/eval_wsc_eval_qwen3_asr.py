#!/usr/bin/env python3
"""
Evaluate Qwen3-ASR on WSC-Eval-ASR (Sichuan Mandarin / 四川话).

Normalization is aligned with the Mandarin Chinese eval
(evaluation/chinese_medical/eval_chinese_medical_asr.py): both reference and
hypothesis are normalized at scoring time via masr_eval_pkg's ChineseNormalizer
(NFKC + number ITN 二零零八→2008 + punctuation removal), not at jsonl-prep time.

Metrics:
  - CER (Character Error Rate): pure character-level, space-free
  - MER (Mixed Error Rate): mixed Chinese/English tokenization
    (Chinese → individual chars, English → words, numbers → tokens)

Usage:
  python evaluation/chinese/wsc/eval_wsc_eval_qwen3_asr.py \
    --jsonl data/chinese/wsc/wsc_eval_easy_hard_qwen3.jsonl \
    --model Qwen/Qwen3-ASR-1.7B \
    --batch_size 4 \
    --output_predictions outputs/qwen3_asr_1.7b_wsc/predictions.jsonl \
    --output_summary outputs/qwen3_asr_1.7b_wsc/summary.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Set

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import parse_asr_output
from masr_eval_pkg import compute_cer, compute_corpus_mer, compute_mer
from masr_eval_pkg.normalizers import get_normalizer

_ASR_TEXT_TAG = "<asr_text>"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WSC-Eval-ASR (Sichuan Mandarin) ASR eval for Qwen3-ASR.")
    p.add_argument("--jsonl", required=True, help="Eval JSONL (audio + text fields)")
    p.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument(
        "--language",
        default=None,
        help="Forced language. 'None' = auto-detect (default). 'Chinese' = force Chinese.",
    )
    p.add_argument("--max_samples", type=int, default=0, help="Evaluate first N samples only.")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--context", type=str, default="")
    p.add_argument("--device_map", type=str, default="cuda:0")
    p.add_argument("--output_predictions", required=True, help="Per-sample predictions JSONL")
    p.add_argument("--output_summary", default="", help="Final metrics summary file")
    return p.parse_args()


def extract_reference_text(label: str) -> str:
    """Strip Qwen3 language prefix if present."""
    s = (label or "").strip()
    if not s:
        return ""
    if _ASR_TEXT_TAG in s:
        return s.split(_ASR_TEXT_TAG, 1)[1].strip()
    return s


def load_manifest(path: str, max_samples: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def load_completed_keys(path: str) -> Set[str]:
    done: Set[str] = set()
    if not os.path.isfile(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "audio" in rec:
                    done.add(rec["audio"])
            except json.JSONDecodeError:
                continue
    return done


def main() -> None:
    args = parse_args()
    rows = load_manifest(args.jsonl, args.max_samples)
    if not rows:
        print("No samples loaded.", file=sys.stderr)
        sys.exit(1)

    for i, ex in enumerate(rows):
        if "audio" not in ex or "text" not in ex:
            print(f"Line {i}: missing 'audio' or 'text'.", file=sys.stderr)
            sys.exit(1)
        if not os.path.isfile(ex["audio"]):
            print(f"Missing audio: {ex['audio']}", file=sys.stderr)
            sys.exit(1)

    # Language
    lang = args.language
    if lang is not None and lang.strip().lower() == "none":
        lang = None
    elif lang:
        lang = lang.strip()
    else:
        lang = None

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    print(f"Loading model from {args.model!r} ...")
    t0 = time.time()
    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=args.device_map,
        max_inference_batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Resume
    completed = load_completed_keys(args.output_predictions)
    todo = [i for i, ex in enumerate(rows) if ex["audio"] not in completed]
    print(f"Total: {len(rows)} | Done: {len(completed)} | Remaining: {len(todo)} | batch={args.batch_size} | lang={lang!r}")

    parent = os.path.dirname(args.output_predictions)
    if parent:
        os.makedirs(parent, exist_ok=True)

    wf = open(args.output_predictions, "a", encoding="utf-8") if todo else None
    t_start = time.time()
    done_count = len(completed)
    batch_count = 0

    try:
        for b_start in range(0, len(todo), args.batch_size):
            batch_idx = todo[b_start : b_start + args.batch_size]
            batch_audios = [rows[i]["audio"] for i in batch_idx]
            t_batch = time.time()

            try:
                outs = model.transcribe(
                    audio=batch_audios,
                    context=args.context or "",
                    language=lang,
                    return_time_stamps=False,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[OOM batch {b_start // args.batch_size}] fallback to 1-by-1", file=sys.stderr)
                outs = []
                for a in batch_audios:
                    try:
                        outs.extend(model.transcribe(audio=[a], context=args.context or "", language=lang, return_time_stamps=False))
                    except Exception as e:
                        print(f"  failed {a}: {e}", file=sys.stderr)
                        class _E:
                            text = ""
                        outs.append(_E())
            except Exception as e:
                print(f"Batch failed: {e}; fallback to 1-by-1", file=sys.stderr)
                outs = []
                for a in batch_audios:
                    try:
                        outs.extend(model.transcribe(audio=[a], context=args.context or "", language=lang, return_time_stamps=False))
                    except Exception as e2:
                        print(f"  failed {a}: {e2}", file=sys.stderr)
                        class _E:
                            text = ""
                        outs.append(_E())

            for idx, o in zip(batch_idx, outs):
                ex = rows[idx]
                ref_text = extract_reference_text(ex["text"])
                raw_out = getattr(o, "text", "") or ""
                _, hyp_text = parse_asr_output(raw_out, user_language=lang)
                hyp_lang, _ = parse_asr_output(raw_out, user_language=None) if raw_out else ("", "")
                rec = {
                    "audio": ex["audio"],
                    "index": ex.get("index", idx),
                    "reference": ref_text,
                    "hypothesis": hyp_text,
                    "detected_language": hyp_lang if raw_out else "",
                }
                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done_count += 1
            wf.flush()
            batch_count += 1

            elapsed = time.time() - t_start
            batch_dt = time.time() - t_batch
            utt_per_s = args.batch_size / max(batch_dt, 1e-6)
            remaining = len(todo) - b_start - len(batch_idx)
            eta_s = remaining / max(utt_per_s, 1e-6)
            print(
                f"[batch {batch_count:4d}] {done_count}/{len(rows)} utt | "
                f"{batch_dt:.2f}s ({utt_per_s:.1f} utt/s) | "
                f"elapsed {elapsed/60:.1f}m | ETA {eta_s/60:.1f}m",
                flush=True,
            )
    finally:
        if wf is not None:
            wf.close()

    # === Scoring ===
    print("\n=== Scoring ===")
    preds = []
    with open(args.output_predictions, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                preds.append(json.loads(line))

    # Align by audio path
    ref_by_audio = {ex["audio"]: extract_reference_text(ex["text"]) for ex in rows}
    refs, hyps = [], []
    missing = 0
    for p in preds:
        ref = ref_by_audio.get(p["audio"])
        if ref is None:
            missing += 1
            continue
        refs.append(ref)
        hyps.append(p["hypothesis"])

    if missing:
        print(f"Warning: {missing} predictions had no matching reference.", file=sys.stderr)

    # Normalize via ChineseNormalizer (NFKC + number ITN + punctuation removal)
    # Aligned with evaluation/chinese_medical/eval_chinese_medical_asr.py
    normalizer = get_normalizer(
        "zh",
        zh_convert="none",           # no traditional/simplified conversion
        number_normalize="to_arabic", # 二零零八 → 2008
    )

    # CER: space-free character-level
    ref_cer = [normalizer.normalize_for_cer(r) for r in refs]
    hyp_cer = [normalizer.normalize_for_cer(h) for h in hyps]
    cer_result = compute_cer(ref_cer, hyp_cer, per_sample=True)

    # MER: mixed Chinese/English tokenization
    ref_mer = [normalizer.normalize_for_mer(r) for r in refs]
    hyp_mer = [normalizer.normalize_for_mer(h) for h in hyps]
    mer_corpus = compute_corpus_mer(ref_mer, hyp_mer, text_preprocessed=True)

    # Per-sample MER
    per_sample_mer = [
        compute_mer(rm, hm, text_preprocessed=True)
        for rm, hm in zip(ref_mer, hyp_mer)
    ]
    per_sample_cer = cer_result["per_sample"]

    # Sentence accuracy
    exact_cer = sum(1 for s in per_sample_cer if s.get("cer", 1.0) == 0.0)
    exact_mer = sum(1 for s in per_sample_mer if s.ref_tokens == s.hyp_tokens)

    cer_val = cer_result["cer"]
    mer_val = mer_corpus.mer
    char_errors = cer_result["substitutions"] + cer_result["deletions"] + cer_result["insertions"]
    token_errors = mer_corpus.total_substitutions + mer_corpus.total_deletions + mer_corpus.total_insertions

    total_time = time.time() - t_start

    summary_lines = [
        "=== WSC-Eval-ASR (Sichuan Mandarin) ASR Evaluation ===",
        f"Model:               {args.model}",
        f"Manifest:            {args.jsonl}",
        f"Predictions:         {args.output_predictions}",
        f"Samples:             {len(preds)} / {len(rows)}",
        f"Language prompting:  {lang!r}",
        f"Batch size:          {args.batch_size}",
        f"Max new tokens:      {args.max_new_tokens}",
        "",
        "--- CER (Character Error Rate) ---",
        f"Corpus CER:          {cer_val * 100:.2f}%",
        f"Substitutions:       {cer_result['substitutions']}",
        f"Deletions:           {cer_result['deletions']}",
        f"Insertions:          {cer_result['insertions']}",
        f"Ref chars:           {cer_result['n_ref_chars']}",
        f"Sentence exact (CER): {exact_cer / max(len(per_sample_cer),1) * 100:.2f}% ({exact_cer}/{len(per_sample_cer)})",
        "",
        "--- MER (Mixed Error Rate) ---",
        f"Corpus MER:          {mer_val * 100:.2f}%",
        f"Substitutions:       {mer_corpus.total_substitutions}",
        f"Deletions:           {mer_corpus.total_deletions}",
        f"Insertions:          {mer_corpus.total_insertions}",
        f"Ref tokens:          {mer_corpus.total_n_ref_tokens}",
        f"Sentence exact (MER): {exact_mer / max(len(per_sample_mer),1) * 100:.2f}% ({exact_mer}/{len(per_sample_mer)})",
        "",
        "--- Normalization (aligned with Mandarin Chinese eval) ---",
        "ChineseNormalizer: NFKC + number ITN (二零零八 → 2008) + punctuation removal",
        "MixedTokenizer: Chinese → chars, English → words, numbers → tokens",
        "",
        f"Total time:          {total_time/60:.1f} min",
    ]
    summary = "\n".join(summary_lines)
    print(summary)

    if args.output_summary:
        parent = os.path.dirname(args.output_summary)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.output_summary, "w", encoding="utf-8") as f:
            f.write(summary + "\n")
        print(f"Wrote summary to {args.output_summary}")


if __name__ == "__main__":
    main()
