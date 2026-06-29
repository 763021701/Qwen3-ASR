#!/usr/bin/env python3
# coding=utf-8
"""
Evaluate Qwen3-ASR on a pure-English medical TTS jsonl manifest.

The manifest is expected to contain one JSON object per line with at least:
  {"audio": "/abs/path.wav", "text": "<reference transcript>"}

The reference text should be raw (no "language ...<asr_text>" prefix). If it
contains the prefix it will be stripped automatically.

Model inference is run in batches and predictions are appended to the output
predictions file incrementally, so a crash can be recovered by re-running with
the same arguments — already-predicted audio paths are skipped.

After all samples are scored, corpus WER is printed using masr_eval_pkg.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
import time
from typing import Any, Dict, List, Set

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import parse_asr_output
from masr_eval_pkg import compute_wer

_ASR_TEXT_TAG = "<asr_text>"
_WS_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="English medical ASR eval for Qwen3-ASR.")
    p.add_argument("--jsonl", type=str, required=True, help="Manifest with audio + text fields.")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument(
        "--language",
        type=str,
        default=None,
        help="Forced inference language (default: auto-detect). Use 'None' string for language=None prompting.",
    )
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--context", type=str, default="")
    p.add_argument("--device_map", type=str, default="cuda:0")
    p.add_argument(
        "--output_predictions",
        type=str,
        required=True,
        help="JSONL file of per-utterance predictions. Also used as resume checkpoint.",
    )
    p.add_argument(
        "--output_summary",
        type=str,
        default="",
        help="If set, write final WER summary to this text file.",
    )
    return p.parse_args()


def extract_reference_text(label: str) -> str:
    s = (label or "").strip()
    if not s:
        return ""
    if _ASR_TEXT_TAG in s:
        return s.split(_ASR_TEXT_TAG, 1)[1].strip()
    return s


def normalize_english(text: str) -> str:
    """Standard English ASR normalization: lowercase, strip punctuation, collapse whitespace."""
    s = (text or "").lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = _WS_RE.sub(" ", s).strip()
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
            except json.JSONDecodeError:
                continue
            k = rec.get("audio")
            if k:
                done.add(k)
    return done


def main() -> None:
    args = parse_args()
    rows = load_manifest(args.jsonl, args.max_samples)
    if not rows:
        print("No samples loaded.", file=sys.stderr)
        sys.exit(1)

    for i, ex in enumerate(rows):
        if "audio" not in ex or "text" not in ex:
            print(f"Line {i}: need 'audio' and 'text' fields.", file=sys.stderr)
            sys.exit(1)
        if not os.path.isfile(ex["audio"]):
            print(f"Missing audio file: {ex['audio']}", file=sys.stderr)
            sys.exit(1)

    # Resolve language argument
    lang_arg = args.language
    if lang_arg is not None and lang_arg.strip().lower() == "none":
        lang = None
    else:
        lang = (lang_arg.strip() or None) if lang_arg else None

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    print(f"Loading model from {args.model!r} ...")
    t_load = time.time()
    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=args.device_map,
        max_inference_batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"Model loaded in {time.time() - t_load:.1f}s")

    # Resume: skip already-predicted audio paths
    completed = load_completed_keys(args.output_predictions)
    todo_indices: List[int] = [i for i, ex in enumerate(rows) if ex["audio"] not in completed]
    print(
        f"Total: {len(rows)} | Already done: {len(completed)} | Remaining: {len(todo_indices)} | "
        f"batch_size={args.batch_size} | language={lang!r}"
    )

    parent = os.path.dirname(args.output_predictions)
    if parent:
        os.makedirs(parent, exist_ok=True)

    wf = open(args.output_predictions, "a", encoding="utf-8") if todo_indices else None
    t_start = time.time()
    done_count = len(completed)
    batch_count = 0

    try:
        for b_start in range(0, len(todo_indices), args.batch_size):
            batch_idx = todo_indices[b_start : b_start + args.batch_size]
            batch_audios = [rows[i]["audio"] for i in batch_idx]
            t0 = time.time()
            try:
                outs = model.transcribe(
                    audio=batch_audios,
                    context=args.context or "",
                    language=lang,
                    return_time_stamps=False,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[OOM at batch {b_start // args.batch_size}] retrying one-by-one", file=sys.stderr)
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
                print(f"Batch failed: {e}; falling back to per-sample", file=sys.stderr)
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
                    "keywords": ex.get("keywords", ""),
                }
                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done_count += 1
            wf.flush()
            batch_count += 1

            elapsed = time.time() - t_start
            batch_dt = time.time() - t0
            utt_per_s = (args.batch_size) / max(batch_dt, 1e-6)
            remaining = len(todo_indices) - b_start - len(batch_idx)
            eta_s = remaining / max(utt_per_s, 1e-6)
            print(
                f"[batch {batch_count:5d}] {done_count}/{len(rows)} utt | "
                f"this batch {batch_dt:.2f}s ({utt_per_s:.1f} utt/s) | "
                f"elapsed {elapsed/60:.1f}m | ETA {eta_s/60:.1f}m",
                flush=True,
            )
    finally:
        if wf is not None:
            wf.close()

    # --- Final scoring ---
    print("\n=== Scoring ===")
    preds = []
    with open(args.output_predictions, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            preds.append(json.loads(line))

    # Align by audio path (in case order differs)
    ref_by_audio = {ex["audio"]: extract_reference_text(ex["text"]) for ex in rows}
    refs = []
    hyps = []
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

    refs_norm = [normalize_english(r) for r in refs]
    hyps_norm = [normalize_english(h) for h in hyps]

    wer_result = compute_wer(refs_norm, hyps_norm, per_sample=True)
    per_sample = wer_result.get("per_sample", [])
    exact_matches = sum(1 for s in per_sample if s.get("wer", 1.0) == 0.0)

    total_time = time.time() - t_start
    summary_lines = [
        "=== English Medical ASR Evaluation ===",
        f"Model:               {args.model}",
        f"Manifest:            {args.jsonl}",
        f"Predictions:         {args.output_predictions}",
        f"Samples:             {len(preds)} / {len(rows)}",
        f"Language prompting:  {lang!r}",
        f"Batch size:          {args.batch_size}",
        f"Max new tokens:      {args.max_new_tokens}",
        f"Normalization:       lowercase + strip punctuation + collapse whitespace",
        f"Corpus WER:          {wer_result['wer'] * 100:.2f}%",
        f"Substitutions:       {wer_result['substitutions']}",
        f"Deletions:           {wer_result['deletions']}",
        f"Insertions:          {wer_result['insertions']}",
        f"Ref tokens (total):  {wer_result['n_ref_tokens']}",
        f"Sentence exact acc:  {exact_matches / max(len(per_sample),1) * 100:.2f}% ({exact_matches}/{len(per_sample)})",
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
