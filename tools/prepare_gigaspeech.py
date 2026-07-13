#!/usr/bin/env python3
# coding=utf-8
"""Prepare GigaSpeech (pre-normalized) metadata for Qwen3-ASR SFT.

Input: a jsonl with one ``{"wav_path", "text"}`` per line, where ``text`` is
already transcript-normalized (case, punctuation, token tags handled) by the
user. This script only:
  1. Builds the supervised ``text`` field: ``language {Spec}<asr_text>{transcript}``.
  2. Shuffles (seeded) and splits into train / dev / test jsonl.

It does NOT re-normalize the transcript body. Re-normalization, if needed, must
happen upstream (see docs/normalize_label.md).

Usage:
    python tools/prepare_gigaspeech.py \
        --input_jsonl /path/to/m_all_metadata_normalized.jsonl \
        --output_dir data/gigaspeech \
        --language English \
        --train_ratio 0.98 --dev_ratio 0.01 --split_seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List

from qwen_asr.inference.utils import normalize_language_spec, validate_language_spec


def build_supervised_text(language_spec: str, transcript: str) -> str:
    """``language {Spec}<asr_text>{transcript}`` - matches scripts/format_label.py."""
    t = transcript if transcript is not None else ""
    return f"language {language_spec}<asr_text>{t}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Prepare GigaSpeech (normalized) for Qwen3-ASR SFT")
    p.add_argument("--input_jsonl", type=str, required=True,
                   help="Pre-normalized jsonl with {wav_path, text} per line.")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to write train/dev/test.jsonl.")
    p.add_argument("--language", type=str, default="English",
                   help="Language spec (single or comma-separated). Default: English.")
    p.add_argument("--train_ratio", type=float, default=0.98)
    p.add_argument("--dev_ratio", type=float, default=0.01,
                   help="Test ratio = 1 - train_ratio - dev_ratio.")
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--max_samples", type=int, default=0,
                   help="0 = all; else cap total records read before split.")
    p.add_argument("--check_audio", type=int, default=0, choices=(0, 1),
                   help="If 1, verify each wav_path exists (slow for large sets).")
    p.add_argument("--report_json", type=str, default="",
                   help="Optional path to write a JSON report of counts.")
    return p.parse_args()


def read_records(path: str, language_spec: str, check_audio: bool) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    skipped_empty = 0
    missing_audio = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                skipped_empty += 1
                continue
            wav_path = o.get("wav_path", "")
            transcript = (o.get("text") or "").strip()
            if not transcript:
                skipped_empty += 1
                continue
            if check_audio and wav_path and not os.path.isfile(wav_path):
                missing_audio += 1
                continue
            records.append({
                "audio": os.path.abspath(wav_path) if wav_path else "",
                "text": build_supervised_text(language_spec, transcript),
            })
    return records, skipped_empty, missing_audio


def main() -> None:
    args = parse_args()

    spec = normalize_language_spec(args.language.strip())
    validate_language_spec(spec)

    if args.train_ratio <= 0 or args.dev_ratio < 0 or args.train_ratio + args.dev_ratio >= 1.0:
        raise ValueError(
            f"Invalid ratios: train={args.train_ratio} dev={args.dev_ratio} "
            f"(train+dev must be < 1.0)."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[read] {args.input_jsonl}")
    records, skipped_empty, missing_audio = read_records(
        args.input_jsonl, spec, bool(args.check_audio)
    )
    total = len(records)
    print(f"[read] kept={total} skipped_empty={skipped_empty} missing_audio={missing_audio}")

    if args.max_samples > 0 and total > args.max_samples:
        # Deterministic subsample (seeded) before split.
        rng = random.Random(args.split_seed)
        rng.shuffle(records)
        records = records[: args.max_samples]
        total = len(records)
        print(f"[subsample] capped to {total} records")

    rng = random.Random(args.split_seed)
    rng.shuffle(records)

    n_train = int(total * args.train_ratio)
    n_dev = int(total * args.dev_ratio)
    n_test = total - n_train - n_dev

    splits = {
        "train": records[:n_train],
        "dev": records[n_train:n_train + n_dev],
        "test": records[n_train + n_dev:],
    }

    report: Dict[str, Any] = {
        "input_jsonl": args.input_jsonl,
        "language": spec,
        "total_kept": total,
        "skipped_empty": skipped_empty,
        "missing_audio": missing_audio,
        "split_seed": args.split_seed,
        "splits": {},
    }

    for name, items in splits.items():
        out_path = os.path.join(args.output_dir, f"{name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as fout:
            for r in items:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[write] {name}: {len(items)} -> {out_path}")
        report["splits"][name] = {"count": len(items), "path": out_path}

    if splits["train"]:
        print(f"[sample] {json.dumps(splits['train'][0], ensure_ascii=False)[:200]}")

    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[report] {args.report_json}")


if __name__ == "__main__":
    main()
