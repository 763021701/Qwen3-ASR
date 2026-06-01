#!/usr/bin/env python3
# coding=utf-8
"""Convert medical TTS metadata.csv to Qwen3-ASR JSONL format with English normalization."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.normalize_transcript import normalize_english
from qwen_asr.inference.utils import normalize_language_spec, validate_language_spec


def qwen3_asr_supervised_text(language: str, transcript: str) -> str:
    lang = (language or "").strip()
    if not lang:
        raise ValueError("language is empty")
    spec = normalize_language_spec(lang)
    validate_language_spec(spec)
    t = transcript if transcript is not None else ""
    return f"language {spec}<asr_text>{t}"


def main():
    p = argparse.ArgumentParser(description="Convert medical TTS metadata.csv to Qwen3-ASR JSONL")
    p.add_argument("--metadata_csv", type=str, required=True, help="Path to metadata.csv")
    p.add_argument("--output_train", type=str, required=True, help="Output train JSONL path")
    p.add_argument("--output_dev", type=str, required=True, help="Output dev JSONL path")
    p.add_argument("--language", type=str, default="English", help="Language label")
    p.add_argument("--train_ratio", type=float, default=0.95, help="Train split ratio")
    p.add_argument("--split_seed", type=int, default=42, help="Split random seed")
    p.add_argument("--max_samples", type=int, default=0, help="Max total samples (0 = all)")
    args = p.parse_args()

    # Read CSV
    rows = []
    with open(args.metadata_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio = (row.get("audio_path") or "").strip()
            text = (row.get("text") or "").strip()
            if not audio or not text:
                continue
            rows.append((audio, text))

    if args.max_samples > 0 and len(rows) > args.max_samples:
        rows = rows[: args.max_samples]

    print(f"Loaded {len(rows)} rows from {args.metadata_csv}")

    # Shuffle and split
    rng = random.Random(args.split_seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    split = int(len(rows) * args.train_ratio)
    train_idx = set(indices[:split])
    dev_idx = set(indices[split:])

    skipped_missing = 0
    skipped_empty = 0

    def write_jsonl(output_path, split_idx):
        nonlocal skipped_missing, skipped_empty
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for i in sorted(split_idx):
                audio_path, raw_text = rows[i]
                if not os.path.isfile(audio_path):
                    skipped_missing += 1
                    continue
                normalized = normalize_english(raw_text)
                if not normalized:
                    skipped_empty += 1
                    continue
                supervised = qwen3_asr_supervised_text(args.language, normalized)
                obj = {"audio": os.path.abspath(audio_path), "text": supervised}
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    write_jsonl(args.output_train, train_idx)
    write_jsonl(args.output_dev, dev_idx)

    print(f"Train: {args.output_train} ({len(train_idx) - skipped_missing - skipped_empty} samples)")
    print(f"Dev:   {args.output_dev} ({len(dev_idx) - skipped_missing - skipped_empty} samples)")
    if skipped_missing:
        print(f"Skipped missing audio: {skipped_missing}")
    if skipped_empty:
        print(f"Skipped empty after normalize: {skipped_empty}")


if __name__ == "__main__":
    main()
