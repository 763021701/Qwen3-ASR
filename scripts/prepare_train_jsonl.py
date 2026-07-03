#!/usr/bin/env python3
# coding=utf-8
"""Convert a MAGICDATA-Hybrid metadata.csv into JSONL for qwen3_asr_sft.py.

qwen3_asr_sft.py loads JSONL with fields ``audio`` (wav path) and ``text``
(transcript). The hybrid CSV instead has ``audio_path`` / ``Transcription`` plus
``split`` / ``dataset`` columns, so this script:
  - keeps only the requested ``split`` (default ``train``),
  - renames audio_path -> audio, Transcription -> text,
  - drops rows with empty audio/text,
  - optionally skips rows whose wav is missing on disk (default on) so a missing
    file can't crash training mid-epoch.

Output is the raw transcript (no ``language X<asr_text>`` prefix); the CTC TN
layer routes Chinese via the CJK heuristic.
"""
import argparse
import csv
import json
import os

from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="metadata.csv path")
    ap.add_argument("--split", default="train", help="split value to keep")
    ap.add_argument("--out", required=True, help="output jsonl path")
    ap.add_argument(
        "--skip_missing_check", type=int, default=0,
        help="1=don't stat wav files (faster, but missing audio crashes training)",
    )
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n_kept = n_missing = n_empty = 0
    with open(args.csv, encoding="utf-8", newline="") as f, \
         open(args.out, "w", encoding="utf-8") as g:
        for row in tqdm(csv.DictReader(f), desc="convert"):
            if (row.get("split") or "").strip() != args.split:
                continue
            audio = (row.get("audio_path") or "").strip()
            text = (row.get("Transcription") or "").strip()
            if not audio or not text:
                n_empty += 1
                continue
            if not args.skip_missing_check and not os.path.exists(audio):
                n_missing += 1
                continue
            g.write(json.dumps({"audio": audio, "text": text}, ensure_ascii=False) + "\n")
            n_kept += 1

    print(
        f"split={args.split} kept={n_kept} "
        f"empty_dropped={n_empty} missing_dropped={n_missing} -> {args.out}"
    )


if __name__ == "__main__":
    main()
