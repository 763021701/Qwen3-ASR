#!/usr/bin/env python3
"""
Convert yidu medical TTS assignments.jsonl to Qwen3-ASR eval JSONL format.

Input:  assignments.jsonl (index, text, ref_audio, originalText, entities, ...)
Output: eval JSONL with {audio, text, index, entities, originalText}

Usage:
  python prepare_chinese_medical_jsonl.py \
    --source /path/to/yidu_wenet_cosyvoice/assignments.jsonl \
    --output evaluation/chinese_medical/yidu_wenet_cosyvoice_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Convert yidu medical assignments to eval JSONL.")
    parser.add_argument("--source", required=True, help="Path to assignments.jsonl")
    parser.add_argument("--output", required=True, help="Output eval JSONL path")
    args = parser.parse_args()

    # Determine wav directory (sibling to assignments.jsonl)
    wav_dir = os.path.join(os.path.dirname(args.source), "wavs")
    if not os.path.isdir(wav_dir):
        print(f"ERROR: wav directory not found: {wav_dir}", file=sys.stderr)
        sys.exit(1)

    n_written = 0
    n_missing = 0
    with open(args.source, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            idx = d.get("index")
            if idx is None:
                print(f"Warning: missing index in line, skipping", file=sys.stderr)
                continue

            wav = os.path.join(wav_dir, f"{idx:06d}.wav")
            if not os.path.isfile(wav):
                n_missing += 1
                continue

            rec = {
                "audio": wav,
                "text": d["text"],
                "index": idx,
                "originalText": d.get("originalText", ""),
                "entities": d.get("entities", []),
                "source_file": d.get("source_file", ""),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Wrote {n_written} rows to {args.output}")
    if n_missing:
        print(f"Warning: {n_missing} missing wav files")


if __name__ == "__main__":
    main()
