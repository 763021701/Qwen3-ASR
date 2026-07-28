#!/usr/bin/env python3
"""Convert tcga_synthesis_manifest.csv into Qwen3-ASR SFT JSONL.

The CSV stores a relative ``audio_path`` and a clean English ``text``
(synthesized medical speech). This script resolves audio to an absolute path
(relative to ``--audio_root``) and wraps the transcript as
``language {LANG}<asr_text>...`` for ``finetuning/qwen3_asr_sft.py``.

All rows are written to a single output file (no train/dev split): full-training
SFT with no held-out eval during training.
"""

from __future__ import annotations

import argparse
import csv
import json
import os


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_csv", required=True)
    p.add_argument(
        "--audio_root",
        required=True,
        help="Directory that the manifest's relative 'audio_path' is relative to.",
    )
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--language", default="English")
    p.add_argument("--check_audio", type=int, default=1, choices=(0, 1))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stats = {"kept": 0, "missing_audio": 0, "empty_text": 0}

    out_dir = os.path.dirname(os.path.abspath(args.output_jsonl))
    os.makedirs(out_dir, exist_ok=True)

    with open(args.input_csv, "r", encoding="utf-8") as f, open(
        args.output_jsonl, "w", encoding="utf-8"
    ) as out:
        reader = csv.DictReader(f)
        for row in reader:
            audio = (row.get("audio_path") or "").strip()
            text = (row.get("text") or "").strip()

            if not audio:
                stats["missing_audio"] += 1
                continue
            audio_abs = (
                audio
                if os.path.isabs(audio)
                else os.path.abspath(os.path.join(args.audio_root, audio))
            )
            if args.check_audio and not os.path.isfile(audio_abs):
                stats["missing_audio"] += 1
                continue
            if not text:
                stats["empty_text"] += 1
                continue

            out.write(
                json.dumps(
                    {
                        "audio": audio_abs,
                        "text": f"language {args.language}<asr_text>{text}",
                        "aug": 0,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            stats["kept"] += 1

    print(
        json.dumps(
            {
                "input_csv": os.path.abspath(args.input_csv),
                "output_jsonl": os.path.abspath(args.output_jsonl),
                "language": args.language,
                "stats": stats,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
