#!/usr/bin/env python3
# coding=utf-8

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from qwen_asr.inference.utils import normalize_language_spec
from tools.normalize_transcript import normalize_transcript


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an audio/text youtube_english JSONL to Qwen3-ASR SFT JSONL."
    )
    parser.add_argument("--input_jsonl", required=True, help="Input JSONL with at least audio and text fields.")
    parser.add_argument("--output_jsonl", required=True, help="Output Qwen3-ASR finetuning JSONL.")
    parser.add_argument("--report_json", default="", help="Optional JSON report path.")
    parser.add_argument("--max_samples", type=int, default=0, help="If >0, write at most this many valid samples.")
    parser.add_argument(
        "--language",
        default="None",
        help='Language prefix for Qwen3-ASR labels, e.g. "None" or "English".',
    )
    parser.add_argument(
        "--normalize_transcript",
        type=int,
        default=1,
        choices=(0, 1),
        help="Apply docs/normalize_label.md English normalization before writing labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    language = normalize_language_spec(args.language)

    stats: Dict[str, Any] = {
        "source": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "language": language,
        "normalize_transcript": bool(args.normalize_transcript),
        "total_lines": 0,
        "written": 0,
        "skipped_invalid_json": 0,
        "skipped_missing_audio": 0,
        "skipped_empty_text": 0,
        "skipped_empty_after_normalize": 0,
        "skipped_missing_audio_examples": [],
        "skipped_invalid_json_examples": [],
    }

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_no, raw in enumerate(fin, start=1):
            raw = raw.strip()
            if not raw:
                continue

            stats["total_lines"] += 1
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                stats["skipped_invalid_json"] += 1
                if len(stats["skipped_invalid_json_examples"]) < 5:
                    stats["skipped_invalid_json_examples"].append({"line": line_no, "error": str(exc)})
                continue

            audio = str(item.get("audio") or "").strip()
            text = str(item.get("text") or "").strip()
            if not text:
                stats["skipped_empty_text"] += 1
                continue
            if not audio or not os.path.isfile(audio):
                stats["skipped_missing_audio"] += 1
                if len(stats["skipped_missing_audio_examples"]) < 5:
                    stats["skipped_missing_audio_examples"].append({"line": line_no, "audio": audio})
                continue
            if args.normalize_transcript:
                text = normalize_transcript(text, "english")
                if not text:
                    stats["skipped_empty_after_normalize"] += 1
                    continue

            record = {
                "audio": os.path.abspath(audio),
                "text": f"language {language}<asr_text>{text}",
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["written"] += 1

            if args.max_samples > 0 and stats["written"] >= args.max_samples:
                break

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
