#!/usr/bin/env python3
# coding=utf-8
"""
Convert Qwen3-style jsonl text from Uyghur Arabic Script (UAS) to Uyghur Latin Script (ULS)
using umsc, while preserving the "language ...<asr_text>" prefix.

Requires:
  pip install umsc

Examples:
  python evaluation/uyghur/uls/convert_umsc_uas_to_uls_jsonl.py \
    --inputs data/uyghur/common_voice/ug_dev_qwen3.jsonl data/uyghur/common_voice/ug_test_qwen3.jsonl data/uyghur/common_voice/ug_train_qwen3.jsonl

  python evaluation/uyghur/uls/convert_umsc_uas_to_uls_jsonl.py \
    --inputs data/uyghur/common_voice/ug_test_qwen3.jsonl \
    --output_dir data/uyghur/common_voice \
    --output_suffix _latin
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

ASR_TEXT_TAG = "<asr_text>"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch convert Qwen3 jsonl text from UAS to ULS with umsc."
    )
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input jsonl file paths.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Optional output directory. Default: same directory as each input.",
    )
    p.add_argument(
        "--output_suffix",
        type=str,
        default="_uls",
        help="Suffix added before .jsonl for output files.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    return p.parse_args()


def split_prefix_and_content(text: str) -> Tuple[str, str]:
    """
    Split into:
      - prefix: possibly 'language X<asr_text>'
      - content: text after <asr_text> (or whole text if tag missing)
    """
    s = str(text or "")
    if ASR_TEXT_TAG in s:
        left, right = s.split(ASR_TEXT_TAG, 1)
        return f"{left}{ASR_TEXT_TAG}", right
    return "", s


def build_output_path(in_path: Path, output_dir: str, output_suffix: str) -> Path:
    out_dir = Path(output_dir) if output_dir else in_path.parent
    return out_dir / f"{in_path.stem}{output_suffix}{in_path.suffix}"


def convert_one_file(in_path: Path, out_path: Path, converter: Any, overwrite: bool) -> Tuple[int, int]:
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists, use --overwrite to replace: {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    converted = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                row: Dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {in_path}:{line_no}: {exc}") from exc

            text = str(row.get("text", ""))
            prefix, content = split_prefix_and_content(text)
            if content:
                content = converter(content)
                converted += 1
            row["text"] = f"{prefix}{content}" if prefix else content

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    return total, converted


def main() -> int:
    args = parse_args()
    try:
        from umsc import UgMultiScriptConverter
    except ImportError:
        print("Missing dependency: pip install umsc", file=sys.stderr)
        return 1

    converter = UgMultiScriptConverter("UAS", "ULS")

    ok = 0
    failed = 0
    for p in args.inputs:
        in_path = Path(p)
        out_path = build_output_path(in_path, args.output_dir, args.output_suffix)
        try:
            total, converted = convert_one_file(in_path, out_path, converter, args.overwrite)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[FAILED] {in_path}: {exc}", file=sys.stderr)
            continue

        ok += 1
        print(f"[DONE] {in_path} -> {out_path}")
        print(f"       lines: {total}, converted_text_rows: {converted}")

    print(f"Summary: success={ok}, failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
