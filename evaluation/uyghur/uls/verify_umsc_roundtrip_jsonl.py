#!/usr/bin/env python3
# coding=utf-8
"""
Round-trip Uyghur script conversion on a Qwen3-style jsonl manifest using umsc:
  UAS (Arabic) -> ULS (Latin) -> UAS, then compare to the original Arabic segment.

Requires: pip install umsc

Example:
  python evaluation/uyghur/uls/verify_umsc_roundtrip_jsonl.py --jsonl data/uyghur/common_voice/ug_test_qwen3.jsonl
  python evaluation/uyghur/uls/verify_umsc_roundtrip_jsonl.py --jsonl data/uyghur/common_voice/ug_test_qwen3.jsonl --max_mismatches 50
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from typing import Any, Dict, Tuple

_ASR_TEXT_TAG = "<asr_text>"
_ZW_RE = re.compile(r"[\u200c\u200d\ufeff]")


def extract_asr_text(label: str) -> str:
    """Strip prefix like 'language Uyghur<asr_text>...'."""
    s = (label or "").strip()
    if not s:
        return ""
    if _ASR_TEXT_TAG in s:
        return s.split(_ASR_TEXT_TAG, 1)[1].strip()
    return s


def normalize_for_compare(s: str) -> str:
    """NFC, drop common zero-width chars, collapse whitespace (same idea as ASR eval)."""
    s = unicodedata.normalize("NFC", (s or "").strip())
    s = _ZW_RE.sub("", s)
    s = " ".join(s.split())
    return s


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="umsc UAS->ULS->UAS round-trip check on Qwen3 jsonl.")
    p.add_argument("--jsonl", type=str, required=True, help="Input manifest (audio + text with <asr_text>).")
    p.add_argument("--max_samples", type=int, default=0, help="If >0, only first N non-empty lines.")
    p.add_argument(
        "--max_mismatches",
        type=int,
        default=20,
        help="Print up to this many mismatch examples (0 = none).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from umsc import UgMultiScriptConverter
    except ImportError:
        print("Missing dependency: pip install umsc", file=sys.stderr)
        return 1

    to_uls = UgMultiScriptConverter("UAS", "ULS")
    to_uas = UgMultiScriptConverter("ULS", "UAS")

    total_lines = 0
    empty_ref = 0
    converted = 0
    exact_ok = 0
    norm_ok = 0
    mismatches: list[Tuple[int, str, str, str, str]] = []

    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                row: Dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Line {line_no}: invalid JSON: {e}", file=sys.stderr)
                return 1
            ref = extract_asr_text(str(row.get("text", "")))
            if not ref:
                empty_ref += 1
                continue

            if args.max_samples > 0 and converted >= args.max_samples:
                break

            latin = to_uls(ref)
            back = to_uas(latin)
            converted += 1

            n_ref = normalize_for_compare(ref)
            n_back = normalize_for_compare(back)

            if ref == back:
                exact_ok += 1
                norm_ok += 1
            else:
                if n_ref == n_back:
                    norm_ok += 1
                if args.max_mismatches <= 0 or len(mismatches) < args.max_mismatches:
                    mismatches.append((line_no, ref, latin, back, row.get("audio", "")))

    print("umsc round-trip (UAS -> ULS -> UAS)")
    print(f"  jsonl: {args.jsonl}")
    print(f"  lines read: {total_lines}")
    print(f"  empty after <asr_text> strip: {empty_ref}")
    print(f"  samples converted: {converted}")
    if converted == 0:
        print("  nothing to convert.")
        return 0
    print(f"  exact string match: {exact_ok} / {converted} ({100.0 * exact_ok / converted:.2f}%)")
    print(f"  match after NFC+ZW normalize: {norm_ok} / {converted} ({100.0 * norm_ok / converted:.2f}%)")

    if mismatches:
        print(f"\nFirst {len(mismatches)} mismatch(es) (line_no, audio, ref, latin, roundtrip):")
        for line_no, ref, latin, back, audio in mismatches:
            print("---")
            print(f"  line: {line_no}")
            if audio:
                print(f"  audio: {audio}")
            print(f"  ref:        {ref}")
            print(f"  uls:        {latin}")
            print(f"  roundtrip:  {back}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
