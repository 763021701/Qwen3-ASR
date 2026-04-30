#!/usr/bin/env python3
# coding=utf-8
"""
Build the Qwen3-ASR supervised `text` field for one transcript line.

Use from shell for quick checks, or import `qwen3_asr_supervised_text` in dataset prep scripts.

Example:
  python scripts/format_label.py --language Uyghur --transcript "بىر مىسال"
"""

from __future__ import annotations

import argparse


def qwen3_asr_supervised_text(language: str, transcript: str) -> str:
    """Match prepare_*_qwen3 scripts (e.g. evaluation/chinese/wsc/, evaluation/cantonese/wsyue_asr/)."""
    lang = (language or "").strip()
    if not lang:
        raise ValueError("language is empty")
    canon = lang[:1].upper() + lang[1:].lower()
    t = transcript if transcript is not None else ""
    return f"language {canon}<asr_text>{t}"


def main() -> None:
    p = argparse.ArgumentParser(description="Format one Qwen3-ASR jsonl `text` value.")
    p.add_argument("--language", type=str, required=True, help="e.g. Uyghur, Cantonese")
    p.add_argument("--transcript", type=str, default="", help="Raw reference transcript")
    args = p.parse_args()
    print(qwen3_asr_supervised_text(args.language, args.transcript))


if __name__ == "__main__":
    main()
