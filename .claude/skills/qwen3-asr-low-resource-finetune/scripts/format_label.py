#!/usr/bin/env python3
# coding=utf-8
"""
Build the Qwen3-ASR supervised `text` field for one transcript line.

Does NOT normalize transcript body (punctuation, case, numbers, etc.). Before calling
this helper, normalize raw text per docs/normalize_label.md (see skill
qwen3-asr-low-resource-finetune).

Use from shell for quick checks, or import `qwen3_asr_supervised_text` in dataset prep scripts.

Example:
  python scripts/format_label.py --language English --transcript "hello world"
"""

from __future__ import annotations

import argparse

from qwen_asr.inference.utils import normalize_language_spec, validate_language_spec


def qwen3_asr_supervised_text(language: str, transcript: str) -> str:
    """Match prepare_*_qwen3 scripts (e.g. evaluation/chinese/wsc/, evaluation/cantonese/wsyue_asr/)."""
    lang = (language or "").strip()
    if not lang:
        raise ValueError("language is empty")
    spec = normalize_language_spec(lang)
    validate_language_spec(spec)
    t = transcript if transcript is not None else ""
    return f"language {spec}<asr_text>{t}"


def main() -> None:
    p = argparse.ArgumentParser(description="Format one Qwen3-ASR jsonl `text` value.")
    p.add_argument("--language", type=str, required=True, help="e.g. Uyghur, Cantonese, None, or Chinese,English")
    p.add_argument("--transcript", type=str, default="", help="Raw reference transcript")
    args = p.parse_args()
    print(qwen3_asr_supervised_text(args.language, args.transcript))


if __name__ == "__main__":
    main()
