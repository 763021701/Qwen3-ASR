#!/usr/bin/env python3
"""Convert a Test_Samples segments_filter.jsonl into Qwen3-ASR SFT JSONL.

The source manifest stores a relative ``audio`` path and a bare ``text``
transcript. This script resolves audio to an absolute path (relative to
``--audio_root``) and wraps the transcript as ``language {LANG}<asr_text>...``
so it can be fed directly to ``finetuning/qwen3_asr_sft.py``.

All kept samples are written to a single output file (no train/dev split):
the goal is full-training-only SFT with no held-out eval during training.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_jsonl", required=True)
    p.add_argument(
        "--audio_root",
        required=True,
        help="Directory that the manifest's relative 'audio' field is relative to.",
    )
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--language", default="English")
    p.add_argument("--check_audio", type=int, default=1, choices=(0, 1))
    return p.parse_args()


def _transcript_body(value: Any) -> str:
    text = str(value or "").strip()
    if "<asr_text>" in text:
        text = text.split("<asr_text>", 1)[1].strip()
    return text


def main() -> None:
    args = parse_args()
    stats = {"kept": 0, "missing_audio": 0, "empty_text": 0, "invalid_json": 0}

    out_dir = os.path.dirname(os.path.abspath(args.output_jsonl))
    os.makedirs(out_dir, exist_ok=True)

    with open(args.input_jsonl, "r", encoding="utf-8") as f, open(
        args.output_jsonl, "w", encoding="utf-8"
    ) as out:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                stats["invalid_json"] += 1
                continue

            audio = str(rec.get("audio") or "").strip()
            text = _transcript_body(rec.get("text"))

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
                    {"audio": audio_abs, "text": f"language {args.language}<asr_text>{text}"},
                    ensure_ascii=False,
                )
                + "\n"
            )
            stats["kept"] += 1

    print(
        json.dumps(
            {
                "input_jsonl": os.path.abspath(args.input_jsonl),
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
