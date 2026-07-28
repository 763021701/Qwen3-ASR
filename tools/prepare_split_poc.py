#!/usr/bin/env python3
"""Convert Test_Samples segments_filter.jsonl to Qwen3-ASR JSONL, splitting
train/test by POC-July source files.

Segments whose ``source_audio`` basename stem matches a top-level ``*.wav`` in
``--poc_dir`` are held out as TEST; the rest are TRAIN. Output is Qwen3-ASR
format (absolute audio path + ``language {LANG}<asr_text>...``). Matching is
case-insensitive on the stem (e.g. 26ss11731 == 26SS11731).
"""

from __future__ import annotations

import argparse
import glob
import json
import os


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_jsonl", required=True)
    p.add_argument("--audio_root", required=True)
    p.add_argument("--poc_dir", required=True, help="Dir of POC-July source wavs (test set sources).")
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--test_jsonl", required=True)
    p.add_argument("--language", default="English")
    p.add_argument("--check_audio", type=int, default=1, choices=(0, 1))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    test_ids = {
        os.path.splitext(os.path.basename(w))[0].lower()
        for w in glob.glob(os.path.join(args.poc_dir, "*.wav"))
    }
    for p in (args.train_jsonl, args.test_jsonl):
        d = os.path.dirname(os.path.abspath(p))
        os.makedirs(d, exist_ok=True)

    stats = {"train": 0, "test": 0, "missing_audio": 0, "empty_text": 0, "test_ids": sorted(test_ids)}
    wfs = {
        "train": open(args.train_jsonl, "w", encoding="utf-8"),
        "test": open(args.test_jsonl, "w", encoding="utf-8"),
    }
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            rec = json.loads(line)
            audio = str(rec.get("audio") or "").strip()
            text = str(rec.get("text") or "").strip()
            if "<asr_text>" in text:
                text = text.split("<asr_text>", 1)[1].strip()
            src = str(rec.get("source_audio") or "")
            stem = os.path.splitext(os.path.basename(src))[0].lower()
            split = "test" if stem in test_ids else "train"

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
            wfs[split].write(
                json.dumps(
                    {"audio": audio_abs, "text": f"language {args.language}<asr_text>{text}"},
                    ensure_ascii=False,
                )
                + "\n"
            )
            stats[split] += 1
    for wf in wfs.values():
        wf.close()

    print(
        json.dumps(
            {
                "input_jsonl": os.path.abspath(args.input_jsonl),
                "train_jsonl": os.path.abspath(args.train_jsonl),
                "test_jsonl": os.path.abspath(args.test_jsonl),
                "stats": stats,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
