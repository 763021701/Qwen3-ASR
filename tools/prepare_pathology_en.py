#!/usr/bin/env python3
"""Prepare the high-quality pathology English manifest for Qwen3-ASR SFT.

The source manifest contains multiple segments from each video. Splitting by
video_id keeps related segments in the same split and avoids video leakage.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare pathology English JSONL for Qwen3-ASR SFT.")
    p.add_argument("--input_jsonl", required=True)
    p.add_argument("--output_dir", default="data/pathology_en")
    p.add_argument("--train_jsonl", default="")
    p.add_argument("--dev_jsonl", default="")
    p.add_argument("--train_ratio", type=float, default=0.95)
    p.add_argument("--dev_ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--check_audio", type=int, default=1, choices=(0, 1))
    return p.parse_args()


def _transcript_body(value: Any) -> str:
    text = str(value or "").strip()
    if "<asr_text>" in text:
        text = text.split("<asr_text>", 1)[1].strip()
    return text


def load_samples(path: str, check_audio: bool) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    samples: List[Dict[str, str]] = []
    stats = {
        "total_lines": 0,
        "kept": 0,
        "invalid_json": 0,
        "missing_audio_field": 0,
        "missing_audio_file": 0,
        "empty_text": 0,
        "missing_video_id": 0,
    }

    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            stats["total_lines"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["invalid_json"] += 1
                continue

            audio = str(record.get("audio") or "").strip()
            text = _transcript_body(record.get("text"))
            video_id = str(record.get("video_id") or "").strip()

            if not audio:
                stats["missing_audio_field"] += 1
                continue
            if check_audio and not os.path.isfile(audio):
                stats["missing_audio_file"] += 1
                continue
            if not text:
                stats["empty_text"] += 1
                continue
            if not video_id:
                video_id = f"__line_{line_no}"
                stats["missing_video_id"] += 1

            samples.append(
                {
                    "audio": os.path.abspath(audio),
                    "text": f"language English<asr_text>{text}",
                    "video_id": video_id,
                }
            )
            stats["kept"] += 1

    return samples, stats


def split_by_video(
    samples: List[Dict[str, str]], train_ratio: float, dev_ratio: float, seed: int
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    if train_ratio <= 0 or dev_ratio < 0 or train_ratio + dev_ratio != 1.0:
        raise ValueError("train_ratio must be > 0 and train_ratio + dev_ratio must equal 1.0")

    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for sample in samples:
        groups[sample["video_id"]].append(sample)

    grouped = list(groups.values())
    random.Random(seed).shuffle(grouped)
    target_train = len(samples) * train_ratio

    train: List[Dict[str, str]] = []
    dev: List[Dict[str, str]] = []
    for group in grouped:
        if len(train) < target_train or not train:
            train.extend(group)
        else:
            dev.extend(group)

    if not dev and train:
        dev.append(train.pop())
    return train, dev


def write_jsonl(path: str, samples: List[Dict[str, str]]) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps({"audio": sample["audio"], "text": sample["text"]}, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    samples, stats = load_samples(args.input_jsonl, check_audio=bool(args.check_audio))
    train, dev = split_by_video(samples, args.train_ratio, args.dev_ratio, args.seed)

    train_path = args.train_jsonl or os.path.join(args.output_dir, "train.jsonl")
    dev_path = args.dev_jsonl or os.path.join(args.output_dir, "dev.jsonl")
    report_path = os.path.join(args.output_dir, "prepare_report.json")

    write_jsonl(train_path, train)
    write_jsonl(dev_path, dev)

    group_ids = {sample["video_id"] for sample in samples}
    train_groups = {sample["video_id"] for sample in train}
    dev_groups = {sample["video_id"] for sample in dev}
    overlap = train_groups & dev_groups
    if overlap:
        raise RuntimeError(f"Video split leakage detected: {len(overlap)} overlapping groups")

    report = {
        "input_jsonl": os.path.abspath(args.input_jsonl),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "dev_ratio": args.dev_ratio,
        "text_prefix": "language English<asr_text>",
        "stats": stats,
        "total_groups": len(group_ids),
        "train_groups": len(train_groups),
        "dev_groups": len(dev_groups),
        "train_samples": len(train),
        "dev_samples": len(dev),
        "train_jsonl": os.path.abspath(train_path),
        "dev_jsonl": os.path.abspath(dev_path),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
