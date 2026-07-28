#!/usr/bin/env python3
"""Build a deterministic 1:1 real/synthetic Qwen3-ASR training manifest."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real_jsonl", required=True)
    parser.add_argument("--synthetic_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--report_json", required=True)
    parser.add_argument("--target_per_source", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check_audio", type=int, default=1, choices=(0, 1))
    return parser.parse_args()


def load_jsonl(path: Path, check_audio: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            audio = str(row.get("audio") or "").strip()
            text = str(row.get("text") or "").strip()
            if not audio or not text:
                raise ValueError(f"{path}:{line_number}: missing audio or text")
            if check_audio and not os.path.isfile(audio):
                raise FileNotFoundError(f"{path}:{line_number}: missing audio: {audio}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No valid rows in {path}")
    return rows


def sample_uniformly(
    rows: list[dict[str, Any]], target: int, rng: random.Random
) -> list[dict[str, Any]]:
    if target <= 0:
        raise ValueError("target_per_source must be positive")
    if len(rows) >= target:
        return rng.sample(rows, target)

    sampled: list[dict[str, Any]] = []
    cycles, remainder = divmod(target, len(rows))
    for _ in range(cycles):
        cycle = list(rows)
        rng.shuffle(cycle)
        sampled.extend(cycle)
    tail = list(rows)
    rng.shuffle(tail)
    sampled.extend(tail[:remainder])
    return sampled


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    real_rows = load_jsonl(Path(args.real_jsonl), bool(args.check_audio))
    synthetic_rows = load_jsonl(Path(args.synthetic_jsonl), bool(args.check_audio))

    real_sampled = sample_uniformly(real_rows, args.target_per_source, rng)
    synthetic_sampled = sample_uniformly(synthetic_rows, args.target_per_source, rng)

    combined = [dict(row, sampling_source="real") for row in real_sampled]
    combined.extend(dict(row, sampling_source="synthetic") for row in synthetic_sampled)
    rng.shuffle(combined)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in combined:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    real_repetitions = Counter(str(row["audio"]) for row in real_sampled)
    report = {
        "seed": args.seed,
        "target_per_source": args.target_per_source,
        "input": {
            "real_jsonl": str(Path(args.real_jsonl).resolve()),
            "real_rows": len(real_rows),
            "synthetic_jsonl": str(Path(args.synthetic_jsonl).resolve()),
            "synthetic_rows": len(synthetic_rows),
        },
        "output": {
            "train_jsonl": str(output_path.resolve()),
            "total_rows": len(combined),
            "real_rows": len(real_sampled),
            "synthetic_rows": len(synthetic_sampled),
            "real_unique_audio": len(real_repetitions),
            "real_min_repetitions": min(real_repetitions.values()),
            "real_max_repetitions": max(real_repetitions.values()),
        },
    }
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
