#!/usr/bin/env python3
"""Build an SFT manifest with unique real audio and a 20-hour synthetic subset."""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--synthetic-hours", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def unique_rows(rows: list[dict], source: str) -> list[dict]:
    """Keep one row per audio path, preserving the first cleaned label."""
    seen: set[str] = set()
    result = []
    for row in rows:
        if row.get("sampling_source") != source:
            continue
        audio = str(row.get("audio") or "")
        if not audio or audio in seen:
            continue
        seen.add(audio)
        result.append(row)
    return result


def main() -> None:
    args = parse_args()
    source_path = Path(args.input)
    output_path = Path(args.output)
    with source_path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    sources = collections.Counter(row.get("sampling_source") for row in rows)
    expected = {"real_target_domain", "tcga_reports_en", "silence"}
    unknown = set(sources) - expected
    if unknown:
        raise ValueError(f"Unexpected sampling_source values: {sorted(unknown)}")

    real_rows = unique_rows(rows, "real_target_domain")
    synthetic_rows = unique_rows(rows, "tcga_reports_en") + unique_rows(rows, "silence")
    target_sec = max(0.0, float(args.synthetic_hours) * 3600.0)
    rng = random.Random(args.seed)
    rng.shuffle(synthetic_rows)
    selected_synthetic = []
    selected_sec = 0.0
    if target_sec > 0:
        for row in synthetic_rows:
            selected_synthetic.append(row)
            selected_sec += float(row.get("duration_sec") or 0.0)
            if selected_sec >= target_sec:
                break
    if selected_sec < target_sec:
        raise ValueError(
            f"Only {selected_sec / 3600.0:.3f} synthetic hours available; "
            f"requested {args.synthetic_hours:.3f}"
        )

    output_rows = real_rows + selected_synthetic
    rng.shuffle(output_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_source = collections.defaultdict(lambda: {"rows": 0, "duration_sec": 0.0})
    for row in output_rows:
        source = row["sampling_source"]
        by_source[source]["rows"] += 1
        by_source[source]["duration_sec"] += float(row.get("duration_sec") or 0.0)
    print(json.dumps({
        "input": str(source_path.resolve()),
        "output": str(output_path.resolve()),
        "seed": args.seed,
        "synthetic_target_hours": args.synthetic_hours,
        "sources_in_input": dict(sources),
        "output_rows": len(output_rows),
        "output_unique_audio": len({row["audio"] for row in output_rows}),
        "by_source": {
            source: {
                "rows": stats["rows"],
                "duration_sec": round(stats["duration_sec"], 3),
                "duration_hours": round(stats["duration_sec"] / 3600.0, 3),
            }
            for source, stats in sorted(by_source.items())
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
