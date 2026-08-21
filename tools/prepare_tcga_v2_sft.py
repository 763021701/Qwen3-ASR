#!/usr/bin/env python3
"""Build an SFT manifest from TCGA v2 synthetic audio and repeated real audio."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-metadata", required=True)
    parser.add_argument("--real-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--real-repeat", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    if args.real_repeat < 1:
        raise ValueError("--real-repeat must be at least 1")

    synthetic_rows = load_jsonl(Path(args.synthetic_metadata))
    real_rows = load_jsonl(Path(args.real_jsonl))
    output_rows: list[dict] = []

    for row in synthetic_rows:
        audio = Path(str(row.get("audio_path") or "")).resolve()
        text = str(row.get("text") or "").strip()
        if not audio.is_file():
            raise FileNotFoundError(f"Missing synthetic audio: {audio}")
        if not text:
            raise ValueError(f"Empty synthetic transcript: {row}")
        output_rows.append(
            {
                "audio": str(audio),
                "text": f"language English<asr_text>{text}",
                "aug": 1,
                "sampling_source": "tcga_reports_en_v2",
                "source_group": str(row.get("speaker") or row.get("index") or "unknown"),
                "duration_sec": round(float(row.get("duration") or 0.0), 6),
            }
        )

    for repeat_index in range(args.real_repeat):
        for row in real_rows:
            audio = Path(str(row.get("audio") or "")).resolve()
            text = str(row.get("text") or "").strip()
            if not audio.is_file():
                raise FileNotFoundError(f"Missing real audio: {audio}")
            if not text:
                raise ValueError(f"Empty real transcript: {row}")
            output_rows.append(
                {
                    "audio": str(audio),
                    "text": text,
                    "aug": 1,
                    "sampling_source": "real_target_domain",
                    "source_group": str(row.get("source_group") or "unknown"),
                    "duration_sec": round(float(row.get("duration_sec") or 0.0), 6),
                    "repeat_index": repeat_index,
                }
            )

    random.Random(args.seed).shuffle(output_rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_source: dict[str, dict[str, float]] = {}
    for row in output_rows:
        stats = by_source.setdefault(row["sampling_source"], {"rows": 0, "duration_sec": 0.0})
        stats["rows"] += 1
        stats["duration_sec"] += row["duration_sec"]
    print(json.dumps({
        "output": str(output_path.resolve()),
        "rows": len(output_rows),
        "unique_audio": len({row["audio"] for row in output_rows}),
        "real_repeat": args.real_repeat,
        "by_source": {
            source: {
                "rows": int(stats["rows"]),
                "duration_hours": round(stats["duration_sec"] / 3600.0, 3),
            }
            for source, stats in sorted(by_source.items())
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
