#!/usr/bin/env python3
"""Filter a GRPO JSONL manifest to one declared sampling source."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input GRPO JSONL manifest.")
    parser.add_argument("--output", required=True, help="Filtered GRPO JSONL manifest.")
    parser.add_argument("--sampling-source", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("--output must differ from --input")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    seen_audio: set[str] = set()
    with input_path.open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("sampling_source") != args.sampling_source:
                continue
            audio = str(row.get("audio", ""))
            if not audio:
                raise ValueError(f"{input_path}:{line_number} has no audio path")
            if audio in seen_audio:
                raise ValueError(f"{input_path}:{line_number} repeats audio {audio!r}")
            seen_audio.add(audio)
            counts[str(row.get("grpo_category", "unknown"))] += 1
            target.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"[done] rows={sum(counts.values())} unique_audio={len(seen_audio)} "
        f"sampling_source={args.sampling_source} categories={dict(sorted(counts.items()))}"
    )


if __name__ == "__main__":
    main()
