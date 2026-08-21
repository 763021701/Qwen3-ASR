#!/usr/bin/env python3
"""Build a three-source Qwen3-ASR manifest balanced by transcript characters."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


LABEL_RE = re.compile(r"^language\s+[^<]+<asr_text>(.*)$", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real_jsonl", required=True)
    parser.add_argument("--synthetic_jsonl", required=True)
    parser.add_argument("--medical_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--report_json", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real_max_repeats", type=int, default=10)
    parser.add_argument("--check_audio", type=int, default=1, choices=(0, 1))
    return parser.parse_args()


def transcript_from_text(value: Any) -> str:
    text = str(value or "").strip()
    match = LABEL_RE.match(text)
    return (match.group(1) if match else text).strip()


def load_rows(path: Path, check_audio: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            audio = str(row.get("audio") or "").strip()
            transcript = transcript_from_text(row.get("text"))
            if not audio or not transcript:
                raise ValueError(f"{path}:{line_number}: missing audio or text")
            if check_audio and not os.path.isfile(audio):
                raise FileNotFoundError(f"{path}:{line_number}: missing audio: {audio}")
            normalized = dict(row)
            normalized["audio"] = audio
            normalized["text"] = f"language English<asr_text>{transcript}"
            if normalized.get("aug") is None:
                normalized["aug"] = 1
            rows.append(normalized)
    if not rows:
        raise ValueError(f"No valid rows in {path}")
    return rows


def text_chars(rows: list[dict[str, Any]]) -> int:
    return sum(len(transcript_from_text(row["text"])) for row in rows)


def sample_to_char_target(
    rows: list[dict[str, Any]],
    target_chars: int,
    rng: random.Random,
    allow_replacement: bool,
) -> tuple[list[dict[str, Any]], str]:
    """Sample rows to a character budget, with replacement only for real data."""
    if target_chars <= 0:
        raise ValueError("target_chars must be positive")
    source_chars = text_chars(rows)
    if allow_replacement and target_chars >= source_chars:
        sampled: list[dict[str, Any]] = []
        current_chars = 0
        while current_chars < target_chars:
            cycle = list(rows)
            rng.shuffle(cycle)
            for row in cycle:
                sampled.append(row)
                current_chars += len(transcript_from_text(row["text"]))
                if current_chars >= target_chars:
                    break
        return sampled, "upsample_with_replacement"

    if source_chars <= target_chars:
        return list(rows), "keep_all"

    sampled = []
    current_chars = 0
    shuffled = list(rows)
    rng.shuffle(shuffled)
    for row in shuffled:
        sampled.append(row)
        current_chars += len(transcript_from_text(row["text"]))
        if current_chars >= target_chars:
            break
    return sampled, "downsample_without_replacement"


def source_report(
    rows: list[dict[str, Any]],
    source_chars: int,
    input_rows: int,
    mode: str,
) -> dict[str, Any]:
    repetitions = Counter(str(row["audio"]) for row in rows)
    return {
        "rows": len(rows),
        "input_rows": input_rows,
        "sampling_mode": mode,
        "row_multiplier": len(rows) / input_rows,
        "text_chars": text_chars(rows),
        "unique_audio": len(repetitions),
        "min_repetitions": min(repetitions.values()),
        "max_repetitions": max(repetitions.values()),
        "source_text_chars_before_sampling": source_chars,
    }


def main() -> None:
    args = parse_args()
    if args.real_max_repeats < 1:
        raise ValueError("real_max_repeats must be positive")
    rng = random.Random(args.seed)
    sources = {
        "real": load_rows(Path(args.real_jsonl), bool(args.check_audio)),
        "synthetic": load_rows(Path(args.synthetic_jsonl), bool(args.check_audio)),
        "medical_word_frequency": load_rows(Path(args.medical_jsonl), bool(args.check_audio)),
    }
    source_sizes = {name: text_chars(rows) for name, rows in sources.items()}
    target_chars = min(
        max(source_sizes.values()),
        source_sizes["real"] * args.real_max_repeats,
    )

    sampled: dict[str, list[dict[str, Any]]] = {}
    sampling_modes: dict[str, str] = {}
    for name, rows in sources.items():
        sampled_rows, mode = sample_to_char_target(
            rows,
            target_chars,
            rng,
            allow_replacement=(name == "real"),
        )
        sampling_modes[name] = mode
        sampled[name] = [dict(row, sampling_source=name) for row in sampled_rows]

    combined = [row for rows in sampled.values() for row in rows]
    rng.shuffle(combined)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in combined:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "seed": args.seed,
        "balance_metric": "transcript_unicode_characters",
        "target_text_chars": target_chars,
        "real_max_repeats": args.real_max_repeats,
        "target_rule": "min(largest_source_chars, real_source_chars * real_max_repeats)",
        "input": {
            "real_jsonl": str(Path(args.real_jsonl).resolve()),
            "synthetic_jsonl": str(Path(args.synthetic_jsonl).resolve()),
            "medical_jsonl": str(Path(args.medical_jsonl).resolve()),
        },
        "sources": {
            name: source_report(
                sampled[name],
                source_sizes[name],
                len(sources[name]),
                sampling_modes[name],
            )
            for name in sources
        },
        "output": {
            "train_jsonl": str(output_path.resolve()),
            "total_rows": len(combined),
            "total_text_chars": text_chars(combined),
        },
    }
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
