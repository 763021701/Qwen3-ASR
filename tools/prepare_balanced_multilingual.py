#!/usr/bin/env python3
# coding=utf-8
"""
Build a balanced multilingual Qwen3-ASR training JSONL from raw/ sources.

Balancing: docs/balanced_dataset.md
Label normalize: docs/normalize_label.md (conservative)
Final text: language None<asr_text>{normalized_body}
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.normalize_transcript import normalize_transcript

_TEXT_PREFIX = "language None<asr_text>"


@dataclass
class Sample:
    audio: str
    transcript: str
    language: str
    source: str


@dataclass
class SourceStats:
    total_lines: int = 0
    valid: int = 0
    skipped_invalid_json: int = 0
    skipped_missing_audio: int = 0
    skipped_empty_text: int = 0
    skipped_empty_after_normalize: int = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare balanced multilingual train.jsonl.")
    p.add_argument(
        "--raw_dir",
        type=str,
        default="raw",
        help="Directory containing youtube_english.jsonl, english_multimed.csv, cantonese_english.jsonl",
    )
    p.add_argument(
        "--output_jsonl",
        type=str,
        default="data/balanced_multilingual/train.jsonl",
    )
    p.add_argument(
        "--report_json",
        type=str,
        default="data/balanced_multilingual/prepare_report.json",
    )
    p.add_argument("--seed", type=int, default=42, help="Fixed seed for all random ops.")
    return p.parse_args()


def _balance_list(items: List[Sample], target: int, rng: random.Random) -> List[Sample]:
    if not items or target <= 0:
        return []
    pool = list(items)
    while len(pool) < target:
        pool.extend(items)
    rng.shuffle(pool)
    return pool[:target]


def balance_sources(
    by_source: Dict[str, List[Sample]], rng: random.Random
) -> tuple[List[Sample], Dict[str, Any]]:
    counts = {name: len(rows) for name, rows in by_source.items()}
    target = max(counts.values()) if counts else 0
    out: List[Sample] = []
    per_source_out: Dict[str, int] = {}
    for name, rows in sorted(by_source.items()):
        picked = _balance_list(rows, target, rng)
        per_source_out[name] = len(picked)
        out.extend(picked)
    return out, {"target_per_source": target, "input_counts": counts, "output_counts": per_source_out}


def balance_language_pools(
    pools: Dict[str, List[Sample]], rng: random.Random
) -> tuple[List[Sample], Dict[str, Any]]:
    counts = {lang: len(rows) for lang, rows in pools.items()}
    target = max(counts.values()) if counts else 0
    out: List[Sample] = []
    per_lang_out: Dict[str, int] = {}
    for lang, rows in sorted(pools.items()):
        picked = _balance_list(rows, target, rng)
        per_lang_out[lang] = len(picked)
        out.extend(picked)
    return out, {"target_per_language": target, "input_counts": counts, "output_counts": per_lang_out}


def load_youtube_english(path: Path, stats: SourceStats) -> List[Sample]:
    rows: List[Sample] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            stats.total_lines += 1
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                stats.skipped_invalid_json += 1
                continue
            audio = str(item.get("audio") or "").strip()
            text = str(item.get("text") or "").strip()
            if not text:
                stats.skipped_empty_text += 1
                continue
            if not audio or not os.path.isfile(audio):
                stats.skipped_missing_audio += 1
                continue
            norm = normalize_transcript(text, "english")
            if not norm:
                stats.skipped_empty_after_normalize += 1
                continue
            stats.valid += 1
            rows.append(
                Sample(
                    audio=os.path.abspath(audio),
                    transcript=norm,
                    language="English",
                    source="youtube_english",
                )
            )
    return rows


def load_english_multimed(path: Path, stats: SourceStats) -> List[Sample]:
    rows: List[Sample] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "audio_path" not in (reader.fieldnames or []) or "text" not in (reader.fieldnames or []):
            raise ValueError(f"CSV must have audio_path,text columns: {path}")
        for line_no, row in enumerate(reader, start=2):
            stats.total_lines += 1
            audio = str(row.get("audio_path") or "").strip()
            text = str(row.get("text") or "").strip()
            if not text:
                stats.skipped_empty_text += 1
                continue
            if not audio or not os.path.isfile(audio):
                stats.skipped_missing_audio += 1
                continue
            norm = normalize_transcript(text, "english")
            if not norm:
                stats.skipped_empty_after_normalize += 1
                continue
            stats.valid += 1
            rows.append(
                Sample(
                    audio=os.path.abspath(audio),
                    transcript=norm,
                    language="English",
                    source="english_multimed",
                )
            )
    return rows


def load_cantonese_english(path: Path, stats: SourceStats) -> List[Sample]:
    rows: List[Sample] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            stats.total_lines += 1
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                stats.skipped_invalid_json += 1
                continue
            audio = str(item.get("audio") or "").strip()
            text = str(item.get("text") or "").strip()
            if not text:
                stats.skipped_empty_text += 1
                continue
            if not audio or not os.path.isfile(audio):
                stats.skipped_missing_audio += 1
                continue
            norm = normalize_transcript(text, "cantonese")
            if not norm:
                stats.skipped_empty_after_normalize += 1
                continue
            stats.valid += 1
            rows.append(
                Sample(
                    audio=os.path.abspath(audio),
                    transcript=norm,
                    language="English,Cantonese",
                    source="cantonese_english",
                )
            )
    return rows


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    rng = random.Random(args.seed)

    source_stats: Dict[str, Any] = {}
    english_by_source: Dict[str, List[Sample]] = {}
    codeswitch_rows: List[Sample] = []

    yt_path = raw_dir / "youtube_english.jsonl"
    em_path = raw_dir / "english_multimed.csv"
    ce_path = raw_dir / "cantonese_english.jsonl"

    st_yt = SourceStats()
    english_by_source["youtube_english"] = load_youtube_english(yt_path, st_yt)
    source_stats["youtube_english"] = st_yt.__dict__

    st_em = SourceStats()
    english_by_source["english_multimed"] = load_english_multimed(em_path, st_em)
    source_stats["english_multimed"] = st_em.__dict__

    st_ce = SourceStats()
    codeswitch_rows = load_cantonese_english(ce_path, st_ce)
    source_stats["cantonese_english"] = st_ce.__dict__

    english_balanced, intra_report = balance_sources(english_by_source, rng)
    language_pools = {
        "English": english_balanced,
        "English,Cantonese": codeswitch_rows,
    }
    final_rows, inter_report = balance_language_pools(language_pools, rng)
    rng.shuffle(final_rows)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for s in final_rows:
            record = {
                "audio": s.audio,
                "text": f"{_TEXT_PREFIX}{s.transcript}",
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = {
        "seed": args.seed,
        "text_prefix": _TEXT_PREFIX,
        "source_stats": source_stats,
        "intra_language_english": intra_report,
        "inter_language": inter_report,
        "final_train_samples": len(final_rows),
        "output_jsonl": str(output_path.resolve()),
    }
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
