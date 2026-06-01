#!/usr/bin/env python3
# coding=utf-8
"""
Build a balanced Qwen3-ASR training JSONL from raw/ sources.

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
from typing import Any, Dict, List, Optional

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
    p = argparse.ArgumentParser(description="Prepare balanced multilingual train/dev JSONL.")
    p.add_argument("--raw_dir", type=str, default="raw", help="Default directory for relative input paths.")
    p.add_argument(
        "--l2arctic_csv",
        type=str,
        default="raw/l2arctic_english_chinese.csv",
        help="L2-ARCTIC CSV (path, transcript).",
    )
    p.add_argument(
        "--switchlingua_jsonl",
        type=str,
        default="raw/switchlingua_cantonese_english.jsonl",
    )
    p.add_argument(
        "--youtube_mix_jsonl",
        type=str,
        default="raw/youtube_english_cantonese.jsonl",
    )
    p.add_argument(
        "--youtube_en_jsonl",
        type=str,
        default="raw/youtube_english.jsonl",
    )
    p.add_argument(
        "--medical_tts_csv",
        type=str,
        default="",
        help="Optional medical TTS metadata CSV (audio_path,text columns).",
    )
    p.add_argument(
        "--medical_tts_jsonl",
        type=str,
        default="",
        help="Optional pre-converted medical TTS jsonl (audio,text fields).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="data/balanced_multilingual",
        help="Writes train.jsonl, dev.jsonl, prepare_report.json under this directory.",
    )
    p.add_argument("--train_jsonl", type=str, default="", help="Override train output path.")
    p.add_argument("--dev_jsonl", type=str, default="", help="Override dev output path.")
    p.add_argument(
        "--report_json",
        type=str,
        default="",
        help="Override prepare report path.",
    )
    p.add_argument("--train_ratio", type=float, default=0.95)
    p.add_argument("--dev_ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42, help="Fixed seed for balancing and split.")
    return p.parse_args()


def _resolve_path(path: str, raw_dir: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return (raw_dir / p).resolve() if (raw_dir / p).exists() else (Path(_REPO_ROOT) / p).resolve()


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


def balance_groups(
    pools: Dict[str, List[Sample]], rng: random.Random
) -> tuple[List[Sample], Dict[str, Any]]:
    counts = {name: len(rows) for name, rows in pools.items()}
    target = max(counts.values()) if counts else 0
    out: List[Sample] = []
    per_group_out: Dict[str, int] = {}
    for name, rows in sorted(pools.items()):
        picked = _balance_list(rows, target, rng)
        per_group_out[name] = len(picked)
        out.extend(picked)
    return out, {"target_per_group": target, "input_counts": counts, "output_counts": per_group_out}


def _load_jsonl_plain(
    path: Path,
    stats: SourceStats,
    locale: str,
    source: str,
) -> List[Sample]:
    rows: List[Sample] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
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
            norm = normalize_transcript(text, locale)
            if not norm:
                stats.skipped_empty_after_normalize += 1
                continue
            stats.valid += 1
            rows.append(
                Sample(
                    audio=os.path.abspath(audio),
                    transcript=norm,
                    source=source,
                )
            )
    return rows


def load_l2arctic(path: Path, stats: SourceStats) -> List[Sample]:
    rows: List[Sample] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "path" not in (reader.fieldnames or []) or "transcript" not in (reader.fieldnames or []):
            raise ValueError(f"CSV must have path,transcript columns: {path}")
        for row in reader:
            stats.total_lines += 1
            audio = str(row.get("path") or "").strip()
            text = str(row.get("transcript") or "").strip()
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
                    source="l2arctic",
                )
            )
    return rows


def load_medical_tts_csv(path: Path, stats: SourceStats) -> List[Sample]:
    rows: List[Sample] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "audio_path" not in (reader.fieldnames or []) or "text" not in (reader.fieldnames or []):
            raise ValueError(f"CSV must have audio_path,text columns: {path}")
        for row in reader:
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
                    source="medical_tts",
                )
            )
    return rows


def balance_groups_capped(
    pools: Dict[str, List[Sample]], cap: int, rng: random.Random
) -> tuple[List[Sample], Dict[str, Any]]:
    """Balance groups to a specific target size rather than the largest group."""
    out: List[Sample] = []
    per_group_out: Dict[str, int] = {}
    for name, rows in sorted(pools.items()):
        picked = _balance_list(rows, cap, rng)
        per_group_out[name] = len(picked)
        out.extend(picked)
    return out, {"target": cap, "input_counts": {n: len(r) for n, r in pools.items()}, "output_counts": per_group_out}


def write_jsonl(path: Path, samples: List[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for s in samples:
            record = {
                "audio": s.audio,
                "text": f"{_TEXT_PREFIX}{s.transcript}",
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_train_dev(
    rows: List[Sample], train_ratio: float, dev_ratio: float, seed: int
) -> tuple[List[Sample], List[Sample]]:
    if train_ratio + dev_ratio <= 0:
        raise ValueError("train_ratio + dev_ratio must be positive")
    pool = list(rows)
    random.Random(seed).shuffle(pool)
    n = len(pool)
    train_n = int(n * train_ratio)
    dev_n = n - train_n
    return pool[:train_n], pool[train_n : train_n + dev_n]


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    rng = random.Random(args.seed)

    paths = {
        "l2arctic": _resolve_path(args.l2arctic_csv, raw_dir),
        "switchlingua": _resolve_path(args.switchlingua_jsonl, raw_dir),
        "youtube_mix": _resolve_path(args.youtube_mix_jsonl, raw_dir),
        "youtube_en": _resolve_path(args.youtube_en_jsonl, raw_dir),
    }

    source_stats: Dict[str, Any] = {}
    other_by_source: Dict[str, List[Sample]] = {}

    st_l2 = SourceStats()
    other_by_source["l2arctic"] = load_l2arctic(paths["l2arctic"], st_l2)
    source_stats["l2arctic"] = st_l2.__dict__

    st_mix = SourceStats()
    other_by_source["youtube_mix"] = _load_jsonl_plain(
        paths["youtube_mix"], st_mix, "cantonese", "youtube_mix"
    )
    source_stats["youtube_mix"] = st_mix.__dict__

    st_en = SourceStats()
    other_by_source["youtube_en"] = _load_jsonl_plain(
        paths["youtube_en"], st_en, "english", "youtube_en"
    )
    source_stats["youtube_en"] = st_en.__dict__

    st_sl = SourceStats()
    switchlingua_rows = _load_jsonl_plain(
        paths["switchlingua"], st_sl, "cantonese", "switchlingua"
    )
    source_stats["switchlingua"] = st_sl.__dict__

    # Optional medical TTS source
    medical_tts_rows: List[Sample] = []
    st_tts = SourceStats()
    tts_csv = args.medical_tts_csv.strip()
    tts_jsonl = args.medical_tts_jsonl.strip()
    if tts_csv:
        if tts_jsonl:
            print(
                "[prepare] Both --medical_tts_csv and --medical_tts_jsonl provided; "
                "using --medical_tts_csv and ignoring --medical_tts_jsonl."
            )
        tts_path = _resolve_path(tts_csv, raw_dir)
        medical_tts_rows = load_medical_tts_csv(tts_path, st_tts)
        source_stats["medical_tts"] = st_tts.__dict__
    elif tts_jsonl:
        tts_path = _resolve_path(tts_jsonl, raw_dir)
        medical_tts_rows = _load_jsonl_plain(tts_path, st_tts, "english", "medical_tts")
        source_stats["medical_tts"] = st_tts.__dict__

    other_balanced, intra_report = balance_sources(other_by_source, rng)

    if medical_tts_rows:
        # Cap TTS to the max of non-TTS groups so it doesn't force massive upsampling
        non_tts_cap = max(len(other_balanced), len(switchlingua_rows))
        language_pools = {
            "other": other_balanced,
            "switchlingua": switchlingua_rows,
            "medical_tts": medical_tts_rows,
        }
        final_rows, inter_report = balance_groups_capped(language_pools, non_tts_cap, rng)
    else:
        language_pools = {
            "other": other_balanced,
            "switchlingua": switchlingua_rows,
        }
        final_rows, inter_report = balance_groups(language_pools, rng)
    rng.shuffle(final_rows)

    train_rows, dev_rows = split_train_dev(
        final_rows, args.train_ratio, args.dev_ratio, args.seed
    )

    out_dir = Path(args.output_dir)
    train_path = Path(args.train_jsonl) if args.train_jsonl else out_dir / "train.jsonl"
    dev_path = Path(args.dev_jsonl) if args.dev_jsonl else out_dir / "dev.jsonl"
    report_path = Path(args.report_json) if args.report_json else out_dir / "prepare_report.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(dev_path, dev_rows)

    report = {
        "seed": args.seed,
        "text_prefix": _TEXT_PREFIX,
        "input_paths": {k: str(v) for k, v in paths.items()},
        "source_stats": source_stats,
        "intra_group_other_sources": intra_report,
        "inter_group": inter_report,
        "final_all_samples": len(final_rows),
        "train_samples": len(train_rows),
        "dev_samples": len(dev_rows),
        "train_ratio": args.train_ratio,
        "dev_ratio": args.dev_ratio,
        "train_jsonl": str(train_path.resolve()),
        "dev_jsonl": str(dev_path.resolve()),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
