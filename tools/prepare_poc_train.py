#!/usr/bin/env python3
"""Prepare the five-source POC training mix and a held-out POC test manifest."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import soundfile as sf


ASR_PREFIX = "language English<asr_text>"
ASR_NONE_PREFIX = "language None<asr_text>"
GROUP_RE = re.compile(r"_seg_", re.IGNORECASE)
CJK_IDEOGRAPH_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002ebef]")

def contains_cjk_ideograph(text: str) -> bool:
    return bool(CJK_IDEOGRAPH_RE.search(text))


def is_from_denoised(audio_path: str) -> bool:
    return "from_denoised" in Path(audio_path).parts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real_jsonl", default="raw/POC_train/real_target_domain/metadata.jsonl")
    parser.add_argument("--tcga_csv", default="raw/POC_train/tcga_reports_en/metadata.csv")
    parser.add_argument(
        "--tcga_audio_root",
        default="/root/autodl-tmp/workspace/project/Medical-Speech-Synthesis",
    )
    parser.add_argument("--youtube_jsonl", default="raw/POC_train/youtube_med_en/manifest.jsonl")
    parser.add_argument("--multimed_csv", default="raw/POC_train/multimed_en/english_manifest_medical_keyword.csv")
    parser.add_argument("--silence_csv", default="raw/POC_train/silence/metadata.csv")
    parser.add_argument("--real_no_cjk_only", type=int, default=0, choices=(0, 1))
    parser.add_argument("--real_no_upsample", type=int, default=0, choices=(0, 1))
    parser.add_argument("--real_denoised_only", type=int, default=0, choices=(0, 1))
    parser.add_argument("--real_label_none", type=int, default=0, choices=(0, 1))
    parser.add_argument("--test_jsonl", default="raw/POC_test/metadata.jsonl")
    parser.add_argument("--output_dir", default="data/poc_train_mix_sft")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real_hours", type=float, default=27.891)
    parser.add_argument("--tcga_hours", type=float, default=31.319)
    parser.add_argument("--youtube_hours", type=float, default=24.62)
    parser.add_argument("--multimed_hours", type=float, default=4.492)
    parser.add_argument("--silence_hours", type=float, default=4.649)
    parser.add_argument("--dev_fraction", type=float, default=0.2)
    parser.add_argument("--check_audio", type=int, default=1, choices=(0, 1))
    return parser.parse_args()


def absolute_audio(raw: Any, source: str, repo_root: Path, tcga_root: Path) -> Path:
    value = str(raw or "").strip()
    if not value:
        return Path()
    path = Path(value)
    if path.is_absolute():
        return path
    if source == "tcga_reports_en":
        return tcga_root / path
    return repo_root / path


def read_duration(path: Path) -> float:
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def transcript_body(value: Any) -> str:
    text = str(value or "").strip()
    if "<asr_text>" in text:
        text = text.split("<asr_text>", 1)[1].strip()
    return text


def qwen_text(value: Any) -> str:
    return f"{ASR_PREFIX}{transcript_body(value)}"


def source_group(audio: Path) -> str:
    return GROUP_RE.split(audio.stem, maxsplit=1)[0]


def iter_rows(path: Path, kind: str) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        if kind == "jsonl":
            for line_no, raw in enumerate(handle, 1):
                if raw.strip():
                    yield line_no, json.loads(raw)
        else:
            for line_no, row in enumerate(csv.DictReader(handle), 2):
                yield line_no, row


def load_source(
    name: str,
    path: Path,
    kind: str,
    audio_key: str,
    repo_root: Path,
    tcga_root: Path,
    check_audio: bool,
    allow_empty_text: bool = False,
    aug: int = 1,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw in iter_rows(path, kind):
        audio = absolute_audio(raw.get(audio_key), name, repo_root, tcga_root)
        text = transcript_body(raw.get("text"))
        if not audio or (check_audio and not audio.is_file()):
            raise FileNotFoundError(f"{path}:{line_no}: missing audio: {audio}")
        if not text and not allow_empty_text:
            raise ValueError(f"{path}:{line_no}: empty transcript")
        duration = read_duration(audio)
        rows.append(
            {
                "audio": str(audio.resolve()),
                "text": "language None<asr_text>" if name == "silence" else qwen_text(text),
                "aug": aug,
                "duration_sec": duration,
                "source": name,
                "source_group": source_group(audio),
            }
        )
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return rows



def rows_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "rows": 0,
            "unique_audio": 0,
            "duration_sec": 0.0,
            "duration_hours": 0.0,
            "mean_sec": 0.0,
        }

    durations = [float(row["duration_sec"]) for row in rows]
    unique = len({row["audio"] for row in rows})
    return {
        "rows": len(rows),
        "unique_audio": unique,
        "duration_sec": round(sum(durations), 3),
        "duration_hours": round(sum(durations) / 3600.0, 3),
        "mean_sec": round(sum(durations) / len(durations), 3),
    }


def choose_dev_groups(rows: list[dict[str, Any]], fraction: float) -> tuple[list[str], float]:
    by_group: dict[str, float] = defaultdict(float)
    for row in rows:
        by_group[str(row["source_group"])] += float(row["duration_sec"])
    keys = sorted(by_group)
    if len(keys) <= 1 or fraction <= 0:
        return [], 0.0
    target = sum(by_group.values()) * fraction
    min_groups = min(3, len(keys) - 1)
    max_groups = min(6, len(keys) - 1)
    best: tuple[float, tuple[str, ...], float] | None = None
    for count in range(min_groups, max_groups + 1):
        for combo in itertools.combinations(keys, count):
            duration = sum(by_group[key] for key in combo)
            candidate = (abs(duration - target), combo, duration)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
    if best is None:
        return [], 0.0
    return list(best[1]), best[2]


def sample_by_duration(
    rows: list[dict[str, Any]], hours: float, rng: random.Random, repeat: bool = False
) -> list[dict[str, Any]]:
    target = max(0.0, hours * 3600.0)
    if target <= 0:
        return []
    selected: list[dict[str, Any]] = []
    duration = 0.0
    cycle = 0
    while duration < target:
        shuffled = [dict(row, repeat_index=cycle) for row in rows]
        rng.shuffle(shuffled)
        for row in shuffled:
            selected.append(row)
            duration += float(row["duration_sec"])
            if duration >= target:
                break
        if duration < target and not repeat:
            raise ValueError(f"Requested {hours} hours from a source with only {duration / 3600.0:.3f} hours")
        cycle += 1
    return selected


def public_row(row: dict[str, Any], sampling_source: str, aug: int | None = None) -> dict[str, Any]:
    out = {
        "audio": row["audio"],
        "text": row["text"],
        "aug": int(row["aug"] if aug is None else aug),
        "sampling_source": sampling_source,
        "source_group": row["source_group"],
        "duration_sec": round(float(row["duration_sec"]), 6),
    }
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if not 0 <= args.dev_fraction < 1:
        raise ValueError("--dev_fraction must be in [0, 1)")
    for option, value in vars(args).items():
        if option.endswith("_hours") and value < 0:
            raise ValueError(f"--{option} must be non-negative")

    repo_root = Path(__file__).resolve().parents[1]
    tcga_root = Path(args.tcga_audio_root).resolve()
    check_audio = bool(args.check_audio)
    sources = {
        "real_target_domain": load_source(
            "real_target_domain", Path(args.real_jsonl), "jsonl", "audio_path",
            repo_root, tcga_root, check_audio, aug=1,
        ),
        "tcga_reports_en": load_source(
            "tcga_reports_en", Path(args.tcga_csv), "csv", "audio_path",
            repo_root, tcga_root, check_audio, aug=1,
        ),
        "youtube_med_en": load_source(
            "youtube_med_en", Path(args.youtube_jsonl), "jsonl", "audio",
            repo_root, tcga_root, check_audio, aug=1,
        ),
        "multimed_en": load_source(
            "multimed_en", Path(args.multimed_csv), "csv", "audio_path",
            repo_root, tcga_root, check_audio, aug=1,
        ),
        "silence": load_source(
            "silence", Path(args.silence_csv), "csv", "audio_path",
            repo_root, tcga_root, check_audio, allow_empty_text=True, aug=1,
        ),
    }

    if args.real_no_cjk_only:
        sources["real_target_domain"] = [
            row
            for row in sources["real_target_domain"]
            if not contains_cjk_ideograph(transcript_body(row["text"]))
        ]
        if not sources["real_target_domain"]:
            raise ValueError("No real_target_domain rows remain after CJK filtering")

    if args.real_denoised_only:
        sources["real_target_domain"] = [
            row for row in sources["real_target_domain"] if is_from_denoised(row["audio"])
        ]
        if not sources["real_target_domain"]:
            raise ValueError("No real_target_domain rows remain after from_denoised filtering")

    if args.real_label_none:
        for row in sources["real_target_domain"]:
            row["text"] = f"{ASR_NONE_PREFIX}{transcript_body(row['text'])}"

    rng = random.Random(args.seed)
    target_dev_groups, target_dev_duration = choose_dev_groups(
        sources["real_target_domain"], args.dev_fraction
    )
    target_dev_set = set(target_dev_groups)
    target_train_rows = [
        row for row in sources["real_target_domain"] if row["source_group"] not in target_dev_set
    ]
    target_dev_rows = [
        row for row in sources["real_target_domain"] if row["source_group"] in target_dev_set
    ]

    sampled = {
        "real_target_domain": (
            list(target_train_rows)
            if args.real_no_upsample
            else sample_by_duration(target_train_rows, args.real_hours, rng, repeat=True)
        ),
        "tcga_reports_en": sample_by_duration(sources["tcga_reports_en"], args.tcga_hours, rng),
        "youtube_med_en": sample_by_duration(sources["youtube_med_en"], args.youtube_hours, rng),
        "multimed_en": sample_by_duration(sources["multimed_en"], args.multimed_hours, rng),
        "silence": sample_by_duration(sources["silence"], args.silence_hours, rng, repeat=True),
    }

    train_rows: list[dict[str, Any]] = []
    for name, rows in sampled.items():
        train_rows.extend(public_row(row, name) for row in rows)
    rng.shuffle(train_rows)

    dev_rows = [public_row(row, "real_target_domain_dev", aug=0) for row in target_dev_rows]
    test_source = load_source(
        "POC_test", Path(args.test_jsonl), "jsonl", "audio_path",
        repo_root, tcga_root, check_audio, aug=0,
    )
    test_rows = [public_row(row, "POC_test", aug=0) for row in test_source]

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "dev.jsonl", dev_rows)
    write_jsonl(output_dir / "test.jsonl", test_rows)

    inventory = {name: rows_stats(rows) for name, rows in sources.items()}
    sampled_report = {name: rows_stats(rows) for name, rows in sampled.items()}
    report = {
        "seed": args.seed,
        "policy": {
            "target_hours": {
                "real_target_domain": args.real_hours,
                "tcga_reports_en": args.tcga_hours,
                "youtube_med_en": args.youtube_hours,
                "multimed_en": args.multimed_hours,
                "silence": args.silence_hours,
            },
            "dev_fraction": args.dev_fraction,
            "real_no_cjk_only": bool(args.real_no_cjk_only),
            "real_no_upsample": bool(args.real_no_upsample),
            "real_denoised_only": bool(args.real_denoised_only),
            "real_label_none": bool(args.real_label_none),
            "synthetic_aug": 1,
            "silence_aug": 1,
        },
        "inventory": inventory,
        "target_domain_split": {
            "dev_groups": target_dev_groups,
            "dev": rows_stats(target_dev_rows),
            "train_before_repeat": rows_stats(target_train_rows),
            "dev_duration_fraction": round(target_dev_duration / sum(row["duration_sec"] for row in sources["real_target_domain"]), 4),
        },
        "sampled_train_sources": sampled_report,
        "output": {
            "train": rows_stats(train_rows),
            "dev": rows_stats(dev_rows),
            "test": rows_stats(test_rows),
            "train_jsonl": str((output_dir / "train.jsonl").resolve()),
            "dev_jsonl": str((output_dir / "dev.jsonl").resolve()),
            "test_jsonl": str((output_dir / "test.jsonl").resolve()),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "prepare_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
