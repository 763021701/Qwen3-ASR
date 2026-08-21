#!/usr/bin/env python3
"""Prepare a real-domain Qwen3-ASR manifest from raw and denoised audio."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import soundfile as sf

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.english_medical.text_normalization import normalize_english

_ADDITIONAL_CSV_COLUMNS = ("source", "index", "audio_path", "text", "original", "duration")


ASR_PREFIX = "language None<asr_text>"
SYNTHETIC_ASR_PREFIX = "language English<asr_text>"
SEGMENT_RE = re.compile(r"_seg_", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw_jsonl",
        default="raw/POC_train/real_target_domain/metadata_raw.jsonl",
    )
    parser.add_argument(
        "--denoised_jsonl",
        default="raw/POC_train/real_target_domain/metadata_denoised.jsonl",
    )
    parser.add_argument("--test_source", default="raw/POC_test/metadata.jsonl")
    parser.add_argument("--output_dir", default="data/poc_train_real_raw_denoised")
    parser.add_argument("--train_jsonl", default="")
    parser.add_argument("--dev_jsonl", default="")
    parser.add_argument("--test_jsonl", default="")
    parser.add_argument("--dev_fraction", type=float, default=0.2)
    parser.add_argument("--use_test_as_dev", type=int, default=0, choices=(0, 1))
    parser.add_argument("--synthetic_jsonl", default="")
    parser.add_argument("--synthetic_metadata_csv", default="")
    parser.add_argument("--synthetic_audio_root", default="")
    parser.add_argument("--synthetic_exclude_priorities", default="test_only")
    parser.add_argument(
        "--additional_synthetic_csv",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Extra TTS CSV (columns: source,index,audio_path,text,original,duration). "
        "Repeatable. Rows are voice-downsampled to at most "
        "--additional_synthetic_max_per_text distinct sources per normalized text.",
    )
    parser.add_argument("--additional_synthetic_max_per_text", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check_audio", type=int, default=1, choices=(0, 1))
    return parser.parse_args()


def transcript_body(value: Any) -> str:
    text = str(value or "").strip()
    if "<asr_text>" in text:
        text = text.split("<asr_text>", 1)[1].strip()
    return text


def resolve_audio(value: Any, manifest_path: Path) -> Path:
    path = Path(str(value or "").strip())
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def source_group(audio: Path) -> str:
    return SEGMENT_RE.split(audio.stem, maxsplit=1)[0]


def segment_id(audio: Path) -> str:
    source_dir = re.sub(r"_(?:raw|denoised)$", "", audio.parent.name, flags=re.IGNORECASE)
    return f"{source_dir}/{audio.stem}"


def duration_seconds(row: dict[str, Any], audio: Path) -> float:
    value = row.get("duration")
    if value is not None and str(value).strip():
        return float(value)
    info = sf.info(str(audio))
    return float(info.frames) / float(info.samplerate)


def load_rows(
    path: Path,
    *,
    sampling_source: str,
    noise_aug: int,
    check_audio: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_segments: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            audio_value = str(row.get("audio_path") or row.get("audio") or "").strip()
            if not audio_value:
                raise ValueError(f"{path}:{line_no}: missing audio path")
            audio = resolve_audio(audio_value, path)
            text = transcript_body(row.get("text"))
            if check_audio and not audio.is_file():
                raise FileNotFoundError(f"{path}:{line_no}: missing audio: {audio}")
            if not text:
                raise ValueError(f"{path}:{line_no}: empty transcript")

            segment_key = segment_id(audio)
            if segment_key in seen_segments:
                raise ValueError(f"{path}:{line_no}: duplicate segment id: {segment_key}")
            seen_segments.add(segment_key)
            rows.append(
                {
                    "audio": str(audio),
                    "text": f"{ASR_PREFIX}{text}",
                    "aug": 1,
                    "noise_aug": int(noise_aug),
                    "sampling_source": sampling_source,
                    "source_group": source_group(audio),
                    "segment_id": segment_key,
                    "duration_sec": round(duration_seconds(row, audio), 6),
                }
            )
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return rows



def load_synthetic_rows(
    text_path: Path,
    metadata_path: Path,
    audio_root: Path,
    *,
    excluded_priorities: set[str],
    heldout_groups: set[str],
    heldout_texts: set[str],
    check_audio: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    text_items: list[dict[str, Any]] = []
    with text_path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            item = json.loads(raw)
            if not transcript_body(item.get("text")):
                raise ValueError(f"{text_path}:{line_no}: empty synthetic transcript")
            text_items.append(item)

    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        metadata_rows = list(csv.DictReader(handle))
    if len(text_items) != len(metadata_rows):
        raise ValueError(
            "Synthetic text/metadata row count mismatch: "
            f"{len(text_items)} != {len(metadata_rows)}"
        )

    metadata_by_text: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for metadata in metadata_rows:
        metadata_by_text[transcript_body(metadata.get("text"))].append(metadata)

    rows: list[dict[str, Any]] = []
    excluded_counts: dict[str, int] = defaultdict(int)
    seen_audio: set[Path] = set()
    for index, item in enumerate(text_items):
        text = transcript_body(item.get("text"))
        matches = metadata_by_text.get(text)
        if not matches:
            raise ValueError(f"Synthetic text mismatch at row {index}")
        metadata = matches.pop(0)

        priority = str(item.get("priority") or "").strip()
        ref_audio_value = str(metadata.get("ref_audio") or "").strip()
        if not ref_audio_value:
            raise ValueError(f"{metadata_path}:{index + 2}: missing ref_audio")
        ref_group = source_group(Path(ref_audio_value))

        reasons: list[str] = []
        if priority in excluded_priorities:
            reasons.append(f"priority:{priority}")
        if ref_group in heldout_groups:
            reasons.append("heldout_ref_group")
        if text in heldout_texts:
            reasons.append("exact_heldout_text")
        for reason in reasons:
            excluded_counts[reason] += 1
        if reasons:
            continue

        audio_value = str(metadata.get("audio_path") or "").strip()
        if not audio_value:
            raise ValueError(f"{metadata_path}:{index + 2}: missing audio_path")
        audio = Path(audio_value)
        if not audio.is_absolute():
            audio = audio_root / audio
        audio = audio.resolve()
        if audio in seen_audio:
            raise ValueError(f"Duplicate synthetic audio: {audio}")
        seen_audio.add(audio)
        if check_audio and not audio.is_file():
            raise FileNotFoundError(f"Missing synthetic audio: {audio}")

        rows.append(
            {
                "audio": str(audio),
                "text": f"{SYNTHETIC_ASR_PREFIX}{text}",
                "aug": 1,
                "noise_aug": 1,
                "sampling_source": "llm_text_syn",
                "source_group": "llm_text_syn",
                "segment_id": f"llm_text_syn/{audio.stem}",
                "duration_sec": round(duration_seconds({}, audio), 6),
            }
        )

    if not rows:
        raise ValueError("Synthetic filtering removed every row")
    report = {
        "input_rows": len(text_items),
        "included": rows_stats(rows),
        "excluded_unique": len(text_items) - len(rows),
        "excluded_counts_nonexclusive": dict(sorted(excluded_counts.items())),
        "excluded_priorities": sorted(excluded_priorities),
        "heldout_groups": sorted(heldout_groups),
    }
    return rows, report


def parse_additional_synthetic_specs(values: list[str]) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []
    seen_names: set[str] = set()
    for value in values or []:
        name, sep, path = value.partition("=")
        if not sep or not name.strip() or not path.strip():
            raise ValueError(f"Invalid --additional_synthetic_csv {value!r}; expected NAME=PATH")
        name = name.strip()
        if name in seen_names:
            raise ValueError(f"Duplicate additional synthetic source name: {name}")
        seen_names.add(name)
        specs.append((name, Path(path.strip())))
    return specs


def load_additional_synthetic_rows(
    name: str,
    csv_path: Path,
    *,
    max_per_text: int,
    check_audio: bool,
    seen_audio: set[Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load a TTS CSV and keep at most `max_per_text` distinct voices per normalized text.

    Voice choice is greedy and deterministic: groups are processed in sorted
    normalized-text order; each pick takes the voice with the fewest samples kept
    so far, breaking ties by source name, then by index/audio_path within a voice.
    """
    if max_per_text < 1:
        raise ValueError("additional_synthetic_max_per_text must be >= 1")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [col for col in _ADDITIONAL_CSV_COLUMNS if col not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{csv_path}: missing required columns: {missing}")
        raw_rows = list(reader)

    parsed: list[dict[str, Any]] = []
    for line_no, row in enumerate(raw_rows, start=2):
        def field(col: str) -> str:
            return str(row.get(col) or "").strip()

        source, index = field("source"), field("index")
        audio_value, text = field("audio_path"), field("text")
        original, duration = field("original"), field("duration")
        if not source or not index or not audio_value or not text or not original:
            raise ValueError(
                f"{csv_path}:{line_no}: empty source/index/audio_path/text/original"
            )
        try:
            duration_sec = float(duration)
        except ValueError:
            raise ValueError(f"{csv_path}:{line_no}: invalid duration {duration!r}")
        if duration_sec < 0:
            raise ValueError(f"{csv_path}:{line_no}: negative duration {duration!r}")

        audio = Path(audio_value)
        if not audio.is_absolute():
            audio = csv_path.parent / audio
        audio = audio.resolve()
        if audio in seen_audio:
            raise ValueError(f"{csv_path}:{line_no}: duplicate audio: {audio}")
        if check_audio and not audio.is_file():
            raise FileNotFoundError(f"{csv_path}:{line_no}: missing audio: {audio}")
        seen_audio.add(audio)
        parsed.append(
            {
                "source": source,
                "index": index,
                "audio": audio,
                "text": text,
                "original": original,
                "duration_sec": duration_sec,
            }
        )
    if not parsed:
        raise ValueError(f"No rows loaded from {csv_path}")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in parsed:
        groups[normalize_english(item["text"])].append(item)

    voice_counts: dict[str, int] = defaultdict(int)
    kept: list[dict[str, Any]] = []
    for key in sorted(groups):
        by_voice: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in groups[key]:
            by_voice[item["source"]].append(item)
        for _ in range(min(max_per_text, len(by_voice))):
            voice = min(by_voice, key=lambda v: (voice_counts.get(v, 0), v))
            item = min(by_voice[voice], key=lambda r: (r["index"], str(r["audio"])))
            kept.append(item)
            voice_counts[voice] += 1
            del by_voice[voice]

    rows = [
        {
            "audio": str(item["audio"]),
            "text": f"{ASR_PREFIX}{item['original']}",
            "aug": 1,
            "noise_aug": 1,
            "sampling_source": name,
            "source_group": name,
            "segment_id": f"{name}/{item['audio'].parent.name}_{item['audio'].stem}",
            "duration_sec": round(item["duration_sec"], 6),
        }
        for item in kept
    ]
    report = {
        "input_rows": len(parsed),
        "normalized_text_groups": len(groups),
        "max_per_text": int(max_per_text),
        "kept": rows_stats(rows),
        "excluded_rows": len(parsed) - len(kept),
        "voice_counts": dict(sorted(voice_counts.items())),
    }
    return rows, report


def validate_pair_alignment(raw_rows: list[dict[str, Any]], denoised_rows: list[dict[str, Any]]) -> None:
    raw_by_id = {row["segment_id"]: row for row in raw_rows}
    den_by_id = {row["segment_id"]: row for row in denoised_rows}
    raw_ids = set(raw_by_id)
    den_ids = set(den_by_id)
    if raw_ids != den_ids:
        missing_raw = sorted(den_ids - raw_ids)
        missing_denoised = sorted(raw_ids - den_ids)
        raise ValueError(
            "Raw/denoised segment sets differ: "
            f"missing_raw={missing_raw[:5]}, missing_denoised={missing_denoised[:5]}"
        )

    mismatched = [
        segment_id
        for segment_id in sorted(raw_ids)
        if transcript_body(raw_by_id[segment_id]["text"])
        != transcript_body(den_by_id[segment_id]["text"])
    ]
    if mismatched:
        raise ValueError(f"Raw/denoised transcript mismatch for segments: {mismatched[:5]}")


def split_by_group(
    rows: list[dict[str, Any]], dev_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be between 0 and 1")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_group"])].append(row)
    if len(grouped) < 2:
        raise ValueError("At least two source groups are required for a train/dev split")

    groups = list(grouped.items())
    random.Random(seed).shuffle(groups)
    groups.sort(key=lambda item: sum(float(row["duration_sec"]) for row in item[1]), reverse=True)
    target = sum(float(row["duration_sec"]) for row in rows) * dev_fraction

    dev_groups: list[str] = []
    dev_duration = 0.0
    for group_id, group_rows in groups:
        group_duration = sum(float(row["duration_sec"]) for row in group_rows)
        remaining_groups = len(groups) - len(dev_groups)
        closer_with_group = abs(dev_duration + group_duration - target) <= abs(dev_duration - target)
        if remaining_groups > 1 and (not dev_groups or closer_with_group):
            dev_groups.append(group_id)
            dev_duration += group_duration

    if not dev_groups:
        dev_groups.append(groups[-1][0])
    if len(dev_groups) == len(groups):
        dev_groups.pop()

    dev_set = set(dev_groups)
    train = [row for row in rows if row["source_group"] not in dev_set]
    dev = [row for row in rows if row["source_group"] in dev_set]
    if not train or not dev:
        raise RuntimeError("Group split produced an empty train or dev split")
    return train, dev, sorted(dev_set)


def rows_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    duration = sum(float(row["duration_sec"]) for row in rows)
    return {
        "rows": len(rows),
        "unique_audio": len({row["audio"] for row in rows}),
        "groups": len({row["source_group"] for row in rows}),
        "duration_sec": round(duration, 3),
        "duration_hours": round(duration / 3600.0, 3),
        "noise_aug_rows": sum(int(row.get("noise_aug", 1)) for row in rows),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    raw_path = Path(args.raw_jsonl).resolve()
    denoised_path = Path(args.denoised_jsonl).resolve()
    test_path = Path(args.test_source).resolve()
    check_audio = bool(args.check_audio)

    raw_rows = load_rows(
        raw_path,
        sampling_source="real_raw",
        noise_aug=0,
        check_audio=check_audio,
    )
    denoised_rows = load_rows(
        denoised_path,
        sampling_source="real_denoised",
        noise_aug=1,
        check_audio=check_audio,
    )
    validate_pair_alignment(raw_rows, denoised_rows)

    test_rows = load_rows(
        test_path,
        sampling_source="poc_test",
        noise_aug=0,
        check_audio=check_audio,
    )
    test_rows = [dict(row, aug=0, noise_aug=0) for row in test_rows]
    use_test_as_dev = bool(getattr(args, "use_test_as_dev", 0))
    if use_test_as_dev:
        train_rows = list(raw_rows + denoised_rows)
        dev_rows = [dict(row) for row in test_rows]
        dev_groups = sorted({str(row["source_group"]) for row in dev_rows})
    else:
        train_rows, dev_rows, dev_groups = split_by_group(
            raw_rows + denoised_rows,
            dev_fraction=float(args.dev_fraction),
            seed=int(args.seed),
        )
        train_rows = list(train_rows)
        dev_rows = [dict(row, aug=0, noise_aug=0) for row in dev_rows]

    real_train_stats = rows_stats(train_rows)
    synthetic_report = None
    synthetic_values = [
        str(getattr(args, "synthetic_jsonl", "") or "").strip(),
        str(getattr(args, "synthetic_metadata_csv", "") or "").strip(),
        str(getattr(args, "synthetic_audio_root", "") or "").strip(),
    ]
    if any(synthetic_values):
        if not all(synthetic_values):
            raise ValueError(
                "Synthetic data requires --synthetic_jsonl, "
                "--synthetic_metadata_csv, and --synthetic_audio_root"
            )
        excluded_priorities = {
            value.strip()
            for value in str(
                getattr(args, "synthetic_exclude_priorities", "test_only") or ""
            ).split(",")
            if value.strip()
        }
        heldout_groups = {str(row["source_group"]) for row in dev_rows + test_rows}
        heldout_texts = {transcript_body(row["text"]) for row in dev_rows + test_rows}
        synthetic_rows, synthetic_report = load_synthetic_rows(
            Path(synthetic_values[0]).resolve(),
            Path(synthetic_values[1]).resolve(),
            Path(synthetic_values[2]).resolve(),
            excluded_priorities=excluded_priorities,
            heldout_groups=heldout_groups,
            heldout_texts=heldout_texts,
            check_audio=check_audio,
        )
        train_rows.extend(synthetic_rows)

    additional_specs = parse_additional_synthetic_specs(
        list(getattr(args, "additional_synthetic_csv", None) or [])
    )
    additional_report = None
    if additional_specs:
        max_per_text = int(getattr(args, "additional_synthetic_max_per_text", 2) or 2)
        seen_audio = {Path(row["audio"]) for row in train_rows + dev_rows + test_rows}
        additional_report = {}
        for source_name, csv_path in additional_specs:
            rows_, report_ = load_additional_synthetic_rows(
                source_name,
                csv_path,
                max_per_text=max_per_text,
                check_audio=check_audio,
                seen_audio=seen_audio,
            )
            train_rows.extend(rows_)
            additional_report[source_name] = report_

    random.Random(int(args.seed)).shuffle(train_rows)

    output_dir = Path(args.output_dir).resolve()
    train_out = Path(args.train_jsonl or output_dir / "train.jsonl")
    dev_out = Path(args.dev_jsonl or output_dir / "dev.jsonl")
    test_out = Path(args.test_jsonl or output_dir / "test.jsonl")
    write_jsonl(train_out, train_rows)
    write_jsonl(dev_out, dev_rows)
    write_jsonl(test_out, test_rows)

    train_groups = {
        row["source_group"]
        for row in train_rows
        if row["sampling_source"] in {"real_raw", "real_denoised"}
    }
    dev_group_set = {row["source_group"] for row in dev_rows}
    overlap = train_groups & dev_group_set
    if overlap:
        raise RuntimeError(f"Train/dev group leakage detected: {sorted(overlap)[:5]}")

    report = {
        "seed": int(args.seed),
        "dev_fraction": None if use_test_as_dev else float(args.dev_fraction),
        "use_test_as_dev": use_test_as_dev,
        "raw_jsonl": str(raw_path),
        "denoised_jsonl": str(denoised_path),
        "test_source": str(test_path),
        "pair_alignment": {
            "raw_rows": len(raw_rows),
            "denoised_rows": len(denoised_rows),
            "paired_segments": len(raw_rows),
        },
        "dev_groups": dev_groups,
        "sources": {
            "real_raw": rows_stats(raw_rows),
            "real_denoised": rows_stats(denoised_rows),
            "real_train": real_train_stats,
        },
        "synthetic": synthetic_report,
        "additional_synthetic": additional_report,
        "output": {
            "train": rows_stats(train_rows),
            "dev": rows_stats(dev_rows),
            "test": rows_stats(test_rows),
            "train_jsonl": str(train_out.resolve()),
            "dev_jsonl": str(dev_out.resolve()),
            "test_jsonl": str(test_out.resolve()),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "prepare_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return report

def main() -> None:
    report = prepare(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
