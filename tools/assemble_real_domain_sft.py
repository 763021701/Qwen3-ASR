#!/usr/bin/env python3
"""Assemble POC train manifests with extra REAL-domain sources.

Base rows come verbatim from an existing prepare output (train.jsonl). Real
extra sources (youtube medical, GigaSpeech) are appended with noise_aug=0 —
they are real recordings, matching metadata_raw.jsonl treatment — and the
same ASR target wrapping as every other row.

GigaSpeech has no duration field, so clip duration is estimated from text
length (~15 chars/s) purely for sampling; --gigaspeech_samples N writes
N independently-sampled manifests (stage_1..stage_N) for per-epoch training
stages, each of --gigaspeech_hours hours.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

ASR_PREFIX = "language None<asr_text>"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            item = json.loads(raw)
            audio = str(item.get("audio") or item.get("audio_path") or item.get("wav_path") or "").strip()
            text = str(item.get("text") or "").strip()
            if not audio or not text:
                raise ValueError(f"{path}:{line_no}: missing audio/text")
            rows.append(item | {"_audio": audio, "_text": text})
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return rows


def real_row(audio: str, text: str, source: str, duration: float, index: int) -> dict:
    stem = Path(audio).stem
    return {
        "audio": audio,
        "text": f"{ASR_PREFIX}{text}",
        "aug": 1,
        "noise_aug": 0,
        "sampling_source": source,
        "source_group": source,
        "segment_id": f"{source}/{stem}_{index:06d}",
        "duration_sec": round(duration, 3),
    }


def check_audio(rows: list[dict], label: str) -> None:
    missing = [r["_audio"] for r in rows if not os.path.isfile(r.get("_audio") or r.get("audio", ""))]
    if missing:
        raise FileNotFoundError(f"{label}: {len(missing)} missing audio, e.g. {missing[:3]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_data_dir", required=True, help="Existing prepare output with train/dev/test.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--youtube_manifest", default="")
    parser.add_argument("--gigaspeech_jsonl", default="")
    parser.add_argument("--gigaspeech_hours", type=float, default=100.0)
    parser.add_argument("--gigaspeech_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_dir = Path(args.base_data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_rows = read_jsonl(base_dir / "train.jsonl")
    print(f"[base] {len(base_rows)} rows from {base_dir / 'train.jsonl'}")
    for name in ("dev.jsonl", "test.jsonl"):
        shutil.copy(base_dir / name, out_dir / name)

    extra_rows: list[dict] = []
    if args.youtube_manifest:
        yt = read_jsonl(Path(args.youtube_manifest))
        check_audio(yt, "youtube")
        for i, r in enumerate(yt):
            extra_rows.append(
                real_row(r["_audio"], r["_text"], "youtube_med_en", float(r.get("duration_sec") or 0.0), i)
            )
        yt_hours = sum(r["duration_sec"] for r in extra_rows if r["sampling_source"] == "youtube_med_en") / 3600
        print(f"[youtube] {len(yt)} rows, {yt_hours:.1f}h")

    giga_stage_rows: list[list[dict]] = []
    if args.gigaspeech_jsonl:
        giga = read_jsonl(Path(args.gigaspeech_jsonl))
        # text-length duration estimate (~15 chars/s English)
        durations = [max(len(r["_text"]), 10) / 15.0 for r in giga]
        total_h = sum(durations) / 3600
        print(f"[gigaspeech] {len(giga)} rows, estimated {total_h:.0f}h")
        target_sec = args.gigaspeech_hours * 3600
        for sample_idx in range(args.gigaspeech_samples):
            rng = random.Random(args.seed + sample_idx)
            order = list(range(len(giga)))
            rng.shuffle(order)
            picked, acc = [], 0.0
            for idx in order:
                picked.append(idx)
                acc += durations[idx]
                if acc >= target_sec:
                    break
            rows = []
            for i, idx in enumerate(picked):
                r = giga[idx]
                rows.append(real_row(r["_audio"], r["_text"], "gigaspeech", durations[idx], i))
            check_audio(rows, f"gigaspeech sample {sample_idx + 1}")
            giga_stage_rows.append(rows)
            print(f"[gigaspeech] sample {sample_idx + 1}: {len(rows)} rows, {acc / 3600:.1f}h")

    if not giga_stage_rows:
        rows = base_rows + extra_rows
        with (out_dir / "train.jsonl").open("w", encoding="utf-8") as handle:
            for r in rows:
                handle.write(json.dumps(r, ensure_ascii=False) + "\n")
        hours = sum(float(r.get("duration_sec") or 0) for r in rows) / 3600
        print(f"[out] {out_dir / 'train.jsonl'}: {len(rows)} rows, ~{hours:.1f}h")
        return

    for sample_idx, giga_rows in enumerate(giga_stage_rows, start=1):
        stage_dir = out_dir / f"stage_{sample_idx}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(base_dir / "dev.jsonl", stage_dir / "dev.jsonl")
        shutil.copy(base_dir / "test.jsonl", stage_dir / "test.jsonl")
        rows = base_rows + extra_rows + giga_rows
        with (stage_dir / "train.jsonl").open("w", encoding="utf-8") as handle:
            for r in rows:
                handle.write(json.dumps(r, ensure_ascii=False) + "\n")
        hours = sum(float(r.get("duration_sec") or 0) for r in rows) / 3600
        print(f"[out] {stage_dir / 'train.jsonl'}: {len(rows)} rows, ~{hours:.1f}h")


if __name__ == "__main__":
    main()
