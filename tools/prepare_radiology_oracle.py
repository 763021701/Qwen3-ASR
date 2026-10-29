#!/usr/bin/env python3
"""Build radiology-only oracle manifests: 8/2 split by recording.

Reads raw/POC_test/radiology.csv (95 labelled segments from 10 recordings),
holds out whole recordings whose total duration is closest to 20% of the
corpus, and writes train/holdout jsonl for a from-base, 1-epoch oracle SFT.

Default holdout: DocB/1866S + DocC/1435R (4.08 min of 21.00 min = 19.4%).
The long recordings (samples/radiology/wav/<source>.wav + .txt) are written to
holdout_long.jsonl for an optional long-audio evaluation; they are not training
rows and must not be mixed into train.jsonl.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO / "raw/POC_test/radiology.csv"
DEFAULT_WAV_ROOT = Path(
    "/root/autodl-tmp/workspace/dataset/Test_Samples/samples/radiology/wav"
)
DEFAULT_HOLDOUT = "DocB/1866S,DocC/1435R"
ASR_PREFIX = "language None<asr_text>"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=str(DEFAULT_CSV))
    p.add_argument("--wav_root", default=str(DEFAULT_WAV_ROOT))
    p.add_argument("--holdout", default=DEFAULT_HOLDOUT,
                   help="Comma-separated recording ids, e.g. DocB/1866S,DocC/1435R.")
    p.add_argument("--out_dir", default="data/poc_radiology_oracle")
    p.add_argument("--noise_aug", type=int, default=0, choices=(0, 1),
                   help="Per-row noise_aug flag (radiology audio is raw -> 0).")
    return p.parse_args()


def label(body: str) -> str:
    s = " ".join(str(body or "").split())
    if not s:
        raise ValueError("empty transcript")
    return ASR_PREFIX + s


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    holdout = [s.strip() for s in args.holdout.split(",") if s.strip()]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    by_source: dict[str, list[dict]] = defaultdict(list)
    with open(args.csv, "r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            by_source[str(row["source"]).strip()].append(row)

    missing = [name for name in holdout if name not in by_source]
    if missing:
        raise ValueError(f"holdout recordings not in {args.csv}: {missing}")

    train_rows: list[dict] = []
    holdout_rows: list[dict] = []
    for source, rows in sorted(by_source.items()):
        target = holdout_rows if source in holdout else train_rows
        for row in rows:
            audio = Path(str(row["audio_path"])).resolve()
            if not audio.is_file():
                raise FileNotFoundError(f"missing audio: {audio}")
            target.append(
                {
                    "audio": str(audio),
                    "text": label(row["text"]),
                    "aug": "1",
                    "noise_aug": str(args.noise_aug),
                    "sampling_source": "radiology",
                    "source_group": source,
                    "segment_id": f"{source}/{audio.name}",
                    "duration_sec": str(row["duration"]),
                }
            )

    long_rows: list[dict] = []
    wav_root = Path(args.wav_root)
    for source in holdout:
        wav = wav_root / f"{source}.wav"
        txt = wav.with_suffix(".txt")
        if not wav.is_file() or not txt.is_file():
            raise FileNotFoundError(f"missing long audio/transcript pair: {wav}, {txt}")
        transcript = " ".join(txt.read_text(encoding="utf-8").split())
        long_rows.append(
            {
                "audio": str(wav),
                "text": label(transcript),
                "aug": "0",
                "noise_aug": "0",
                "sampling_source": "radiology_long",
                "source_group": source,
                "segment_id": f"{source}/full",
            }
        )

    write_jsonl(out / "train.jsonl", train_rows)
    write_jsonl(out / "holdout.jsonl", holdout_rows)
    write_jsonl(out / "holdout_long.jsonl", long_rows)

    def dur(rows: list[dict]) -> float:
        return sum(float(r["duration_sec"]) for r in rows)

    split = {
        "holdout_recordings": holdout,
        "train_recordings": sorted(set(by_source) - set(holdout)),
        "n_train": len(train_rows),
        "n_holdout": len(holdout_rows),
        "duration_train_sec": round(dur(train_rows), 3),
        "duration_holdout_sec": round(dur(holdout_rows), 3),
        "holdout_duration_share": round(
            dur(holdout_rows) / (dur(train_rows) + dur(holdout_rows)), 4
        ),
        "noise_aug": args.noise_aug,
        "source_csv": str(Path(args.csv).resolve()),
    }
    (out / "split.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        "train: %d rows, %.2f min | holdout: %d rows, %.2f min (%.1f%%)"
        % (
            len(train_rows),
            dur(train_rows) / 60,
            len(holdout_rows),
            dur(holdout_rows) / 60,
            split["holdout_duration_share"] * 100,
        )
    )
    print("holdout recordings:", ", ".join(holdout))
    print("wrote:", out / "train.jsonl", out / "holdout.jsonl", out / "holdout_long.jsonl")


if __name__ == "__main__":
    main()