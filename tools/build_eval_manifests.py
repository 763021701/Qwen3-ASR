#!/usr/bin/env python3
"""Rebuild evaluation manifests from raw/POC_test (labels are the raw labels)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HOLDOUT_SOURCES = {"DocB/1866S", "DocC/1435R"}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="data/eval_unified7")
    args = ap.parse_args()
    out = Path(args.out_dir).resolve()

    radiology = []
    with (REPO / "raw/POC_test/radiology.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for item in csv.DictReader(handle):
            audio = Path(item["audio_path"])
            if not audio.is_absolute():
                audio = (REPO / "raw/POC_test" / audio).resolve()
            if not audio.is_file():
                raise FileNotFoundError(audio)
            radiology.append({"audio": str(audio), "text": item["text"]})
    holdout = [row for row in radiology
               if "/".join(row["audio"].split("/output/", 1)[-1].split("/")[:2]) in HOLDOUT_SOURCES]
    if len(radiology) != 95 or len(holdout) != 19:
        raise ValueError(f"expected 95/19, got {len(radiology)}/{len(holdout)}")

    write_jsonl(out / "radiology95.jsonl", radiology)
    write_jsonl(out / "holdout19.jsonl", holdout)
    print(f"[write] radiology95={len(radiology)} holdout19={len(holdout)} -> {out}")


if __name__ == "__main__":
    main()
