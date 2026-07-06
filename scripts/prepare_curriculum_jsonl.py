#!/usr/bin/env python3
# coding=utf-8
"""Convert a curriculum stage CSV into JSONL for qwen3_asr_sft.py.

The curriculum manifests (MAGICDATA-Hybrid/curriculum/stage{k}_metadata.csv)
have columns: audio,Transcription,SpeakerID,dataset,dur_bucket,duration,stage.
The ``audio`` field is a **piece-list spec** (``W|<wav>|<start>|<end>`` and
``S|<secs>``, ``;``-joined), NOT a plain wav path; ``load_audio`` in
``qwen3_asr_sft.py`` parses it. We pass it through verbatim.

Emits JSONL with fields:
  - audio      : piece-list spec (passthrough string)
  - text       : Transcription (raw Chinese; TN applied downstream)
  - dur_bucket : kept for the BucketBatchSampler
  - duration   : kept for the sampler / length grouping
"""
import argparse
import csv
import json
import os

from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="stage{k}_metadata.csv path")
    ap.add_argument("--out", required=True, help="output jsonl path")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n_kept = n_drop = 0
    buckets = {}
    with open(args.csv, encoding="utf-8", newline="") as f, \
         open(args.out, "w", encoding="utf-8") as g:
        for row in tqdm(csv.DictReader(f), desc="convert"):
            audio = (row.get("audio") or "").strip()
            text = (row.get("Transcription") or "").strip()
            if not audio or not text:
                n_drop += 1
                continue
            bucket = (row.get("dur_bucket") or "").strip()
            g.write(json.dumps({
                "audio": audio,
                "text": text,
                "dur_bucket": bucket,
                "duration": float(row.get("duration") or 0.0),
            }, ensure_ascii=False) + "\n")
            n_kept += 1
            buckets[bucket] = buckets.get(bucket, 0) + 1

    print(f"kept={n_kept} dropped={n_drop} -> {args.out}")
    print("buckets: " + ", ".join(f"{k}={v}" for k, v in sorted(buckets.items())))


if __name__ == "__main__":
    main()
