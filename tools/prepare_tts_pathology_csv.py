#!/usr/bin/env python3
"""Convert tts_pathology synthesis jsonl into the additional-synthetic CSV format.

Reads the two VoxCPM synthesis variants (base_plain / lora_ultimate) under
raw/POC_train/tts_pathology/ and emits one CSV consumable by
tools/prepare_real_raw_denoised_sft.py --additional_synthetic_csv
(columns: source,index,audio_path,text,original,duration).

The `source` column is the voice identity used by the max-per-text voice
downsampling: the LoRA variant's real-speaker `speaker` field when present,
otherwise the `variant` name.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def transcript_body(value: object) -> str:
    text = str(value or "").strip()
    if "<asr_text>" in text:
        text = text.split("<asr_text>", 1)[1].strip()
    return text


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            item = json.loads(raw)
            if not transcript_body(item.get("text")):
                raise ValueError(f"{path}:{line_no}: empty transcript")
            if not str(item.get("audio_path") or "").strip():
                raise ValueError(f"{path}:{line_no}: missing audio_path")
            rows.append(item)
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return rows


def voice_of(item: dict) -> str:
    return str(item.get("speaker") or item.get("variant") or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base_jsonl",
        default="raw/POC_train/tts_pathology/tts_corpus_synth_base_plain.jsonl",
    )
    parser.add_argument(
        "--lora_jsonl",
        default="raw/POC_train/tts_pathology/tts_corpus_synth_lora_ultimate.jsonl",
    )
    parser.add_argument("--test_source", default="raw/POC_test/metadata.jsonl")
    parser.add_argument(
        "--output_csv", default="raw/POC_train/tts_pathology/metadata.csv"
    )
    args = parser.parse_args()

    base_path = Path(args.base_jsonl).resolve()
    lora_path = Path(args.lora_jsonl).resolve()
    test_path = Path(args.test_source).resolve()
    output_csv = Path(args.output_csv).resolve()

    rows = read_jsonl(base_path) + read_jsonl(lora_path)

    seen_audio: set[str] = set()
    written: list[dict] = []
    voice_counts: Counter[str] = Counter()
    for item in rows:
        audio_path = str(item["audio_path"]).strip()
        if audio_path in seen_audio:
            continue  # duplicate audio across variants; the loader rejects dup paths
        seen_audio.add(audio_path)
        voice = voice_of(item)
        if not voice:
            raise ValueError(f"row without speaker/variant: {item.get('id')}")
        text = transcript_body(item["text"])
        raw_id = item.get("id", "")
        try:
            row_index = f"{int(raw_id):06d}"  # legacy corpus: numeric ids
        except (TypeError, ValueError):
            row_index = str(raw_id)  # v2 corpus: string ids like A_00215dc4...
        written.append(
            {
                "source": voice,
                "index": row_index,
                "audio_path": audio_path,
                "text": text,
                "original": text,
                "duration": f"{float(item['duration']):.3f}",
            }
        )
        voice_counts[voice] += 1

    # Leak guard: the additional-synthetic loader does no heldout-text exclusion
    # of its own, so assert disjointness against the test set here.
    test_rows = read_jsonl(test_path)
    train_texts = {item["original"] for item in written}
    test_texts = {transcript_body(row.get("text")) for row in test_rows}
    test_audio = {
        str(Path(str(row.get("audio_path") or row.get("audio"))).resolve())
        for row in test_rows
    }
    text_overlap = train_texts & test_texts
    audio_overlap = seen_audio & test_audio
    if text_overlap:
        raise ValueError(f"Train/test text overlap: {sorted(text_overlap)[:5]}")
    if audio_overlap:
        raise ValueError(f"Train/test audio overlap: {sorted(audio_overlap)[:5]}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["source", "index", "audio_path", "text", "original", "duration"]
        )
        writer.writeheader()
        writer.writerows(written)

    total_hours = sum(float(row["duration"]) for row in written) / 3600.0
    report = {
        "output_csv": str(output_csv),
        "rows": len(written),
        "unique_texts": len(train_texts),
        "duration_hours": round(total_hours, 2),
        "voice_counts": dict(sorted(voice_counts.items())),
        "test_text_overlap": len(text_overlap),
        "test_audio_overlap": len(audio_overlap),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
