#!/usr/bin/env python3
# coding=utf-8
"""Extract a clean eval JSONL plus local audio from WebDataset tar shards."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import soundfile as sf

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from finetuning.data_pipeline.datasets import iter_webdataset_shard
from finetuning.data_pipeline.spec import Fields, SourceSpec

_SAFE = re.compile(r"[^0-9A-Za-z._-]+")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard", action="append", required=True, help="Local .tar shard.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--language", default="Chinese")
    p.add_argument("--text_key", default="")
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument("--audio_format", default="flac", choices=("wav", "flac"))
    p.add_argument("--fields_audio", default="flac")
    p.add_argument("--fields_text", default="txt")
    p.add_argument("--fields_meta", default="json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample_rate", type=int, default=16000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_samples <= 0:
        raise ValueError("--max_samples must be > 0.")
    out_root = Path(args.output_dir)
    audio_dir = out_root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    text_key = args.text_key.strip() or None
    source = SourceSpec(
        name=args.name,
        backend="webdataset",
        path=str(args.shard[0]),
        language=args.language,
        fields=Fields(
            audio=args.fields_audio,
            text=args.fields_text,
            meta=args.fields_meta,
        ),
        text_key=text_key,
    )
    rows = []
    for shard in args.shard:
        for sample in iter_webdataset_shard(
            Path(shard),
            source,
            target_sr=args.sample_rate,
            seed=args.seed,
            skip_errors=True,
        ):
            if len(rows) >= args.max_samples:
                break
            key = str(sample["key"])
            filename = f"{len(rows):06d}_{_SAFE.sub('_', key).strip('._') or 'utt'}.{args.audio_format}"
            target = audio_dir / filename
            sf.write(
                str(target),
                sample["audio"],
                args.sample_rate,
                format="WAV" if args.audio_format == "wav" else "FLAC",
                subtype="PCM_16",
            )
            rows.append(
                {
                    "audio": str(target.resolve()),
                    "text": sample["target"],
                    "language": args.language,
                    "source": args.name,
                    "key": key,
                    "duration": sample.get("duration"),
                }
            )
        if len(rows) >= args.max_samples:
            break
    if not rows:
        raise RuntimeError("No eval samples extracted.")
    manifest = out_root / f"{args.name}.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[eval] {args.name}: {len(rows)} samples -> {manifest}")


if __name__ == "__main__":
    main()
