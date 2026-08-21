#!/usr/bin/env python3
"""Generate zero-amplitude silence WAV files and metadata.csv for ASR training."""

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf


def main() -> None:
    p = argparse.ArgumentParser(description="Generate silence WAV dataset")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--num_samples", type=int, default=1000)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--min_sec", type=float, default=2.0)
    p.add_argument("--max_sec", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    wav_dir = args.out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    mean = (args.min_sec + args.max_sec) / 2.0
    std = (args.max_sec - args.min_sec) / 4.0
    durations = np.clip(mean + std * rng.standard_normal(args.num_samples), args.min_sec, args.max_sec)

    rows = []
    for i, dur in enumerate(durations):
        n_samples = int(round(dur * args.sr))
        actual_dur = n_samples / args.sr
        wav_path = wav_dir / f"silence_{i:06d}.wav"
        sf.write(wav_path, np.zeros(n_samples, dtype=np.float32), args.sr)
        rows.append(
            {
                "audio_path": str(wav_path.resolve()),
                "text": "",
            }
        )

    csv_path = args.out_dir / "metadata.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "text"])
        writer.writeheader()
        writer.writerows(rows)

    durs = [n_samples / args.sr for n_samples in [int(round(d * args.sr)) for d in durations]]
    print(f"Generated {len(rows)} silence wav files in {wav_dir}")
    print(f"Wrote {csv_path}")
    print(f"Duration: min={min(durs):.3f}s max={max(durs):.3f}s mean={np.mean(durs):.3f}s")


if __name__ == "__main__":
    main()
