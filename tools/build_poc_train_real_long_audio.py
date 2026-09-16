#!/usr/bin/env python3
"""Rebuild data/poc_train_real_long_audio from from_lid_raw clips only.

The previous windows interleaved from_lid_raw with from_raw, so one utterance
played the same dictation twice. This script concatenates consecutive
from_lid_raw clips of the SAME case until duration >= --target_sec.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
ASR_PREFIX = "language None<asr_text>"
SAMPLE_RATE = 16000
SOURCE_DIR_NAME = "from_lid_raw"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--clips_jsonl",
        default=str(REPO / "raw/POC_train/real_target_domain/metadata_raw.jsonl"),
    )
    p.add_argument(
        "--output_dir",
        default=str(REPO / "data/poc_train_real_long_audio"),
    )
    p.add_argument("--target_sec", type=float, default=30.0)
    p.add_argument("--drop_short_sec", type=float, default=10.0)
    p.add_argument("--sr", type=int, default=SAMPLE_RATE)
    return p.parse_args()


def load_clips(path: Path) -> dict[str, list[dict]]:
    by_case: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            audio = Path(str(row.get("audio_path") or row.get("audio") or ""))
            if audio.parent.name != SOURCE_DIR_NAME:
                continue
            if "_seg_" not in audio.stem:
                raise ValueError(f"{path}:{line_no}: unexpected stem {audio.stem}")
            case, idx = audio.stem.split("_seg_", 1)
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            if not audio.is_file():
                raise FileNotFoundError(audio)
            by_case[case].append(
                {
                    "idx": int(idx),
                    "text": text,
                    "dur": float(row["duration"]),
                    "audio": audio,
                }
            )
    if not by_case:
        raise ValueError(f"No {SOURCE_DIR_NAME} clips in {path}")
    for clips in by_case.values():
        clips.sort(key=lambda c: c["idx"])
    return by_case


def pack_windows(
    clips: list[dict], target_sec: float, drop_short_sec: float
) -> list[list[dict]]:
    windows: list[list[dict]] = []
    i = 0
    while i < len(clips):
        j = i
        dur = 0.0
        while j < len(clips) and dur < target_sec:
            dur += clips[j]["dur"]
            j += 1
        windows.append(clips[i:j])
        i = j
    return [w for w in windows if sum(c["dur"] for c in w) >= drop_short_sec]


def load_mono(path: Path, sr: int) -> np.ndarray:
    wav, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if file_sr == sr:
        return wav
    return librosa.resample(wav, orig_sr=file_sr, target_sr=sr)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    wav_dir = out_dir / "wav"
    if wav_dir.exists():
        shutil.rmtree(wav_dir)
    wav_dir.mkdir(parents=True, exist_ok=True)

    by_case = load_clips(Path(args.clips_jsonl).resolve())
    rows = []
    for case in sorted(by_case):
        for idx, window in enumerate(pack_windows(by_case[case], args.target_sec, args.drop_short_sec)):
            audio = np.concatenate([load_mono(c["audio"], args.sr) for c in window])
            stem = f"{case}_win{idx:03d}"
            dest = wav_dir / f"{stem}.wav"
            sf.write(str(dest), audio, args.sr)
            duration = round(audio.shape[0] / args.sr, 3)
            rel = dest.relative_to(REPO).as_posix()
            rows.append(
                {
                    "audio": rel,
                    "text": ASR_PREFIX + " ".join(c["text"] for c in window),
                    "aug": 1,
                    "noise_aug": 0,
                    "sampling_source": "real_long_audio",
                    "source_group": case,
                    "segment_id": f"real_long_audio/{stem}",
                    "duration_sec": duration,
                }
            )
            print(
                f"{stem}.wav clips={len(window)} dur={duration:.2f}s "
                f"segs=" + ",".join(f"{c['idx']:04d}" for c in window)
            )

    train_path = out_dir / "train.jsonl"
    with train_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    hours = sum(r["duration_sec"] for r in rows) / 3600
    print(f"wrote {len(rows)} windows, {hours:.3f}h -> {train_path}")


if __name__ == "__main__":
    main()
