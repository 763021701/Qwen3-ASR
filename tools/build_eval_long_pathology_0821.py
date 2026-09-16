#!/usr/bin/env python3
"""Rebuild data/eval_long_pathology_0821 from per-case continuous dictation.

The previous windows grouped short clips by g00N index across cases, so one
wav mixed unrelated specimens. This script packs consecutive clips of the
SAME case (folder id) into windows whose original-timeline span is <= --max_sec,
slices the source WAV, and writes 16 kHz mono extra-eval jsonl.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
_CLIP_RE = re.compile(r"(g\d+)_(\d+\.\d+)-(\d+\.\d+)\.wav$", re.I)
SAMPLE_RATE = 16000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--clips_jsonl",
        default=str(REPO / "raw/POC_test/pathology_0821.jsonl"),
    )
    p.add_argument(
        "--source_wav_dir",
        default="/root/autodl-tmp/workspace/dataset/Test_Samples/samples/pathology_0821/WAV",
    )
    p.add_argument(
        "--output_dir",
        default=str(REPO / "data/eval_long_pathology_0821"),
    )
    p.add_argument("--max_sec", type=float, default=45.0)
    p.add_argument("--sr", type=int, default=SAMPLE_RATE)
    return p.parse_args()


def load_clips(path: Path) -> dict[str, list[dict]]:
    by_case: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            audio = Path(str(row.get("audio_path") or row.get("audio") or ""))
            match = _CLIP_RE.search(audio.name)
            if match is None:
                raise ValueError(f"cannot parse clip timestamps: {audio}")
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            by_case[audio.parent.name].append(
                {
                    "seg": match.group(1),
                    "start": float(match.group(2)),
                    "end": float(match.group(3)),
                    "text": text,
                }
            )
    for clips in by_case.values():
        clips.sort(key=lambda c: (c["start"], c["end"]))
    return by_case


def pack_windows(clips: list[dict], max_sec: float) -> list[list[dict]]:
    windows: list[list[dict]] = []
    i = 0
    while i < len(clips):
        j = i
        while j + 1 < len(clips) and clips[j + 1]["end"] - clips[i]["start"] <= max_sec:
            j += 1
        windows.append(clips[i : j + 1])
        i = j + 1
    return windows


def slice_wav(path: Path, start: float, end: float, sr: int) -> np.ndarray:
    wav, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    n = wav.shape[0]
    a = max(0, min(n, int(round(start * file_sr))))
    b = max(a + 1, min(n, int(round(end * file_sr))))
    chunk = wav[a:b]
    if file_sr == sr:
        return chunk
    return librosa.resample(chunk, orig_sr=file_sr, target_sr=sr)


def main() -> None:
    args = parse_args()
    clips_jsonl = Path(args.clips_jsonl).resolve()
    src_dir = Path(args.source_wav_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    wav_dir = out_dir / "wav"
    if wav_dir.exists():
        shutil.rmtree(wav_dir)
    wav_dir.mkdir(parents=True, exist_ok=True)

    by_case = load_clips(clips_jsonl)
    rows = []
    for case in sorted(by_case):
        src = src_dir / f"{case}.wav"
        if not src.is_file():
            raise FileNotFoundError(src)
        for idx, window in enumerate(pack_windows(by_case[case], args.max_sec)):
            start, end = window[0]["start"], window[-1]["end"]
            audio = slice_wav(src, start, end, args.sr)
            name = f"{case}_win{idx:03d}.wav"
            dest = wav_dir / name
            sf.write(str(dest), audio, args.sr)
            rel = dest.relative_to(REPO).as_posix()
            text = " ".join(c["text"] for c in window)
            rows.append({"audio": rel, "text": text})
            print(
                f"{name} clips={len(window)} span={end - start:.2f}s "
                f"out={audio.shape[0] / args.sr:.2f}s segs="
                + ",".join(c["seg"] for c in window)
            )

    eval_path = out_dir / "eval.jsonl"
    with eval_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} windows -> {eval_path}")


if __name__ == "__main__":
    main()
