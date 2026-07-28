#!/usr/bin/env python3
"""Extract non-speech (background-noise) segments from WAV_Seg_Tagged audio.

Builds a small domain-matched noise library by collecting the low-energy
(non-speech) portions of each clip: leading/trailing silence and inter-phrase
pauses. These carry the recording environment's background noise and are used
to augment denoised training data.

Method (dependency-free, offline):
  1. Per-frame RMS energy (librosa.feature.rms), 30 ms frames / 10 ms hop.
  2. Otsu threshold on the log-energy histogram -> speech (above) vs non-speech.
  3. Dilate speech regions by a margin so speech edges are not grabbed as noise.
  4. Keep non-speech runs >= min_dur as noise segments; save each as 16 kHz mono wav.

Conservative against speech contamination: Otsu splits at the speech/silence
valley and the speech margin excludes speech tails/heads.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import librosa
import numpy as np
import soundfile as sf


def frame_rms_db(y: np.ndarray, sr: int, frame_ms: int = 30, hop_ms: int = 10):
    frame = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    db = 20.0 * np.log10(rms + 1e-12)
    return db, hop


def otsu_threshold(db: np.ndarray):
    lo, hi = float(db.min()), float(db.max())
    if hi - lo < 1e-6:
        return None
    bins = 256
    hist, edges = np.histogram(db, bins=bins, range=(lo, hi))
    hist = hist / max(hist.sum(), 1)
    mid = (edges[:-1] + edges[1:]) / 2.0
    w0 = 0.0
    sum0 = 0.0
    sum_total = float(np.sum(mid * hist))
    best_t, best_var = None, -1.0
    for i in range(bins):
        w0 += hist[i]
        sum0 += mid[i] * hist[i]
        if w0 <= 0 or w0 >= 1:
            continue
        w1 = 1.0 - w0
        m0 = sum0 / w0
        m1 = (sum_total - sum0) / w1
        var_between = w0 * w1 * (m0 - m1) ** 2
        if var_between > best_var:
            best_var = var_between
            best_t = mid[i]
    return best_t


def extract_non_speech(y: np.ndarray, sr: int, min_dur: float = 0.3, margin: float = 0.15):
    db, hop = frame_rms_db(y, sr)
    t = otsu_threshold(db)
    if t is None:
        return []
    is_speech = db > t
    n = len(is_speech)
    mframes = int(margin * sr / hop)
    if mframes > 0:
        idx = np.arange(n)
        dil = np.zeros(n, dtype=bool)
        for i in range(n):
            lo = max(0, i - mframes)
            hi = min(n, i + mframes + 1)
            dil[i] = is_speech[lo:hi].any()
        is_speech = dil
    is_noise = ~is_speech

    segs = []
    i = 0
    while i < n:
        if is_noise[i]:
            j = i
            while j < n and is_noise[j]:
                j += 1
            start_t = i * hop / sr
            end_t = j * hop / sr
            if end_t - start_t >= min_dur:
                segs.append((int(start_t * sr), int(end_t * sr)))
            i = j
        else:
            i += 1
    return segs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio_dir", required=True, help="Dir of source wavs (e.g. WAV_Seg_Tagged).")
    p.add_argument("--output_dir", required=True, help="Dir to write noise wavs.")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--min_dur", type=float, default=0.3)
    p.add_argument("--margin", type=float, default=0.15)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    wavs = sorted(glob.glob(os.path.join(args.audio_dir, "*.wav")))
    print(f"Found {len(wavs)} source wavs in {args.audio_dir}")

    n_seg = 0
    total_dur = 0.0
    n_clip_with_noise = 0
    for wi, w in enumerate(wavs):
        try:
            y, _ = librosa.load(w, sr=args.sr, mono=True)
        except Exception as e:
            print(f"  skip {w}: {e}", file=sys.stderr)
            continue
        if len(y) < int(args.sr * 0.1):
            continue
        segs = extract_non_speech(y, args.sr, min_dur=args.min_dur, margin=args.margin)
        if not segs:
            continue
        n_clip_with_noise += 1
        for s, e in segs:
            chunk = y[s:e]
            if len(chunk) < int(args.sr * args.min_dur):
                continue
            out = os.path.join(args.output_dir, f"noise_{n_seg:06d}.wav")
            sf.write(out, chunk, args.sr, subtype="PCM_16")
            n_seg += 1
            total_dur += len(chunk) / args.sr
        if (wi + 1) % 50 == 0:
            print(f"  processed {wi+1}/{len(wavs)} | noise segs so far: {n_seg} ({total_dur:.1f}s)")

    print(
        f"\nDone. clips_with_noise={n_clip_with_noise}/{len(wavs)} | "
        f"noise_segments={n_seg} | total_duration={total_dur:.1f}s | "
        f"out={args.output_dir}"
    )


if __name__ == "__main__":
    main()
