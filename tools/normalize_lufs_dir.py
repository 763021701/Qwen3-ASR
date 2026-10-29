#!/usr/bin/env python3
"""Loudness-normalise a simulation output directory to a target LUFS.

Real radiology recordings measure ~-22.1 LUFS; some Mega-ASR scenarios (far_field
-32.6, echo_reverb -26.7) are much quieter, which is a domain mismatch ASR training
does not need to learn. This rewrites <in>/<scenario>/*.wav at target loudness
(peak-safe) and emits a new manifest.jsonl in --out-dir.

    python tools/normalize_lufs_dir.py --in-dir data/sim_radiology/simB1_ep1 \
        --out-dir data/sim_radiology/simB1n_ep1 --target -22.0 --workers 16
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from tqdm import tqdm

_G = {}


def _init(target: float) -> None:
    _G["target"] = target
    _G["meter"] = pyln.Meter(16000)


def _process(task):
    rel, src, dst = task
    wav, sr = sf.read(src, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if len(wav) < int(0.4 * sr) or np.max(np.abs(wav)) < 1e-6:
        gain_db = 0.0
    else:
        try:
            cur = _G["meter"].integrated_loudness(wav.astype(np.float64))
            gain_db = _G["target"] - cur if np.isfinite(cur) else 0.0
        except Exception:
            gain_db = 0.0
    out = wav * (10.0 ** (gain_db / 20.0))
    peak = np.max(np.abs(out))
    if peak > 0.99:
        out = out / peak * 0.99
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, out.astype(np.float32), sr, subtype="PCM_16")
    return rel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--target", type=float, default=-22.0)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    manifest = in_dir / "manifest.jsonl"
    if not manifest.is_file():
        sys.exit(f"missing {manifest}")
    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    tasks = []
    new_rows = []
    for item in rows:
        rel = item["audio"]
        tasks.append((rel, str(in_dir / rel), str(out_dir / rel)))
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(args.target,)) as ex:
        for rel in tqdm(ex.map(_process, tasks), total=len(tasks)):
            pass
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"normalised {len(tasks)} clips to {args.target} LUFS -> {out_dir}")


if __name__ == "__main__":
    main()