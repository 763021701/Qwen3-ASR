#!/usr/bin/env python3
"""Build noise_target_domain_curated from noise_target_domain.

Mirrors the three source subdirectories under raw/POC_train/noise/noise_target_domain.
Per source, applies the same duration rules as the legacy flat curated set:
  - keep all clips with duration > 2.0 s
  - sample up to 40 from [1.0, 2.0) s
  - sample up to 12 from [0.5, 1.0) s
Clips below 0.5 s are dropped. Output entries are symlinks to the source wav files.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SRC = REPO / "raw/POC_train/noise/noise_target_domain"
DEFAULT_OUT = REPO / "raw/POC_train/noise/noise_target_domain_curated"
SOURCE_NAMES = (
    "noise_from_pathology",
    "noise_from_pathology_0821",
    "noise_from_radiology",
)


def collect_wavs(source_root: Path) -> list[Path]:
    out: list[Path] = []
    for pattern in ("*.wav", "**/*.wav"):
        for p in source_root.glob(pattern):
            if p.is_file() or (p.is_symlink() and p.suffix.lower() == ".wav"):
                out.append(p.resolve())
    # stable unique order
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in sorted(out):
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def curate_paths(paths: list[Path], seed: int) -> tuple[list[Path], dict]:
    rng = random.Random(seed)
    by_bucket: dict[str, list[Path]] = {"gt2": [], "mid": [], "short": [], "drop": []}
    for p in paths:
        try:
            dur = sf.info(p).duration
        except (OSError, RuntimeError) as e:
            raise RuntimeError(f"failed to read {p}") from e
        if dur > 2.0:
            by_bucket["gt2"].append(p)
        elif dur >= 1.0:
            by_bucket["mid"].append(p)
        elif dur >= 0.5:
            by_bucket["short"].append(p)
        else:
            by_bucket["drop"].append(p)

    picked = list(by_bucket["gt2"])
    mid_n = min(40, len(by_bucket["mid"]))
    short_n = min(12, len(by_bucket["short"]))
    if mid_n:
        picked.extend(rng.sample(by_bucket["mid"], mid_n))
    if short_n:
        picked.extend(rng.sample(by_bucket["short"], short_n))

    stats = {
        "source_total": len(paths),
        "gt2_kept": len(by_bucket["gt2"]),
        "1_2_sampled": mid_n,
        "1_2_pool": len(by_bucket["mid"]),
        "05_1_sampled": short_n,
        "05_1_pool": len(by_bucket["short"]),
        "dropped_lt_05s": len(by_bucket["drop"]),
        "output_total": len(picked),
    }
    return sorted(picked), stats


def duration_summary(paths: list[Path]) -> dict:
    durs = [sf.info(p).duration for p in paths]
    arr = np.array(durs, dtype=np.float64)
    return {
        "min": float(arr.min()) if len(arr) else 0.0,
        "p50": float(np.percentile(arr, 50)) if len(arr) else 0.0,
        "p95": float(np.percentile(arr, 95)) if len(arr) else 0.0,
        "max": float(arr.max()) if len(arr) else 0.0,
        "mean": float(arr.mean()) if len(arr) else 0.0,
        "total": float(arr.sum()) if len(arr) else 0.0,
    }


def install_symlink(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    os.symlink(src, dest)


def rebuild(out_root: Path, src_root: Path, seed: int, dry_run: bool) -> dict:
    manifest_sources: dict = {}
    total_out = 0

    if not dry_run and out_root.exists():
        for child in out_root.iterdir():
            if child.name == "manifest.json":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            elif child.is_file() or child.is_symlink():
                child.unlink()

    for idx, name in enumerate(SOURCE_NAMES):
        src_sub = src_root / name
        if not src_sub.exists():
            raise FileNotFoundError(f"missing source subdir {src_sub}")
        wavs = collect_wavs(src_sub)
        picked, stats = curate_paths(wavs, seed=seed + idx)
        out_sub = out_root / name
        file_entries = []
        for src in picked:
            dest = out_sub / src.name
            if not dry_run:
                install_symlink(src, dest)
            file_entries.append({"src": str(src), "name": src.name})
        stats["duration_sec"] = duration_summary(picked) if picked else {}
        stats["files"] = file_entries
        manifest_sources[name] = stats
        total_out += len(picked)

    manifest = {
        "built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_root": str(src_root.resolve()),
        "output_root": str(out_root.resolve()),
        "seed_base": seed,
        "rules": {
            "gt2_sec": "keep all duration > 2.0s",
            "05_1_sec": "random sample up to 12 from [0.5, 1.0)s",
            "1_2_sec": "random sample up to 40 from [1.0, 2.0)s",
            "drop_lt_05s": "discard duration < 0.5s",
        },
        "sources": manifest_sources,
        "output_total": total_out,
    }
    if not dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-root", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    manifest = rebuild(args.out_root, args.src_root, args.seed, args.dry_run)
    print(json.dumps({k: manifest[k] for k in ("output_total", "sources")}, indent=2))
    for name in SOURCE_NAMES:
        print(
            f"{name}: {manifest['sources'][name]['output_total']}"
            f" / {manifest['sources'][name]['source_total']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())