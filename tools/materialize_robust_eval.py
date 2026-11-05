#!/usr/bin/env python3
# coding=utf-8
"""Materialize a deterministic robust ASR eval manifest from a clean JSONL."""

from __future__ import annotations

import argparse
import json
import re
import sys
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import soundfile as sf

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from finetuning.augmentation.service import SimulationConfig, augment_clean_sample
from finetuning.augmentation.simulator import (
    NoisePool,
    RIRPool,
    resolve_scenario_weights,
)

DEFAULT_SCENARIOS = (
    "noise",
    "far_field",
    "obstructed",
    "echo_reverb",
    "recording",
    "distortion",
    "dropout",
    "far_field+noise",
    "echo_reverb+noise",
    "obstructed+noise",
)

_SAFE_NAME = re.compile(r"[^0-9A-Za-z._-]+")


def _stable_seed(value: str, base_seed: int) -> int:
    return (zlib.crc32(value.encode("utf-8")) + int(base_seed)) & ((1 << 63) - 1)


def _load_rows(path: str, max_samples: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def _safe_stem(path: Path) -> str:
    stem = _SAFE_NAME.sub("_", path.stem).strip("._")
    return stem or "audio"


def materialize_robust_eval(
    rows: Sequence[Dict[str, Any]],
    output_dir: str,
    *,
    scenarios: Sequence[str] = DEFAULT_SCENARIOS,
    severity: Optional[float] = None,
    severity_min: float = 0.2,
    severity_max: float = 0.8,
    copies: int = 1,
    seed: int = 42,
    noise_dir: Optional[str] = None,
    rir_dir: Optional[str] = None,
    reverb_mode: str = "synthetic",
    rir_real_prob: float = 1.0,
    rir_mix_prob: float = 0.8,
    noise_cache_files: int = 64,
    rir_cache_files: int = 128,
    audio_format: str = "flac",
    keep_noise_flag: bool = False,
) -> Path:
    if not rows:
        raise ValueError("No input rows.")
    if copies < 1:
        raise ValueError("copies must be >= 1.")
    scenarios = tuple(scenarios)
    if not scenarios:
        raise ValueError("At least one scenario is required.")
    if any("noise" in name.split("+") for name in scenarios) and not noise_dir:
        raise ValueError("noise_dir is required for noise-based scenarios.")
    if severity is not None:
        severity_min = severity_max = float(severity)
    if not 0.0 <= severity_min <= severity_max <= 1.0:
        raise ValueError("severity range must satisfy 0 <= min <= max <= 1.")
    if audio_format.lower().lstrip(".") not in {"wav", "flac"}:
        raise ValueError("audio_format must be wav or flac.")
    audio_format = audio_format.lower().lstrip(".")

    out_root = Path(output_dir)
    audio_dir = out_root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    config = SimulationConfig(
        spool_dir=str(out_root / "_spool"),
        scenarios=scenarios,
        clean_prob=0.0,
        noise_dir=noise_dir,
        noise_cache_files=noise_cache_files,
        reverb_mode=reverb_mode,
        rir_dir=rir_dir,
        rir_cache_files=rir_cache_files,
        rir_real_prob=rir_real_prob,
        rir_mix_prob=rir_mix_prob,
        sample_rate=16000,
        audio_format=audio_format,
        severity_min=severity_min,
        severity_max=severity_max,
        seed=seed,
        num_simulators=1,
        stats_log_interval_sec=0.0,
    )
    noise_pool = NoisePool(
        noise_dir,
        sample_rate=config.sample_rate,
        max_cached_files=noise_cache_files,
    )
    rir_pool = None
    if reverb_mode != "synthetic":
        rir_pool = RIRPool(
            rir_dir,
            sample_rate=config.sample_rate,
            max_cached_files=rir_cache_files,
            real_prob=rir_real_prob,
        )
        if not rir_pool.available:
            raise RuntimeError(f"No RIR files found under {rir_dir!r}.")
    weights = resolve_scenario_weights(scenarios, None)
    manifest: List[Dict[str, Any]] = []
    scenario_counts: Counter = Counter()
    for index, row in enumerate(rows):
        if "audio" not in row or "text" not in row:
            raise ValueError(f"Row {index} requires audio and text fields.")
        source_audio = Path(str(row["audio"]))
        if not source_audio.is_file():
            raise FileNotFoundError(f"Row {index}: missing audio {source_audio}")
        for copy_index in range(copies):
            sample = dict(row)
            sample["noise_aug"] = 1 if not keep_noise_flag else sample.get("noise_aug", 1)
            sample["sample_seed"] = _stable_seed(
                f"{source_audio.resolve()}:{copy_index}", seed
            )
            forced_scenario = scenarios[(index * copies + copy_index) % len(scenarios)]
            audio, metadata = augment_clean_sample(
                sample,
                config,
                noise_pool,
                weights,
                rir_pool,
                forced_scenario=forced_scenario,
            )
            scenario = str(metadata["scenario"])
            scenario_counts[scenario] += 1
            filename = (
                f"{index:06d}_{_safe_stem(source_audio)}"
                f"_c{copy_index}_{_SAFE_NAME.sub('_', scenario)}.{audio_format}"
            )
            target = audio_dir / filename
            sf.write(
                str(target),
                np.asarray(audio, dtype=np.float32),
                config.sample_rate,
                format="WAV" if audio_format == "wav" else "FLAC",
                subtype="PCM_16",
            )
            item = dict(row)
            item["audio"] = str(target.resolve())
            item["source_audio"] = str(source_audio)
            item["noise_aug"] = 1 if not keep_noise_flag else item.get("noise_aug", 1)
            item["scenario"] = scenario
            item["severity"] = float(metadata["severity"])
            item["simulated"] = bool(metadata["simulated"])
            item["sim_version"] = str(metadata["sim_version"])
            item["sample_seed"] = int(metadata["sample_seed"])
            manifest.append(item)

    manifest_path = out_root / "eval.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for item in manifest:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[robust-eval] rows={len(rows)} copies={copies} samples={len(manifest)}")
    print(f"[robust-eval] scenarios={dict(sorted(scenario_counts.items()))}")
    print(f"[robust-eval] manifest={manifest_path}")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True, help="Clean eval JSONL.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--scenarios",
        default=",".join(DEFAULT_SCENARIOS),
        help="Comma-separated mega-asr-sim scenarios.",
    )
    parser.add_argument("--severity", type=float, default=None)
    parser.add_argument("--severity_min", type=float, default=0.2)
    parser.add_argument("--severity_max", type=float, default=0.8)
    parser.add_argument("--copies", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise_dir", default=None)
    parser.add_argument("--rir_dir", default=None)
    parser.add_argument(
        "--reverb_mode", default="synthetic", choices=("synthetic", "real", "mixed")
    )
    parser.add_argument("--rir_real_prob", type=float, default=1.0)
    parser.add_argument("--rir_mix_prob", type=float, default=0.8)
    parser.add_argument("--noise_cache_files", type=int, default=64)
    parser.add_argument("--rir_cache_files", type=int, default=128)
    parser.add_argument("--audio_format", default="flac", choices=("wav", "flac"))
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument(
        "--keep_noise_flag",
        action="store_true",
        help="Preserve input noise_aug instead of forcing noise augmentation on.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = tuple(
        item.strip() for item in args.scenarios.split(",") if item.strip()
    )
    rows = _load_rows(args.jsonl, args.max_samples)
    materialize_robust_eval(
        rows,
        args.output_dir,
        scenarios=scenarios,
        severity=args.severity,
        severity_min=args.severity_min,
        severity_max=args.severity_max,
        copies=args.copies,
        seed=args.seed,
        noise_dir=args.noise_dir,
        rir_dir=args.rir_dir,
        reverb_mode=args.reverb_mode,
        rir_real_prob=args.rir_real_prob,
        rir_mix_prob=args.rir_mix_prob,
        noise_cache_files=args.noise_cache_files,
        rir_cache_files=args.rir_cache_files,
        audio_format=args.audio_format,
        keep_noise_flag=args.keep_noise_flag,
    )


if __name__ == "__main__":
    main()
