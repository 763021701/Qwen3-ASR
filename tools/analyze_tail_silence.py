#!/usr/bin/env python3
# coding=utf-8
"""Measure the trailing-silence distribution of a local WebDataset config."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from finetuning.data_pipeline import load_data_config
from finetuning.data_pipeline.datasets import iter_webdataset_shard
from finetuning.data_pipeline.index import index_source


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_config", required=True)
    p.add_argument("--samples_per_source", type=int, default=500)
    p.add_argument("--shards_per_source", type=int, default=2)
    p.add_argument("--frame_ms", type=float, default=25.0)
    p.add_argument("--hop_ms", type=float, default=10.0)
    p.add_argument("--top_db", type=float, default=40.0)
    p.add_argument("--output", default="outputs/tail_silence_stats.json")
    return p.parse_args()


def trailing_silence(
    wav: np.ndarray,
    sample_rate: int,
    top_db: float,
    frame_ms: float,
    hop_ms: float,
) -> float:
    """Seconds of trailing audio below ``top_db`` relative to the loudest frame."""
    x = np.asarray(wav, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return 0.0
    x = x - float(x.mean())
    frame = max(1, int(sample_rate * frame_ms / 1000.0))
    hop = max(1, int(sample_rate * hop_ms / 1000.0))
    if x.size < frame:
        return x.size / sample_rate
    count = 1 + (x.size - frame) // hop
    starts = np.arange(count, dtype=np.int64) * hop
    ends = starts + frame
    energy = np.concatenate(([0.0], np.cumsum(x.astype(np.float64) ** 2)))
    rms = np.sqrt((energy[ends] - energy[starts]) / float(frame))
    peak = float(rms.max()) if rms.size else 0.0
    if peak <= 0.0:
        return x.size / sample_rate
    threshold = peak * (10.0 ** (-top_db / 20.0))
    active = np.flatnonzero(rms > threshold)
    if active.size == 0:
        return x.size / sample_rate
    end = min(x.size, int(active[-1]) * hop + frame)
    return max(0.0, (x.size - end) / sample_rate)


def histogram(values: List[float]) -> Dict[str, int]:
    bins = [
        (0.0, 0.0),
        (0.0, 0.05),
        (0.05, 0.1),
        (0.1, 0.2),
        (0.2, 0.5),
        (0.5, 1.0),
        (1.0, 2.0),
        (2.0, None),
    ]
    result = {}
    for index, (low, high) in enumerate(bins):
        if index == 0:
            count = sum(1 for value in values if value <= 0.0)
        elif high is None:
            count = sum(1 for value in values if value > low)
        else:
            count = sum(1 for value in values if low < value <= high)
        label = "0" if index == 0 else (
            f">{low:g}" if high is None else f"({low:g},{high:g}]"
        )
        result[label] = count
    return result


def quantiles(values: List[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()) if array.size else 0.0,
        "median": float(np.percentile(array, 50)) if array.size else 0.0,
        "p75": float(np.percentile(array, 75)) if array.size else 0.0,
        "p90": float(np.percentile(array, 90)) if array.size else 0.0,
        "p95": float(np.percentile(array, 95)) if array.size else 0.0,
        "p99": float(np.percentile(array, 99)) if array.size else 0.0,
        "max": float(array.max()) if array.size else 0.0,
    }


def source_samples(source, index, count: int, shards_per_source: int) -> Iterable:
    shards = list(index.shards[: max(1, shards_per_source)])
    if not shards:
        return
    per_shard = max(1, int(math.ceil(count / len(shards))))
    for shard in shards:
        for sample in itertools.islice(
            iter_webdataset_shard(shard.path, source, target_sr=16000),
            per_shard,
        ):
            yield sample


def main() -> None:
    args = parse_args()
    if args.samples_per_source <= 0:
        raise ValueError("--samples_per_source must be > 0.")
    config = load_data_config(args.data_config)
    results: Dict[str, object] = {
        "data_config": str(Path(args.data_config).resolve()),
        "top_db": args.top_db,
        "frame_ms": args.frame_ms,
        "hop_ms": args.hop_ms,
        "samples_per_source": args.samples_per_source,
        "sources": {},
    }
    print(
        f"{'source':18s} {'n':>5s} {'mean':>7s} {'median':>7s} "
        f"{'p90':>7s} {'p99':>7s} {'max':>7s} {'>0.2':>7s} {'>0.5':>7s} {'>1s':>7s}"
    )
    for source in config.sources:
        if source.provider != "local":
            print(f"{source.name:18s} skipped (not local)")
            continue
        index = index_source(source)
        values: List[float] = []
        for sample in source_samples(
            source, index, args.samples_per_source, args.shards_per_source
        ):
            audio = np.asarray(sample["audio"], dtype=np.float32)
            values.append(
                trailing_silence(
                    audio,
                    config.sample_rate,
                    args.top_db,
                    args.frame_ms,
                    args.hop_ms,
                )
            )
        if not values:
            print(f"{source.name:18s} no samples")
            continue
        stats = quantiles(values)
        total = len(values)
        stats["n"] = total
        stats["fraction_gt_0.2"] = sum(v > 0.2 for v in values) / total
        stats["fraction_gt_0.5"] = sum(v > 0.5 for v in values) / total
        stats["fraction_gt_1.0"] = sum(v > 1.0 for v in values) / total
        stats["histogram"] = histogram(values)
        results["sources"][source.name] = stats
        print(
            f"{source.name:18s} {total:5d} {stats['mean']:7.3f} {stats['median']:7.3f} "
            f"{stats['p90']:7.3f} {stats['p99']:7.3f} {stats['max']:7.3f} "
            f"{stats['fraction_gt_0.2']:7.2%} {stats['fraction_gt_0.5']:7.2%} "
            f"{stats['fraction_gt_1.0']:7.2%}"
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
