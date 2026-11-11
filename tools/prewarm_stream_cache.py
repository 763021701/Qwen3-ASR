#!/usr/bin/env python3
# coding=utf-8
"""Pre-download the first shard of every streaming source.

The training reader walks sources serially: its first ``next()`` on each
source blocks on that source's first shard download. With 4GB shards on a
cold cache this can stall the spool for tens of minutes, so warm the exact
shard order used by ``LocalWebDataset`` before launching training.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from finetuning.data_pipeline import build_data_pipeline, load_data_config
from finetuning.data_pipeline.index import split_shards_for_worker
from finetuning.data_pipeline.labels import stable_seed
from finetuning.data_pipeline.mixer import per_source_target_hours, select_source_shards
from finetuning.data_pipeline.providers import ModelScopeShardProvider


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_config", required=True)
    p.add_argument("--shards_per_source", type=int, default=1)
    return p.parse_args()


def first_shards(source, index, config, target, count: int):
    selected = select_source_shards(
        source, index.shards, target, config.mixture, pass_index=0
    )
    worker_shards = split_shards_for_worker(selected, 0, 1)
    rng = random.Random(stable_seed(config.seed, source.name, "worker", 0))
    rng.shuffle(worker_shards)
    return worker_shards[:count]


def main() -> None:
    args = parse_args()
    if args.shards_per_source < 1:
        raise ValueError("--shards_per_source must be >= 1.")
    config = load_data_config(args.data_config)
    pipeline = build_data_pipeline(config)

    targets = per_source_target_hours(config.sources, config.mixture.total_hours)
    jobs = []
    providers = {}
    for source, index, target in zip(config.sources, pipeline.source_indexes, targets):
        if index is None or source.provider != "modelscope":
            continue
        providers[source.name] = ModelScopeShardProvider(source, config.cache)
        for shard in first_shards(
            source, index, config, target, args.shards_per_source
        ):
            jobs.append((source, shard))
    if not jobs:
        print("[prewarm] no ModelScope shards to download")
        return

    total_bytes = sum(shard.size_bytes for _, shard in jobs)
    print(
        f"[prewarm] {len(jobs)} shard(s), {total_bytes / 1e9:.2f}GB, "
        f"{len(providers)} source(s)"
    )
    started = time.time()

    def fetch(source, shard):
        path = providers[source.name].ensure_local(shard)
        return source.name, shard.remote_path, path.stat().st_size

    try:
        with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as pool:
            futures = [pool.submit(fetch, source, shard) for source, shard in jobs]
            for future in as_completed(futures):
                name, remote, size = future.result()
                print(f"[prewarm] {name}: {remote} ({size / 1e9:.2f}GB)")
    finally:
        for provider in providers.values():
            provider.close()
    print(f"[prewarm] done in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
