#!/usr/bin/env python3
# coding=utf-8
"""Inspect a local data config: sources, shards, hours, and mixture plan."""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from finetuning.data_pipeline.datasets import load_manifest_rows
from finetuning.data_pipeline.factory import build_data_pipeline
from finetuning.data_pipeline.mixer import per_source_target_hours, select_source_shards
from finetuning.data_pipeline.spec import load_data_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Data config YAML.")
    parser.add_argument("--max-shards", type=int, default=5, help="Shards to print per source.")
    args = parser.parse_args()

    config = load_data_config(args.config)
    pipeline = build_data_pipeline(config)
    print(f"mode={pipeline.mode} streaming={pipeline.is_streaming}")
    if pipeline.estimated_samples is not None:
        print(f"estimated_samples={pipeline.estimated_samples}")
    if pipeline.estimated_hours is not None:
        print(f"estimated_hours={pipeline.estimated_hours:.2f}")

    targets = per_source_target_hours(config.sources, config.mixture.total_hours)
    for position, (source, target_hours) in enumerate(zip(config.sources, targets)):
        print(
            f"\n[{source.name}] backend={source.backend} provider={source.provider} "
            f"weight={source.weight}"
        )
        print(f"  path={source.path or source.path_in_repo}")
        if source.backend == "jsonl":
            print(f"  rows={len(load_manifest_rows(str(source.path)))}")
            continue
        index = pipeline.source_indexes[position]
        if index is None:
            print("  no shard index available")
            continue
        print(
            f"  shards={len(index.shards)} samples={index.total_samples} "
            f"hours={index.total_hours} bytes={index.total_bytes}"
        )
        if target_hours is not None:
            selected = select_source_shards(
                source, index.shards, target_hours, config.mixture, pass_index=0
            )
            selected_hours = sum(
                shard.hours for shard in selected if shard.hours is not None
            )
            print(
                f"  target_hours={target_hours:.2f} selected_shards={len(selected)} "
                f"selected_hours={selected_hours:.2f}"
            )
            for shard in selected[: args.max_shards]:
                hours = "?" if shard.hours is None else f"{shard.hours:.2f}h"
                print(f"    {shard.path.name} ({hours})")
        else:
            for shard in index.shards[: args.max_shards]:
                hours = "?" if shard.hours is None else f"{shard.hours:.2f}h"
                print(f"    {shard.path.name} ({hours})")


if __name__ == "__main__":
    main()
