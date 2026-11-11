#!/usr/bin/env python3
# coding=utf-8
"""Stage the 18 local shards used by the 200-hour multilingual pilot."""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from finetuning.data_pipeline.providers import ModelScopeShardProvider
from finetuning.data_pipeline.spec import load_data_config

PILOT_SHARDS = {
    "gigaspeech": [
        "webdataset/gigaspeech-train-03312-03319.tar",
        "webdataset/gigaspeech-train-06672-06679.tar",
        "webdataset/gigaspeech-train-00016-00023.tar",
        "webdataset/gigaspeech-train-00024-00031.tar",
    ],
    "wenetspeech": [
        "TRAIN/wenetspeech-L-00000746-00000747.tar",
        "TRAIN/wenetspeech-L-00000819-00000820.tar",
        "TRAIN/wenetspeech-L-00000002-00000003.tar",
        "TRAIN/wenetspeech-L-00000004-00000005.tar",
        "TRAIN/wenetspeech-L-00000006-00000007.tar",
    ],
    "magicdata_read": ["MAGICDATA-Mandarin-Read/magicdata-read-train-00002.tar"],
    "magicdata_conv": [
        "MAGICDATA-Mandarin-Conversational/magicdata-conversational-train-00000.tar"
    ],
    "magicdata_heavy": [
        "MAGICDATA-Mandarin-Heavy-Accent/magicdata-heavy-accent-train-00000.tar"
    ],
    "kespeech": ["kespeech-train_phase2-00001.tar"],
    "chuan": ["labeled/audios_042-005.tar", "labeled/audios_047-005.tar"],
    "yue": [
        "labeled/part_040-001.tar",
        "labeled/part_062-001.tar",
        "labeled/part_001-000.tar",
    ],
}


def main() -> None:
    config = load_data_config(
        str(_REPO_ROOT / "configs/data_configs/robust_multilingual_pilot_200h.yaml")
    )
    out_root = _REPO_ROOT / "data/pilot_shards"
    out_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    providers = []
    for source in config.sources:
        provider = ModelScopeShardProvider(source, config.cache)
        providers.append(provider)
        index = provider.index_source()
        by_remote = {shard.remote_path: shard for shard in index.shards}
        out_dir = out_root / source.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for remote in PILOT_SHARDS[source.name]:
            shard = by_remote.get(remote)
            if shard is None:
                raise RuntimeError(f"{source.name}: missing remote shard {remote}")
            jobs.append((source.name, provider, shard, out_dir))
    print(f"staging {len(jobs)} shards", flush=True)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(provider.ensure_local, shard): (name, shard, out_dir)
                for name, provider, shard, out_dir in jobs
            }
            for future in as_completed(futures):
                name, shard, out_dir = futures[future]
                path = future.result()
                target = out_dir / shard.path.name
                if not target.exists():
                    target.symlink_to(path.resolve())
                print(f"OK {name} {shard.remote_path}", flush=True)
    finally:
        for provider in providers:
            provider.close()


if __name__ == "__main__":
    main()
