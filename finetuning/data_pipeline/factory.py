# coding=utf-8
"""Factory for local and ModelScope-on-demand JSONL / WebDataset pipelines."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

from torch.utils.data import ConcatDataset, Dataset

from .datasets import (
    LocalWebDataset,
    ManifestDataset,
    ManifestIterableDataset,
    WeightedSourceMixer,
    load_manifest_rows,
)
from .index import ShardInfo, SourceIndex, index_source
from .mixer import MixtureSampler, normalize_weights, per_source_target_hours
from .providers import ModelScopeShardProvider
from .spec import DataConfig, SourceSpec, load_data_config, parse_data_config

logger = logging.getLogger(__name__)


@dataclass
class DataPipeline:
    dataset: Union[Dataset, Any]
    sampler: Optional[MixtureSampler]
    mode: str
    is_streaming: bool
    sources: Tuple[SourceSpec, ...]
    config: Optional[DataConfig] = None
    shards: Tuple[ShardInfo, ...] = ()
    source_indexes: Tuple[Optional[SourceIndex], ...] = ()
    estimated_samples: Optional[int] = None
    estimated_hours: Optional[float] = None


def _index_with_provider(source: SourceSpec, data_config: DataConfig) -> SourceIndex:
    if source.provider == "modelscope":
        provider = ModelScopeShardProvider(source, data_config.cache)
        try:
            return provider.index_source()
        finally:
            provider.close()
    return index_source(source)


def _all_jsonl(sources: Sequence[SourceSpec]) -> bool:
    return all(source.backend == "jsonl" for source in sources)


def build_eval_dataset(
    path: str, sample_rate: int = 16000, seed: int = 42
) -> ManifestDataset:
    """Build a finite map-style dataset from a local eval JSONL manifest."""
    eval_path = Path(path).resolve()
    source = SourceSpec(
        name=eval_path.stem,
        backend="jsonl",
        provider="local",
        path=str(eval_path),
        text_template="raw",
        weight=1.0,
        check_audio=True,
    )
    return ManifestDataset(
        source=source,
        rows=load_manifest_rows(str(eval_path)),
        sample_rate=sample_rate,
        seed=seed,
    )


def build_data_pipeline(
    config: Union[DataConfig, str, Path, dict],
    force_iterable: bool = False,
) -> DataPipeline:
    """Build a training pipeline (map-style JSONL or streaming)."""
    if isinstance(config, (str, Path)):
        data_config = load_data_config(str(config))
    elif isinstance(config, dict):
        data_config = parse_data_config(config)
    elif isinstance(config, DataConfig):
        data_config = config
    else:
        raise TypeError("config must be a DataConfig, path, or mapping.")

    sources = data_config.sources
    weights = [source.weight for source in sources]
    normalize_weights(weights)

    if _all_jsonl(sources) and not force_iterable:
        if data_config.mixture.total_hours is not None:
            logger.warning(
                "mixture.total_hours is ignored for JSONL-only sources; "
                "use mixture.samples_per_epoch instead."
            )
        datasets = []
        for source in sources:
            rows = load_manifest_rows(str(source.path))
            datasets.append(
                ManifestDataset(
                    source=source,
                    rows=rows,
                    sample_rate=data_config.sample_rate,
                    seed=data_config.seed,
                )
            )
        if len(datasets) == 1:
            dataset: Dataset = datasets[0]
        else:
            dataset = ConcatDataset(datasets)
        sampler = MixtureSampler(
            source_sizes=[len(item) for item in datasets],
            weights=weights,
            num_samples=data_config.mixture.samples_per_epoch,
            seed=data_config.seed,
        )
        return DataPipeline(
            dataset=dataset,
            sampler=sampler,
            mode="offline_jsonl",
            is_streaming=False,
            sources=sources,
            config=data_config,
            source_indexes=tuple(None for _ in sources),
            estimated_samples=len(sampler),
        )

    indexes: List[Optional[SourceIndex]] = []
    for source in sources:
        if source.backend == "webdataset":
            indexes.append(_index_with_provider(source, data_config))
        else:
            indexes.append(None)

    if data_config.mixture.total_hours is not None:
        if any(source.backend == "jsonl" for source in sources):
            logger.warning(
                "mixture.total_hours only applies to WebDataset sources; "
                "JSONL sources are mixed at sample level."
            )
        for source, source_index in zip(sources, indexes):
            if source_index is not None and source_index.total_hours is None:
                raise ValueError(
                    f"Source {source.name!r} has no hours metadata; "
                    "provide dataset_info.json or omit mixture.total_hours."
                )

    targets = per_source_target_hours(sources, data_config.mixture.total_hours)
    datasets = []
    shards: List[ShardInfo] = []
    jsonl_samples: List[int] = []
    for source, source_index, target_hours in zip(sources, indexes, targets):
        if source_index is not None:
            shards.extend(source_index.shards)
            datasets.append(
                LocalWebDataset(
                    source=source,
                    shards=source_index.shards,
                    mixture=data_config.mixture,
                    target_hours=target_hours,
                    sample_rate=data_config.sample_rate,
                    seed=data_config.seed,
                    provider_mode=source.provider,
                    cache_spec=data_config.cache if source.provider == "modelscope" else None,
                    prefetch_shards=(
                        data_config.cache.prefetch_shards
                        if source.provider == "modelscope"
                        else 0
                    ),
                )
            )
        else:
            manifest = ManifestDataset(
                source=source,
                rows=load_manifest_rows(str(source.path)),
                sample_rate=data_config.sample_rate,
                seed=data_config.seed,
            )
            datasets.append(ManifestIterableDataset(manifest, seed=data_config.seed))
            jsonl_samples.append(len(manifest))

    mixer = WeightedSourceMixer(datasets, weights, seed=data_config.seed)
    estimated_hours = None
    if data_config.mixture.total_hours is not None:
        estimated_hours = sum(
            target
            for source, target in zip(sources, targets)
            if source.backend == "webdataset" and target is not None
        )
    else:
        web_hours = [
            index.total_hours for index in indexes if index is not None
        ]
        if web_hours and all(value is not None for value in web_hours):
            estimated_hours = sum(web_hours)
    estimated_samples = None
    if data_config.mixture.total_hours is None:
        web_samples = [
            index.total_samples for index in indexes if index is not None
        ]
        if all(value is not None for value in web_samples):
            estimated_samples = sum(value for value in web_samples if value is not None)
            estimated_samples += sum(jsonl_samples)
    if force_iterable and _all_jsonl(sources):
        mode = "offline_jsonl_stream"
    elif any(source.provider == "modelscope" for source in sources):
        mode = "modelscope_stream"
    else:
        mode = "local_webdataset"
    return DataPipeline(
        dataset=mixer,
        sampler=None,
        mode=mode,
        is_streaming=True,
        sources=sources,
        config=data_config,
        shards=tuple(shards),
        source_indexes=tuple(indexes),
        estimated_samples=estimated_samples,
        estimated_hours=estimated_hours,
    )


def build_stream_dataset(
    config: Union[DataConfig, str, Path, dict],
) -> DataPipeline:
    """Always return an infinite iterable pipeline (needed by augmentation)."""
    return build_data_pipeline(config, force_iterable=True)
