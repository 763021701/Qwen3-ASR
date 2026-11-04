# coding=utf-8
"""Data config and source specs for the local data pipeline (P1/P2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml


@dataclass(frozen=True)
class Fields:
    """WebDataset field names (file extensions without the dot)."""

    audio: str = "flac"
    text: str = "txt"
    meta: str = "json"


@dataclass(frozen=True)
class SourceSpec:
    """One local or ModelScope JSONL / WebDataset source."""

    name: str
    backend: str
    provider: str = "local"
    path: Optional[str] = None
    repo_id: Optional[str] = None
    revision: str = "master"
    path_in_repo: Optional[str] = None
    split: Optional[str] = None
    shard_pattern: str = "*.tar"
    audio_root: Optional[str] = None
    language: Optional[str] = None
    text_template: str = "language {language}<asr_text>{text}"
    weight: float = 1.0
    max_hours: Optional[float] = None
    max_repeat: float = 4.0
    check_audio: bool = True
    fields: Fields = field(default_factory=Fields)


@dataclass(frozen=True)
class CacheSpec:
    """Bounded local cache for on-demand ModelScope shard downloads."""

    dir: Optional[str] = None
    max_gb: float = 100.0
    min_free_gb: float = 20.0
    prefetch_shards: int = 1
    max_concurrent_downloads: int = 1
    retries: int = 3
    timeout_sec: float = 120.0
    token_env: str = "MODELSCOPE_API_TOKEN"
    refresh_index: bool = False
    index_ttl_sec: int = 0


@dataclass(frozen=True)
class MixtureSpec:
    """How sources are mixed and how long one streaming pass should be."""

    total_hours: Optional[float] = None
    seed: int = 42
    samples_per_epoch: Optional[int] = None
    sample_shuffle_buffer: int = 2000
    repeat_small_sources: bool = True


@dataclass(frozen=True)
class DataConfig:
    """Top-level data config consumed by :func:`build_data_pipeline`."""

    sources: Tuple[SourceSpec, ...]
    mode: str = "auto"
    seed: int = 42
    sample_rate: int = 16000
    mixture: MixtureSpec = field(default_factory=MixtureSpec)
    cache: CacheSpec = field(default_factory=CacheSpec)


def _as_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _resolve_path(value: Optional[str], base_dir: Optional[Path]) -> Optional[str]:
    if not value or base_dir is None:
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _parse_fields(raw: Any) -> Fields:
    if raw is None:
        return Fields()
    if not isinstance(raw, dict):
        raise ValueError("source.fields must be a mapping with audio/text/meta keys.")
    unknown = set(raw) - {"audio", "text", "meta"}
    if unknown:
        raise ValueError(f"Unknown source.fields keys: {sorted(unknown)}")
    return Fields(
        audio=str(raw.get("audio", Fields.audio)).lstrip("."),
        text=str(raw.get("text", Fields.text)).lstrip("."),
        meta=str(raw.get("meta", Fields.meta)).lstrip("."),
    )


def _parse_source(raw: Dict[str, Any], base_dir: Optional[Path]) -> SourceSpec:
    if not isinstance(raw, dict):
        raise ValueError("Each data source must be a mapping.")
    name = str(raw.get("name") or "").strip()
    if not name:
        raise ValueError("Each data source requires a non-empty name.")
    backend = str(raw.get("backend") or "jsonl").strip().lower()
    if backend not in {"jsonl", "webdataset"}:
        raise ValueError(f"Unsupported backend {backend!r}; use 'jsonl' or 'webdataset'.")
    provider = str(raw.get("provider") or "local").strip().lower()
    if provider not in {"local", "modelscope"}:
        raise ValueError(
            f"Source {name!r}: provider must be 'local' or 'modelscope'."
        )
    repo_id = str(raw.get("repo_id") or "").strip() or None
    path_in_repo = str(raw.get("path_in_repo") or "").strip() or None
    path = _resolve_path(str(raw.get("path") or "") or None, base_dir)
    if provider == "local":
        if not path:
            raise ValueError(f"Source {name!r} requires a local path.")
    else:
        if backend != "webdataset":
            raise ValueError(
                f"Source {name!r}: provider=modelscope requires backend=webdataset."
            )
        if not repo_id or "/" not in repo_id:
            raise ValueError(
                f"Source {name!r}: ModelScope sources require repo_id='namespace/name'."
            )
        if not path_in_repo:
            raise ValueError(
                f"Source {name!r}: ModelScope sources require path_in_repo."
            )
        path = None

    text_template = str(
        raw.get("text_template")
        or ("raw" if backend == "jsonl" else "language {language}<asr_text>{text}")
    )
    language = raw.get("language")
    if text_template != "raw":
        if "{text}" not in text_template:
            raise ValueError(
                f"Source {name!r}: text_template must be 'raw' or contain '{{text}}'."
            )
        if "{language}" in text_template and language is None:
            language = "None"

    weight = float(raw.get("weight", 1.0))
    if weight <= 0:
        raise ValueError(f"Source {name!r}: weight must be > 0.")
    max_hours = raw.get("max_hours")
    if max_hours is not None:
        max_hours = float(max_hours)
        if max_hours <= 0:
            raise ValueError(f"Source {name!r}: max_hours must be > 0.")
    max_repeat = float(raw.get("max_repeat", 4.0))
    if max_repeat <= 0:
        raise ValueError(f"Source {name!r}: max_repeat must be > 0.")

    split = raw.get("split")
    if backend == "webdataset" and split is None:
        split = "train"
    return SourceSpec(
        name=name,
        backend=backend,
        provider=provider,
        path=path,
        repo_id=repo_id,
        revision=str(raw.get("revision") or "master"),
        path_in_repo=path_in_repo,
        split=None if split is None else str(split),
        shard_pattern=str(raw.get("shard_pattern") or "*.tar"),
        audio_root=_resolve_path(str(raw.get("audio_root") or "") or None, base_dir),
        language=None if language is None else str(language),
        text_template=text_template,
        weight=weight,
        max_hours=max_hours,
        max_repeat=max_repeat,
        check_audio=_as_bool(raw.get("check_audio"), default=True) if provider == "local" else False,
        fields=_parse_fields(raw.get("fields")),
    )


def _parse_cache(raw: Any, base_dir: Optional[Path]) -> CacheSpec:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("data.cache must be a mapping.")
    max_gb = float(raw.get("max_gb", 100.0))
    if max_gb <= 0:
        raise ValueError("data.cache.max_gb must be > 0.")
    min_free_gb = float(raw.get("min_free_gb", 20.0))
    if min_free_gb < 0:
        raise ValueError("data.cache.min_free_gb must be >= 0.")
    prefetch_shards = int(raw.get("prefetch_shards", 1))
    if prefetch_shards < 0:
        raise ValueError("data.cache.prefetch_shards must be >= 0.")
    max_concurrent = int(raw.get("max_concurrent_downloads", 1))
    if max_concurrent < 1:
        raise ValueError("data.cache.max_concurrent_downloads must be >= 1.")
    retries = int(raw.get("retries", 3))
    if retries < 0:
        raise ValueError("data.cache.retries must be >= 0.")
    timeout_sec = float(raw.get("timeout_sec", 120.0))
    if timeout_sec <= 0:
        raise ValueError("data.cache.timeout_sec must be > 0.")
    ttl = int(raw.get("index_ttl_sec", 0))
    if ttl < 0:
        raise ValueError("data.cache.index_ttl_sec must be >= 0.")
    return CacheSpec(
        dir=_resolve_path(str(raw.get("dir") or "") or None, base_dir),
        max_gb=max_gb,
        min_free_gb=min_free_gb,
        prefetch_shards=prefetch_shards,
        max_concurrent_downloads=max_concurrent,
        retries=retries,
        timeout_sec=timeout_sec,
        token_env=str(raw.get("token_env") or "MODELSCOPE_API_TOKEN"),
        refresh_index=_as_bool(raw.get("refresh_index"), default=False),
        index_ttl_sec=ttl,
    )


def parse_data_config(raw: Dict[str, Any], base_dir: Optional[Path] = None) -> DataConfig:
    """Parse a data config mapping. Relative paths resolve against ``base_dir``."""
    if not isinstance(raw, dict):
        raise ValueError("Data config must be a mapping.")
    mode = str(raw.get("mode") or "auto").strip().lower()
    if mode not in {"auto", "offline", "streaming"}:
        raise ValueError("data.mode must be one of: auto, offline, streaming.")

    raw_sources = raw.get("sources") or []
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("data.sources must be a non-empty list.")
    sources = tuple(_parse_source(item, base_dir) for item in raw_sources)
    names = [source.name for source in sources]
    if len(set(names)) != len(names):
        raise ValueError("data.sources names must be unique.")
    cache = _parse_cache(raw.get("cache"), base_dir)
    if any(source.provider == "modelscope" for source in sources) and not cache.dir:
        raise ValueError(
            "ModelScope sources require data.cache.dir for downloaded shards."
        )

    raw_mixture = raw.get("mixture") or {}
    total_hours = raw_mixture.get("total_hours")
    if total_hours is not None:
        total_hours = float(total_hours)
        if total_hours <= 0:
            raise ValueError("data.mixture.total_hours must be > 0.")
    samples_per_epoch = raw_mixture.get("samples_per_epoch")
    if samples_per_epoch is not None:
        samples_per_epoch = int(samples_per_epoch)
        if samples_per_epoch <= 0:
            raise ValueError("data.mixture.samples_per_epoch must be > 0.")
    shuffle_buffer = int(raw_mixture.get("sample_shuffle_buffer", 2000))
    if shuffle_buffer <= 0:
        raise ValueError("data.mixture.sample_shuffle_buffer must be > 0.")

    mixture = MixtureSpec(
        total_hours=total_hours,
        seed=int(raw_mixture.get("seed", raw.get("seed", 42))),
        samples_per_epoch=samples_per_epoch,
        sample_shuffle_buffer=shuffle_buffer,
        repeat_small_sources=_as_bool(
            raw_mixture.get("repeat_small_sources"), default=True
        ),
    )
    return DataConfig(
        sources=sources,
        mode=mode,
        seed=int(raw.get("seed", mixture.seed)),
        sample_rate=int(raw.get("sample_rate", 16000)),
        mixture=mixture,
        cache=cache,
    )


def load_data_config(path: str) -> DataConfig:
    """Load a YAML data config. Relative paths resolve against the config file."""
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if "data" in raw:
        raw = raw["data"]
    return parse_data_config(raw, base_dir=config_path.parent)
