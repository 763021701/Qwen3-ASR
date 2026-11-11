# coding=utf-8
"""Local WebDataset shard discovery and a small on-disk-free shard index."""

from __future__ import annotations

import fnmatch
import glob
import json
import logging
import os
import tarfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .spec import SourceSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShardInfo:
    path: Path
    size_bytes: int
    num_samples: Optional[int] = None
    hours: Optional[float] = None
    remote_path: Optional[str] = None


@dataclass(frozen=True)
class SourceIndex:
    source: SourceSpec
    shards: Tuple[ShardInfo, ...]
    total_bytes: int
    total_samples: Optional[int]
    total_hours: Optional[float]


def _dataset_info_path(source: SourceSpec) -> Optional[Path]:
    if not source.path:
        return None
    raw = Path(source.path)
    if raw.is_dir():
        candidate = raw / "dataset_info.json"
    elif raw.is_file():
        candidate = raw.parent / "dataset_info.json"
    else:
        candidate = Path(glob.escape(str(raw))).parent / "dataset_info.json"
    return candidate if candidate.is_file() else None


def _load_split_info(source: SourceSpec) -> Optional[Mapping[str, Any]]:
    info_path = _dataset_info_path(source)
    if info_path is None:
        return None
    try:
        with info_path.open("r", encoding="utf-8") as handle:
            info = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable %s: %s", info_path, exc)
        return None
    split = source.split or "train"
    splits = info.get("splits")
    if isinstance(splits, dict) and split in splits:
        return splits[split]
    if isinstance(info.get("shards"), list):
        return info
    return None


def matches_shard_exclude(path: Any, source: SourceSpec) -> bool:
    if not source.shard_exclude:
        return False
    text = str(path)
    name = Path(text).name
    return any(
        fnmatch.fnmatch(text, pattern) or fnmatch.fnmatch(name, pattern)
        for pattern in source.shard_exclude
    )


def apply_estimated_hours(
    shards: Iterable[ShardInfo], estimated_hours: Optional[float]
) -> Tuple[ShardInfo, ...]:
    """Assign missing shard hours proportionally when a source has no metadata."""
    shards = tuple(shards)
    if estimated_hours is None:
        return shards
    missing = [shard for shard in shards if shard.hours is None]
    if not missing:
        return shards
    known = sum(float(shard.hours) for shard in shards if shard.hours is not None)
    remaining = max(0.0, float(estimated_hours) - known)
    weights = [
        float(shard.num_samples)
        if shard.num_samples is not None and shard.num_samples > 0
        else float(shard.size_bytes)
        for shard in missing
    ]
    total = sum(weights)
    if total <= 0:
        weights = [1.0] * len(missing)
        total = float(len(missing))
    result: List[ShardInfo] = []
    for shard in shards:
        if shard.hours is not None:
            result.append(shard)
            continue
        weight = (
            float(shard.num_samples)
            if shard.num_samples is not None and shard.num_samples > 0
            else float(shard.size_bytes)
        )
        result.append(replace(shard, hours=remaining * weight / total))
    return tuple(result)


def split_shard_metadata(split_info: Optional[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not split_info:
        return {}
    entries = split_info.get("shards") or []
    result: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        if isinstance(entry, str):
            result[Path(entry).name] = {}
        elif isinstance(entry, dict):
            name = entry.get("file") or entry.get("path") or entry.get("name")
            if name:
                result[Path(str(name)).name] = entry
    return result


def _list_candidate_files(source: SourceSpec) -> List[Path]:
    if not source.path:
        return []
    raw = Path(source.path)
    if raw.is_file():
        return [raw]
    if raw.is_dir():
        return sorted(path for path in raw.rglob(source.shard_pattern) if path.is_file())
    return sorted(
        Path(item)
        for item in glob.glob(str(raw), recursive=True)
        if Path(item).is_file()
    )


def scan_tar_shard(path: Path, fields: Any) -> Tuple[int, Optional[float]]:
    """Count samples and optional durations by scanning tar headers/meta members."""
    keys = set()
    total_duration = 0.0
    meta_count = 0
    meta_with_duration = 0
    meta_ext = str(fields.meta).lower()
    with tarfile.open(path, "r:*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = os.path.basename(member.name)
            if "." not in name:
                continue
            key, ext = name.rsplit(".", 1)
            ext = ext.lower()
            if not key:
                continue
            keys.add(key)
            if ext != meta_ext:
                continue
            meta_count += 1
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            try:
                meta = json.loads(extracted.read().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            duration = meta.get("duration", meta.get("duration_sec"))
            if duration is None:
                continue
            meta_with_duration += 1
            total_duration += float(duration)
    if not keys:
        raise ValueError(f"No WebDataset samples found in {path}")
    hours = None
    if meta_count > 0 and meta_with_duration == meta_count:
        hours = total_duration / 3600.0
    return len(keys), hours


def _build_shard_info(
    path: Path,
    metadata: Mapping[str, Any],
    fields: Any,
    scan_missing: bool,
) -> ShardInfo:
    num_samples = metadata.get("samples", metadata.get("num_samples"))
    hours = metadata.get("hours")
    if num_samples is None or hours is None:
        if not scan_missing:
            raise ValueError(
                f"Missing shard metadata for {path}; provide dataset_info.json "
                "or enable scanning."
            )
        scanned_samples, scanned_hours = scan_tar_shard(path, fields)
        if num_samples is None:
            num_samples = scanned_samples
        if hours is None:
            hours = scanned_hours
    return ShardInfo(
        path=path,
        size_bytes=path.stat().st_size,
        num_samples=int(num_samples) if num_samples is not None else None,
        hours=float(hours) if hours is not None else None,
    )


def index_source(source: SourceSpec, scan_missing: bool = True) -> SourceIndex:
    """Discover local shards and attach sample/hour metadata when available."""
    if source.backend != "webdataset":
        raise ValueError(f"Source {source.name!r} is not a WebDataset source.")
    candidates = [
        path for path in _list_candidate_files(source)
        if not matches_shard_exclude(path, source)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No shards matched {source.path!r} with pattern {source.shard_pattern!r} "
            "after shard_exclude."
        )

    split_info = _load_split_info(source)
    metadata_by_name = split_shard_metadata(split_info)
    if metadata_by_name:
        allowed = set(metadata_by_name)
        selected = [path for path in candidates if path.name in allowed]
        if not selected:
            raise ValueError(
                f"No shards for split {source.split!r} in {source.path!r}; "
                f"dataset_info lists {sorted(allowed)[:3]}..."
            )
        candidates = selected

    shards = apply_estimated_hours(
        (
            _build_shard_info(
                path=path,
                metadata=metadata_by_name.get(path.name, {}),
                fields=source.fields,
                scan_missing=scan_missing,
            )
            for path in candidates
        ),
        source.estimated_hours,
    )
    total_bytes = sum(shard.size_bytes for shard in shards)
    sample_values = [shard.num_samples for shard in shards]
    total_samples = None if any(value is None for value in sample_values) else sum(sample_values)
    hour_values = [shard.hours for shard in shards]
    total_hours = None if any(value is None for value in hour_values) else sum(hour_values)
    return SourceIndex(
        source=source,
        shards=shards,
        total_bytes=total_bytes,
        total_samples=total_samples,
        total_hours=total_hours,
    )


def split_shards_for_worker(shards: Iterable[ShardInfo], worker_id: int, num_workers: int) -> List[ShardInfo]:
    """Deterministically assign shards to one DataLoader worker."""
    if num_workers <= 0:
        return list(shards)
    return list(shards)[worker_id::num_workers]
