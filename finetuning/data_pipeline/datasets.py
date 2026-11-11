# coding=utf-8
"""Map-style JSONL datasets and local WebDataset iterables."""

from __future__ import annotations

import io
import json
import logging
import os
import random
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .index import ShardInfo, split_shards_for_worker
from .labels import make_target, stable_seed
from .mixer import normalize_weights, select_source_shards
from .spec import CacheSpec, MixtureSpec, SourceSpec

logger = logging.getLogger(__name__)


def load_manifest_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    manifest = Path(path)
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {manifest}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {manifest}:{line_number}.")
            if not row.get("audio") or "text" not in row:
                raise ValueError(
                    f"Row {manifest}:{line_number} requires 'audio' and 'text' fields."
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"No training rows found in {manifest}.")
    return rows


def _row_duration(row: Dict[str, Any]) -> Optional[float]:
    for key in ("duration", "duration_sec", "dur"):
        value = row.get(key)
        if value is not None:
            return float(value)
    return None


def _row_key(row: Dict[str, Any], source: SourceSpec, index: int) -> str:
    for key in ("key", "id", "utt", "utterance_id"):
        value = row.get(key)
        if value:
            return str(value)
    return f"{source.name}:{index}"


def resolve_audio_path(
    raw_path: str, manifest_dir: Path, audio_root: Optional[str]
) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    if audio_root:
        return (Path(audio_root) / path).resolve()
    return (manifest_dir / path).resolve()


class ManifestDataset(Dataset):
    """One local JSONL source; returns audio paths for the collator to load."""

    def __init__(
        self,
        source: SourceSpec,
        rows: Sequence[Dict[str, Any]],
        sample_rate: int = 16000,
        seed: int = 42,
    ):
        self.source = source
        self.rows = list(rows)
        self.sample_rate = int(sample_rate)
        self.seed = int(seed)
        self.manifest_dir = Path(source.path or ".").resolve().parent

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        key = _row_key(row, self.source, index)
        language = row.get("language") or self.source.language
        target = make_target(row["text"], language, self.source.text_template)
        audio_path = resolve_audio_path(
            row["audio"], self.manifest_dir, self.source.audio_root
        )
        if self.source.check_audio and not audio_path.is_file():
            raise FileNotFoundError(
                f"Audio for {self.source.name}:{key} not found: {audio_path}"
            )
        return {
            "audio": str(audio_path),
            "target": target,
            "prompt": str(row.get("prompt", "") or ""),
            "aug": int(row.get("aug", 1)),
            "noise_aug": int(row.get("noise_aug", 1)),
            "source": self.source.name,
            "key": key,
            "duration": _row_duration(row),
            "language": language,
            "sample_seed": stable_seed(self.seed, self.source.name, key),
        }


class ManifestIterableDataset(IterableDataset):
    """Infinite per-worker shuffled view of a JSONL source."""

    def __init__(self, dataset: ManifestDataset, seed: int = 42):
        if len(dataset) == 0:
            raise ValueError(f"JSONL source {dataset.source.name!r} is empty.")
        self.dataset = dataset
        self.seed = int(seed)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        num_workers = info.num_workers if info is not None else 1
        rng = random.Random(stable_seed(self.seed, "manifest", worker_id))
        while True:
            total = len(self.dataset)
            if num_workers <= total:
                indices = list(range(worker_id, total, num_workers))
            else:
                # Small replay sources are duplicated across workers instead of
                # leaving some workers with no samples.
                indices = list(range(total))
            rng.shuffle(indices)
            for index in indices:
                yield self.dataset[index]


def decode_audio_bytes(data: bytes, target_sr: int) -> np.ndarray:
    import soundfile as sf

    wav, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    if wav.size == 0:
        raise ValueError("Decoded audio is empty.")
    wav = wav.mean(axis=1)
    if sample_rate != target_sr:
        import librosa

        wav = librosa.resample(
            wav, orig_sr=sample_rate, target_sr=target_sr, res_type="soxr_hq"
        )
    return np.asarray(wav, dtype=np.float32)


def _build_webdataset_sample(
    key: str,
    fields: Dict[str, bytes],
    source: SourceSpec,
    target_sr: int,
    seed: int,
    shard: Path,
) -> Optional[Dict[str, Any]]:
    audio_ext = source.fields.audio.lower()
    text_ext = source.fields.text.lower()
    meta_ext = source.fields.meta.lower()
    if audio_ext not in fields:
        raise ValueError(f"Sample {key!r} in {shard} is missing audio ({audio_ext}).")
    meta: Dict[str, Any] = {}
    if meta_ext in fields:
        try:
            meta = json.loads(fields[meta_ext].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid JSON metadata for {key!r} in {shard}: {exc}") from exc
        if not isinstance(meta, dict):
            raise ValueError(f"Metadata for {key!r} in {shard} must be an object.")
    if source.text_key is not None:
        if meta_ext not in fields:
            raise ValueError(
                f"Sample {key!r} in {shard} sets text_key but has no {meta_ext} metadata."
            )
        value = meta.get(source.text_key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Sample {key!r} in {shard} has no text under "
                f"metadata key {source.text_key!r}."
            )
        text = value.strip()
    else:
        if text_ext not in fields:
            raise ValueError(
                f"Sample {key!r} in {shard} is missing text ({text_ext})."
            )
        text = fields[text_ext].decode("utf-8", errors="replace").strip()
    language = meta.get("language") or source.language
    target = make_target(text, language, source.text_template)
    audio = decode_audio_bytes(fields[audio_ext], target_sr)
    duration = meta.get("duration", meta.get("duration_sec"))
    if duration is None:
        duration = len(audio) / max(1, int(target_sr))
    sample_key = str(meta.get("utt") or meta.get("key") or key)
    return {
        "audio": audio,
        "target": target,
        "prompt": str(meta.get("prompt", "") or ""),
        "aug": int(meta.get("aug", 1)),
        "noise_aug": int(meta.get("noise_aug", 1)),
        "source": source.name,
        "key": sample_key,
        "duration": float(duration) if duration is not None else None,
        "language": language,
        "sample_seed": stable_seed(seed, source.name, sample_key),
    }


def iter_webdataset_shard(
    path: Path,
    source: SourceSpec,
    target_sr: int = 16000,
    seed: int = 42,
    skip_errors: bool = True,
) -> Iterator[Dict[str, Any]]:
    """Iterate one tar shard following the WebDataset per-key grouping convention."""
    audio_ext = source.fields.audio.lower()
    text_ext = source.fields.text.lower()
    meta_ext = source.fields.meta.lower()
    known_exts = {audio_ext, meta_ext}
    if source.text_key is None:
        known_exts.add(text_ext)
    current_key: Optional[str] = None
    current_fields: Dict[str, bytes] = {}

    def flush() -> Iterator[Dict[str, Any]]:
        if current_key is None:
            return
        try:
            sample = _build_webdataset_sample(
                current_key, current_fields, source, target_sr, seed, path
            )
        except Exception as exc:
            if not skip_errors:
                raise
            logger.warning("Skipping sample %r in %s: %s", current_key, path, exc)
            return
        if sample is not None:
            yield sample

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
            if key != current_key:
                yield from flush()
                current_key = key
                current_fields = {}
            if ext not in known_exts:
                continue
            extracted = archive.extractfile(member)
            if extracted is not None:
                current_fields[ext] = extracted.read()
    yield from flush()


class LocalWebDataset(IterableDataset):
    """Infinite WebDataset stream with per-pass shard selection/shuffling.

    DataLoader workers must not exceed the number of selected shards; shards are
    assigned whole to workers so each sample is decoded once per pass. Providers
    are created inside each worker so ModelScope clients/executors are not forked.
    """

    def __init__(
        self,
        source: SourceSpec,
        shards: Sequence[ShardInfo],
        mixture: MixtureSpec,
        target_hours: Optional[float] = None,
        sample_rate: int = 16000,
        seed: Optional[int] = None,
        provider_mode: str = "local",
        cache_spec: Optional[CacheSpec] = None,
        prefetch_shards: int = 0,
    ):
        if not shards:
            raise ValueError(f"WebDataset source {source.name!r} has no shards.")
        if target_hours is not None and target_hours <= 0:
            raise ValueError(f"WebDataset source {source.name!r} target_hours must be > 0.")
        if provider_mode not in {"local", "modelscope"}:
            raise ValueError(f"Unsupported provider_mode {provider_mode!r}.")
        if provider_mode == "modelscope" and cache_spec is None:
            raise ValueError("provider_mode=modelscope requires cache_spec.")
        self.source = source
        self.shards = tuple(shards)
        self.mixture = mixture
        self.target_hours = target_hours
        self.sample_rate = int(sample_rate)
        self.seed = int(mixture.seed if seed is None else seed)
        self.provider_mode = provider_mode
        self.cache_spec = cache_spec
        self.prefetch_shards = max(0, int(prefetch_shards))

    def _make_provider(self):
        if self.provider_mode == "modelscope":
            from .providers import ModelScopeShardProvider

            return ModelScopeShardProvider(self.source, self.cache_spec)
        from .providers import LocalProvider

        return LocalProvider()

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        num_workers = info.num_workers if info is not None else 1
        rng = random.Random(stable_seed(self.seed, self.source.name, "worker", worker_id))
        pass_index = 0
        provider = self._make_provider()
        try:
            while True:
                selected = select_source_shards(
                    self.source, self.shards, self.target_hours, self.mixture, pass_index
                )
                worker_shards = split_shards_for_worker(selected, worker_id, num_workers)
                if selected and not worker_shards:
                    raise RuntimeError(
                        f"Source {self.source.name!r} selected {len(selected)} shard(s) "
                        f"for {num_workers} workers. Reduce dataloader num_workers or "
                        "increase mixture.total_hours / shard count."
                    )
                rng.shuffle(worker_shards)
                for position, shard in enumerate(worker_shards):
                    for ahead in range(1, self.prefetch_shards + 1):
                        next_index = position + ahead
                        if next_index < len(worker_shards):
                            provider.prefetch(worker_shards[next_index])
                    provider.ensure_local(shard)
                    buffer: List[Dict[str, Any]] = []
                    with provider.lease(shard) as local_path:
                        for sample in iter_webdataset_shard(
                            local_path,
                            self.source,
                            target_sr=self.sample_rate,
                            seed=self.seed,
                            skip_errors=True,
                        ):
                            buffer.append(sample)
                            if len(buffer) >= self.mixture.sample_shuffle_buffer:
                                rng.shuffle(buffer)
                                yield from buffer
                                buffer.clear()
                    rng.shuffle(buffer)
                    yield from buffer
                pass_index += 1
        finally:
            provider.close()


class WeightedSourceMixer(IterableDataset):
    """Weighted mix of infinite per-source iterables, by samples or hours."""

    def __init__(
        self,
        datasets: Sequence[IterableDataset],
        weights: Sequence[float],
        seed: int = 42,
        weight_mode: str = "samples",
    ):
        if not datasets:
            raise ValueError("WeightedSourceMixer requires at least one dataset.")
        if len(datasets) != len(weights):
            raise ValueError("datasets and weights must have the same length.")
        if weight_mode not in {"samples", "hours"}:
            raise ValueError("weight_mode must be 'samples' or 'hours'.")
        self.datasets = list(datasets)
        self.weights = normalize_weights(weights)
        self.seed = int(seed)
        self.weight_mode = weight_mode

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        rng = random.Random(stable_seed(self.seed, "weighted_mixer", worker_id))
        iterators = [iter(dataset) for dataset in self.datasets]
        if self.weight_mode == "hours":
            yield from self._iter_hours(iterators, rng)
            return
        while True:
            source_index = rng.choices(
                range(len(iterators)), weights=self.weights, k=1
            )[0]
            try:
                yield next(iterators[source_index])
            except StopIteration:
                iterators[source_index] = iter(self.datasets[source_index])
                yield next(iterators[source_index])

    def _iter_hours(self, iterators, rng) -> Iterator[Dict[str, Any]]:
        yielded_hours = [0.0] * len(iterators)
        total_hours = 0.0
        active = [index for index, weight in enumerate(self.weights) if weight > 0]
        if not active:
            raise ValueError("All source weights are zero.")
        while True:
            deficits = [
                self.weights[index] * total_hours - yielded_hours[index]
                for index in active
            ]
            best = max(deficits)
            candidates = [
                index
                for index, deficit in zip(active, deficits)
                if deficit >= best - 1e-9
            ]
            source_index = rng.choice(candidates)
            try:
                sample = next(iterators[source_index])
            except StopIteration:
                iterators[source_index] = iter(self.datasets[source_index])
                sample = next(iterators[source_index])
            duration = float(sample.get("duration") or 0.0)
            if duration <= 0:
                duration = 1.0
            yielded_hours[source_index] += duration
            total_hours += duration
            yield sample
