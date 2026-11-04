# coding=utf-8
"""Deterministic source mixture: map-style sampler and WebDataset shard planning."""

from __future__ import annotations

import logging
import math
import random
from typing import List, Optional, Sequence

from .index import ShardInfo
from .labels import stable_seed
from .spec import MixtureSpec, SourceSpec

logger = logging.getLogger(__name__)


def normalize_weights(weights: Sequence[float]) -> List[float]:
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Source weights must sum to a positive value.")
    return [float(weight) / total for weight in weights]


def per_source_target_hours(
    sources: Sequence[SourceSpec], total_hours: Optional[float]
) -> List[Optional[float]]:
    if total_hours is None:
        return [None for _ in sources]
    weights = normalize_weights([source.weight for source in sources])
    targets: List[Optional[float]] = []
    for source, weight in zip(sources, weights):
        target = total_hours * weight
        if source.max_hours is not None:
            target = min(target, source.max_hours)
        targets.append(target)
    return targets


def select_source_shards(
    source: SourceSpec,
    shards: Sequence[ShardInfo],
    target_hours: Optional[float],
    mixture: MixtureSpec,
    pass_index: int = 0,
) -> List[ShardInfo]:
    """Select (and repeat) shards for one source and one streaming pass."""
    rng = random.Random(stable_seed(mixture.seed, source.name, "pass", pass_index))
    shuffled = list(shards)
    rng.shuffle(shuffled)
    if target_hours is None:
        return shuffled
    if target_hours <= 0:
        return []
    missing = [shard.path for shard in shuffled if shard.hours is None]
    if missing:
        raise ValueError(
            f"Source {source.name!r} uses mixture.total_hours but {len(missing)} "
            "shards have no hours metadata."
        )

    if not mixture.repeat_small_sources:
        available = sum(shard.hours for shard in shuffled if shard.hours is not None)
        if target_hours > available:
            logger.warning(
                "Source %s target %.2fh exceeds available %.2fh; using all shards once.",
                source.name,
                target_hours,
                available,
            )
        return shuffled

    selected: List[ShardInfo] = []
    total = 0.0
    repeats = 0
    while total + 1e-9 < target_hours:
        for shard in shuffled:
            selected.append(shard)
            total += float(shard.hours or 0.0)
            if total + 1e-9 >= target_hours:
                break
        repeats += 1
        if repeats >= source.max_repeat:
            break
        rng.shuffle(shuffled)
    if total + 1e-9 < target_hours:
        logger.warning(
            "Source %s reached max_repeat=%.1f at %.2fh < target %.2fh.",
            source.name,
            source.max_repeat,
            total,
            target_hours,
        )
    return selected


class MixtureSampler:
    """Quota-based sampler for map-style JSONL sources.

    Each epoch draws an exact number of samples per source (subject to source
    size and replacement), then shuffles the combined list. ``Trainer`` calls
    :meth:`set_epoch` when present.
    """

    def __init__(
        self,
        source_sizes: Sequence[int],
        weights: Sequence[float],
        num_samples: Optional[int] = None,
        seed: int = 42,
    ):
        if len(source_sizes) != len(weights):
            raise ValueError("source_sizes and weights must have the same length.")
        if any(size < 0 for size in source_sizes):
            raise ValueError("source_sizes must be non-negative.")
        self.source_sizes = [int(size) for size in source_sizes]
        self.weights = normalize_weights(weights)
        self.num_samples = int(num_samples) if num_samples is not None else sum(self.source_sizes)
        self.seed = int(seed)
        self.epoch = 0
        self._offsets: List[int] = []
        running = 0
        for size in self.source_sizes:
            self._offsets.append(running)
            running += size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def _quotas(self, total: int) -> List[int]:
        active = [index for index, size in enumerate(self.source_sizes) if size > 0]
        quotas = [0 for _ in self.source_sizes]
        if not active or total <= 0:
            return quotas
        weight_sum = sum(self.weights[index] for index in active)
        raw = {
            index: total * self.weights[index] / weight_sum
            for index in active
        }
        for index in active:
            quotas[index] = int(math.floor(raw[index]))
        remainder = total - sum(quotas)
        order = sorted(active, key=lambda index: (-(raw[index] - quotas[index]), index))
        for index in order[:remainder]:
            quotas[index] += 1
        return quotas

    def __iter__(self):
        rng = random.Random(stable_seed(self.seed, "mixture", self.epoch))
        quotas = self._quotas(self.num_samples)
        indices: List[int] = []
        for source_index, quota in enumerate(quotas):
            size = self.source_sizes[source_index]
            if quota <= 0 or size <= 0:
                continue
            offset = self._offsets[source_index]
            if quota <= size:
                picked = rng.sample(range(size), quota)
            else:
                picked = [rng.randrange(size) for _ in range(quota)]
            indices.extend(offset + item for item in picked)
        rng.shuffle(indices)
        return iter(indices)
