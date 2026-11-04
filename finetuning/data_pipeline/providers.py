# coding=utf-8
"""Shard providers for local files and on-demand ModelScope downloads."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from .cache import ShardCache
from .index import ShardInfo, SourceIndex, index_source
from .spec import CacheSpec, SourceSpec


class LocalProvider:
    """Local shards are already on disk."""

    def index_source(self, source: SourceSpec) -> SourceIndex:
        return index_source(source)

    def ensure_local(self, shard: ShardInfo) -> Path:
        return shard.path

    def prefetch(self, shard: ShardInfo) -> None:
        return None

    @contextmanager
    def lease(self, shard: ShardInfo) -> Iterator[Path]:
        yield shard.path

    def close(self) -> None:
        return None


class ModelScopeShardProvider:
    """On-demand shard downloads backed by :class:`ShardCache`."""

    def __init__(
        self,
        source: SourceSpec,
        cache_spec: CacheSpec,
        token: Optional[str] = None,
    ):
        self.source = source
        self.cache = ShardCache(source, cache_spec, token=token)

    def index_source(self) -> SourceIndex:
        return self.cache.index_source()

    def ensure_local(self, shard: ShardInfo) -> Path:
        return self.cache.ensure_local(shard)

    def prefetch(self, shard: ShardInfo) -> None:
        self.cache.prefetch(shard)

    def lease(self, shard: ShardInfo) -> Iterator[Path]:
        return self.cache.lease(shard)

    def close(self) -> None:
        self.cache.close()
