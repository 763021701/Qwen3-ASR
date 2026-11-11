# coding=utf-8
"""On-demand ModelScope WebDataset shard cache with LRU, locking, and prefetch."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import os
import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import requests

from .index import (
    ShardInfo,
    SourceIndex,
    apply_estimated_hours,
    matches_shard_exclude,
    split_shard_metadata,
)
from .spec import CacheSpec, SourceSpec

logger = logging.getLogger(__name__)

try:
    import fcntl
except ImportError:  # pragma: no cover - non-Unix fallback
    fcntl = None


def _safe_name(value: str) -> str:
    return value.replace(os.sep, "__").replace("/", "__").replace("\\", "__")


def _sha1(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


@contextmanager
def _flock(path: Path, exclusive: bool, blocking: bool) -> Iterator[bool]:
    if fcntl is None:
        yield True
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        flags = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        if not blocking:
            flags |= fcntl.LOCK_NB
        try:
            fcntl.flock(handle.fileno(), flags)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class ShardCache:
    """Download and cache ModelScope shard files on demand."""

    def __init__(self, source: SourceSpec, spec: CacheSpec, token: Optional[str] = None):
        if source.provider != "modelscope":
            raise ValueError("ShardCache requires a modelscope source.")
        if not source.repo_id or "/" not in source.repo_id:
            raise ValueError("ModelScope source requires repo_id='namespace/name'.")
        if not spec.dir:
            raise ValueError("ModelScope source requires data.cache.dir.")
        self.source = source
        self.spec = spec
        self.max_gb = float(
            spec.per_source_max_gb.get(source.name, spec.max_gb)
        )
        self.repo_id = source.repo_id
        self.namespace, self.repo_name = self.repo_id.split("/", 1)
        self.token = token or os.environ.get(spec.token_env) or None
        self.cache_root = Path(spec.dir)
        self.root = self.cache_root / _safe_name(self.repo_id) / _safe_name(source.revision)
        self.index_dir = (
            self.cache_root / "index" / _safe_name(self.repo_id) / _safe_name(source.revision)
        )
        self.locks_dir = self.cache_root / "locks" / _safe_name(self.repo_id)
        for directory in (self.root, self.index_dir, self.locks_dir):
            directory.mkdir(parents=True, exist_ok=True)
        from modelscope.hub.api import HubApi

        self.api = HubApi(token=self.token)
        self.endpoint: Optional[str] = None
        self._cookies: Any = None
        self._files: Optional[List[Dict[str, Any]]] = None
        self._remote_info_entry: Optional[Dict[str, Any]] = None
        self._dataset_info: Optional[Dict[str, Any]] = None
        self._dataset_info_loaded = False
        self._index_result: Optional[SourceIndex] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._futures: Dict[str, Future] = {}
        self._future_lock = threading.Lock()

    # ---------------------------------------------------------------- paths
    def local_path(self, remote_path: str) -> Path:
        root = self.root.resolve()
        target = (self.root / remote_path).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"Remote path escapes cache root: {remote_path!r}")
        return target

    @property
    def _index_path(self) -> Path:
        return self.index_dir / (_safe_name(self.source.path_in_repo or "root") + ".json")

    # ---------------------------------------------------------------- auth / API
    def _endpoint(self) -> Optional[str]:
        if self.endpoint is None:
            try:
                self.endpoint = self.api.get_endpoint_for_read(
                    self.repo_id, repo_type="dataset", token=self.token
                )
            except Exception as exc:
                logger.warning(
                    "Could not resolve ModelScope endpoint for %s: %s", self.repo_id, exc
                )
                self.endpoint = getattr(self.api, "endpoint", None)
        return self.endpoint

    def _cookie_jar(self) -> Any:
        if self._cookies is None:
            try:
                self._cookies = self.api.get_cookies(access_token=self.token)
            except Exception as exc:
                logger.warning("Could not load ModelScope credentials: %s", exc)
                self._cookies = None
        return self._cookies

    def _remote_url(self, remote_path: str) -> str:
        return self.api.get_dataset_file_url(
            file_name=remote_path,
            dataset_name=self.repo_name,
            namespace=self.namespace,
            revision=self.source.revision,
            endpoint=self._endpoint(),
        )

    # ---------------------------------------------------------------- index
    def _load_cached_index(self) -> Optional[List[Dict[str, Any]]]:
        if self.spec.refresh_index:
            return None
        path = self._index_path
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring unreadable shard index %s: %s", path, exc)
            return None
        fetched_at = float(payload.get("fetched_at", 0.0))
        if self.spec.index_ttl_sec and time.time() - fetched_at > self.spec.index_ttl_sec:
            return None
        files = payload.get("files")
        if not isinstance(files, list) or not files:
            return None
        self._dataset_info = payload.get("dataset_info")
        self._dataset_info_loaded = True
        logger.info("Loaded %d remote shard entries from %s", len(files), path)
        return files

    def _save_cached_index(self, files: List[Dict[str, Any]]) -> None:
        payload = {
            "repo_id": self.repo_id,
            "revision": self.source.revision,
            "path_in_repo": self.source.path_in_repo,
            "fetched_at": time.time(),
            "files": files,
            "dataset_info": self._dataset_info,
        }
        tmp = self._index_path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        os.replace(tmp, self._index_path)

    def _fetch_remote_files(self) -> List[Dict[str, Any]]:
        endpoint = self._endpoint()
        page = 1
        page_size = 100
        files: List[Dict[str, Any]] = []
        prefix = (self.source.path_in_repo or "").strip("/")
        info_candidates = {
            (prefix + "/dataset_info.json") if prefix else "dataset_info.json",
            "dataset_info.json",
        }
        self._remote_info_entry = None
        while True:
            try:
                entries = self.api.get_dataset_files(
                    self.repo_id,
                    revision=self.source.revision,
                    root_path="/",
                    recursive=True,
                    page_number=page,
                    page_size=page_size,
                    endpoint=endpoint,
                    token=self.token,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to list ModelScope dataset {self.repo_id}: {exc}"
                ) from exc
            if not entries:
                break
            if not isinstance(entries, list):
                raise RuntimeError(
                    f"Unexpected ModelScope file list for {self.repo_id}: {type(entries).__name__}"
                )
            for entry in entries:
                if entry.get("Type") == "tree":
                    continue
                path = str(entry.get("Path") or "").strip().lstrip("/")
                if not path:
                    continue
                if path in info_candidates:
                    self._remote_info_entry = {
                        "path": path,
                        "size": int(entry.get("Size") or 0),
                    }
                if prefix and not (path == prefix or path.startswith(prefix + "/")):
                    continue
                files.append({"path": path, "size": int(entry.get("Size") or 0)})
            if len(entries) < page_size:
                break
            page += 1
        if not files:
            raise FileNotFoundError(
                f"No ModelScope files under {self.source.path_in_repo!r} in {self.repo_id}."
            )
        files.sort(key=lambda item: item["path"])
        return files

    def list_remote_files(self, force: bool = False) -> List[Dict[str, Any]]:
        if not force and self._files is not None:
            return self._files
        if not force:
            cached = self._load_cached_index()
            if cached is not None:
                self._files = cached
                return cached
        self._dataset_info = None
        self._dataset_info_loaded = False
        files = self._fetch_remote_files()
        self._files = files
        self._load_dataset_info()
        self._save_cached_index(files)
        return files

    def _load_dataset_info(self) -> None:
        if self._dataset_info_loaded:
            return
        self._dataset_info_loaded = True
        entry = self._remote_info_entry
        if not entry:
            return
        remote = entry["path"]
        target = self.local_path(remote)
        try:
            self._download_file(remote, target, int(entry.get("size") or 0))
            with target.open("r", encoding="utf-8") as handle:
                info = json.load(handle)
        except Exception as exc:
            logger.warning("Could not load %s from %s: %s", remote, self.repo_id, exc)
            return
        if isinstance(info, dict):
            self._dataset_info = info

    def index_source(self) -> SourceIndex:
        if self._index_result is not None and not self.spec.refresh_index:
            return self._index_result
        self._index_result = None
        files = self.list_remote_files(force=self.spec.refresh_index)
        split = self.source.split or "train"
        split_info = None
        if isinstance(self._dataset_info, dict):
            splits = self._dataset_info.get("splits")
            if isinstance(splits, dict):
                split_info = splits.get(split)
            if split_info is None and isinstance(self._dataset_info.get("shards"), list):
                split_info = self._dataset_info
        metadata_by_name = split_shard_metadata(split_info)
        if not metadata_by_name:
            logger.warning(
                "No dataset_info split metadata for %s split %r; "
                "using every shard matching %r.",
                self.repo_id,
                split,
                self.source.shard_pattern,
            )
        shards: List[ShardInfo] = []
        for entry in files:
            remote = entry["path"]
            name = Path(remote).name
            if matches_shard_exclude(remote, self.source):
                continue
            if metadata_by_name and name not in metadata_by_name:
                continue
            if not (
                fnmatch.fnmatch(remote, self.source.shard_pattern)
                or fnmatch.fnmatch(name, self.source.shard_pattern)
            ):
                continue
            metadata = metadata_by_name.get(name, {})
            samples = metadata.get("samples", metadata.get("num_samples"))
            hours = metadata.get("hours")
            shards.append(
                ShardInfo(
                    path=self.local_path(remote),
                    size_bytes=int(entry.get("size") or 0),
                    num_samples=int(samples) if samples is not None else None,
                    hours=float(hours) if hours is not None else None,
                    remote_path=remote,
                )
            )
        if not shards:
            raise FileNotFoundError(
                f"No ModelScope shards matched {self.source.path_in_repo!r} / "
                f"{self.source.shard_pattern!r} for split {split!r}."
            )
        shards = apply_estimated_hours(shards, self.source.estimated_hours)
        total_bytes = sum(shard.size_bytes for shard in shards)
        sample_values = [shard.num_samples for shard in shards]
        total_samples = (
            None if any(value is None for value in sample_values) else sum(sample_values)
        )
        hour_values = [shard.hours for shard in shards]
        total_hours = (
            None if any(value is None for value in hour_values) else sum(hour_values)
        )
        self._index_result = SourceIndex(
            source=self.source,
            shards=tuple(shards),
            total_bytes=total_bytes,
            total_samples=total_samples,
            total_hours=total_hours,
        )
        return self._index_result

    # ---------------------------------------------------------------- cache
    def _cache_files(self) -> List[Path]:
        return [
            path
            for path in self.root.rglob("*")
            if path.is_file()
            and (
                path.name.endswith(".tar")
                or path.name.endswith(".tar.gz")
                or path.name.endswith(".tgz")
            )
        ]

    def _cache_bytes(self) -> int:
        total = 0
        for path in self.root.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    continue
        return total

    def _try_evict(self, path: Path) -> bool:
        if fcntl is None:
            try:
                path.unlink()
                return True
            except OSError:
                return False
        try:
            handle = path.open("rb")
        except OSError:
            return False
        try:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            try:
                path.unlink()
                return True
            except OSError:
                return False
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def _evict_for(self, required_bytes: int) -> None:
        max_bytes = int(self.max_gb * 1e9)
        min_free = int(self.spec.min_free_gb * 1e9)

        def _mtime(path: Path) -> float:
            try:
                return path.stat().st_mtime
            except OSError:
                return 0.0

        candidates = sorted(self._cache_files(), key=_mtime)
        for path in candidates:
            cache_bytes = self._cache_bytes()
            free = shutil.disk_usage(self.root).free
            if cache_bytes + required_bytes <= max_bytes and free - required_bytes >= min_free:
                return
            if self._try_evict(path):
                logger.info("Evicted cached shard %s", path)
        cache_bytes = self._cache_bytes()
        free = shutil.disk_usage(self.root).free
        if cache_bytes + required_bytes > max_bytes or free - required_bytes < min_free:
            raise RuntimeError(
                f"Shard cache {self.root} cannot free enough space for "
                f"{required_bytes} bytes (cache={cache_bytes}, free={free})."
            )

    def _ensure_space(self, required_bytes: int) -> None:
        required = max(0, int(required_bytes))
        max_bytes = int(self.max_gb * 1e9)
        min_free = int(self.spec.min_free_gb * 1e9)
        cache_bytes = self._cache_bytes()
        free = shutil.disk_usage(self.root).free
        if cache_bytes + required <= max_bytes and free - required >= min_free:
            return
        self._evict_for(required)

    def _download_file(self, remote_path: str, target: Path, expected_size: int = 0) -> Path:
        expected = int(expected_size) if expected_size and expected_size > 0 else None
        if target.is_file():
            size = target.stat().st_size
            if expected is None or size == expected:
                os.utime(target, None)
                return target
            target.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_space(expected or 0)
        lock_path = self.locks_dir / (_sha1(remote_path) + ".download.lock")
        with _flock(lock_path, exclusive=True, blocking=True):
            if target.is_file():
                size = target.stat().st_size
                if expected is None or size == expected:
                    os.utime(target, None)
                    return target
                target.unlink()
            self._download_attempts(remote_path, target, expected)
        os.utime(target, None)
        return target

    def _download_attempts(
        self, remote_path: str, target: Path, expected: Optional[int]
    ) -> None:
        part = target.with_name(target.name + ".part")
        attempts = max(1, self.spec.retries + 1)
        for attempt in range(attempts):
            downloaded = part.stat().st_size if part.is_file() else 0
            if expected is not None and downloaded == expected:
                os.replace(part, target)
                return
            if expected is not None and downloaded > expected:
                part.unlink()
                downloaded = 0
            headers = {"User-Agent": "qwen3-asr-webdataset-cache"}
            if downloaded:
                headers["Range"] = f"bytes={downloaded}-"
            try:
                url = self._remote_url(remote_path)
                with requests.get(
                    url,
                    stream=True,
                    cookies=self._cookie_jar(),
                    headers=headers,
                    timeout=self.spec.timeout_sec,
                ) as response:
                    if (
                        response.status_code == 416
                        and downloaded
                        and (expected is None or downloaded == expected)
                    ):
                        os.replace(part, target)
                        return
                    response.raise_for_status()
                    mode = "ab" if response.status_code == 206 and downloaded else "wb"
                    if response.status_code == 200:
                        downloaded = 0
                        mode = "wb"
                    with part.open(mode) as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                size = part.stat().st_size
                if expected is not None and size != expected:
                    raise RuntimeError(
                        f"Downloaded {remote_path} has {size} bytes, expected {expected}."
                    )
                os.replace(part, target)
                return
            except Exception as exc:
                logger.warning(
                    "Download attempt %d/%d failed for %s: %s",
                    attempt + 1,
                    attempts,
                    remote_path,
                    exc,
                )
                if attempt + 1 >= attempts:
                    raise RuntimeError(
                        f"Failed to download {remote_path} from {self.repo_id}: {exc}"
                    ) from exc
                time.sleep(min(30, 2 ** attempt))

    def ensure_local(self, shard: ShardInfo) -> Path:
        if not shard.remote_path:
            return shard.path
        if shard.path.is_file():
            if shard.size_bytes <= 0 or shard.path.stat().st_size == shard.size_bytes:
                os.utime(shard.path, None)
                return shard.path
        return self._download_file(shard.remote_path, shard.path, shard.size_bytes)

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.spec.max_concurrent_downloads
            )
        return self._executor

    def prefetch(self, shard: ShardInfo) -> None:
        if not shard.remote_path:
            return
        if shard.path.is_file():
            if shard.size_bytes <= 0 or shard.path.stat().st_size == shard.size_bytes:
                return
        key = str(shard.path)
        with self._future_lock:
            future = self._futures.get(key)
            if future is not None and not future.done():
                return
            future = self._get_executor().submit(self.ensure_local, shard)
            self._futures[key] = future
            future.add_done_callback(lambda done: self._prefetch_done(key, done))

    def _prefetch_done(self, key: str, future: Future) -> None:
        with self._future_lock:
            self._futures.pop(key, None)
        exc = future.exception()
        if exc is not None:
            logger.warning("Prefetch failed for %s: %s", key, exc)

    @contextmanager
    def lease(self, shard: ShardInfo) -> Iterator[Path]:
        path = shard.path
        if not path.is_file():
            path = self.ensure_local(shard)
        if fcntl is None:
            yield path
            return
        for attempt in range(2):
            try:
                handle = path.open("rb")
                break
            except FileNotFoundError:
                if attempt == 0:
                    path = self.ensure_local(shard)
                    continue
                raise
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            yield path
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        with self._future_lock:
            self._futures.clear()
