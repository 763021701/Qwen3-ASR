# coding=utf-8
"""Tests for the ModelScope shard cache and provider (P3)."""

import functools
import http.server
import itertools
import json
import os
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from finetuning.data_pipeline.cache import ShardCache
from finetuning.data_pipeline.datasets import LocalWebDataset
from finetuning.data_pipeline.index import ShardInfo
from finetuning.data_pipeline.spec import CacheSpec, Fields, MixtureSpec, SourceSpec


def _source(path_in_repo: str = "shards") -> SourceSpec:
    return SourceSpec(
        name="ms",
        backend="webdataset",
        provider="modelscope",
        repo_id="owner/data",
        revision="master",
        path_in_repo=path_in_repo,
        split="train",
        language="Chinese",
        fields=Fields(audio="wav", text="txt", meta="json"),
    )


def _cache(tmp: Path, max_gb: float = 1.0) -> ShardCache:
    spec = CacheSpec(dir=str(tmp / "cache"), max_gb=max_gb, min_free_gb=0)
    return ShardCache(_source(), spec)


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        return None


def _write_tiny_tar(path: Path, key: str) -> None:
    import io
    import tarfile

    import numpy as np
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(1600, dtype=np.float32), 16000, format="WAV", subtype="PCM_16")
    members = {
        f"{key}.wav": buffer.getvalue(),
        f"{key}.txt": b"hello",
        f"{key}.json": b'{"duration": 0.1}',
    }
    with tarfile.open(path, "w") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


class ShardCacheTest(unittest.TestCase):
    def test_index_from_remote_listing_and_dataset_info(self):
        with tempfile.TemporaryDirectory() as td:
            cache = _cache(Path(td))
            cache._files = [
                {"path": "shards/a.tar", "size": 100},
                {"path": "shards/b.tar", "size": 200},
                {"path": "shards/readme.txt", "size": 10},
            ]
            cache._dataset_info = {
                "splits": {
                    "train": {
                        "shards": [
                            {"file": "a.tar", "samples": 10, "hours": 1.5},
                            {"file": "b.tar", "samples": 20, "hours": 2.5},
                        ]
                    }
                }
            }
            cache._dataset_info_loaded = True
            index = cache.index_source()
            self.assertEqual(len(index.shards), 2)
            self.assertEqual(index.total_samples, 30)
            self.assertAlmostEqual(index.total_hours, 4.0)
            self.assertEqual(index.shards[0].remote_path, "shards/a.tar")
            self.assertEqual(index.shards[0].num_samples, 10)
            self.assertTrue(str(index.shards[0].path).endswith("shards/a.tar"))
            cache.close()

    def test_index_cache_is_reused_without_api(self):

        with tempfile.TemporaryDirectory() as td:
            cache = _cache(Path(td))
            cache._index_path.parent.mkdir(parents=True, exist_ok=True)
            cache._index_path.write_text(
                json.dumps({
                    "fetched_at": time.time(),
                    "files": [{"path": "shards/a.tar", "size": 100}],
                    "dataset_info": {
                        "splits": {"train": {"shards": [
                            {"file": "a.tar", "samples": 1, "hours": 0.1}
                        ]}}
                    },
                }),
                encoding="utf-8",
            )
            cache._files = None
            index = cache.index_source()
            self.assertEqual(len(index.shards), 1)
            self.assertEqual(index.total_samples, 1)
            cache.close()

    def test_download_and_reuse(self):

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            served = root / "served"
            served.mkdir()
            payload = b"shard-bytes" * 1024
            (served / "shards").mkdir()
            (served / "shards" / "a.tar").write_bytes(payload)
            handler = functools.partial(_QuietHandler, directory=str(served))
            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                cache = _cache(root)
                port = server.server_address[1]
                cache._remote_url = lambda remote: f"http://127.0.0.1:{port}/{remote}"
                cache._cookie_jar = lambda: None
                target = cache.local_path("shards/a.tar")
                cache._download_file("shards/a.tar", target, len(payload))
                self.assertEqual(target.read_bytes(), payload)
                cache._remote_url = lambda remote: (_ for _ in ()).throw(
                    AssertionError("cache hit should not download")
                )
                cache._download_file("shards/a.tar", target, len(payload))
                cache.close()
            finally:
                server.shutdown()
                server.server_close()

    def test_lru_eviction_respects_max_cache(self):

        with tempfile.TemporaryDirectory() as td:
            cache = _cache(Path(td), max_gb=0.000002)
            cache.root.mkdir(parents=True, exist_ok=True)
            first = cache.root / "a.tar"
            second = cache.root / "b.tar"
            first.write_bytes(b"x" * 1500)
            second.write_bytes(b"x" * 1500)
            os.utime(first, (time.time() - 100, time.time() - 100))
            cache._ensure_space(0)
            remaining = sum(path.stat().st_size for path in cache.root.glob("*.tar"))
            self.assertLessEqual(remaining, 2000)
            cache.close()

    def test_prefetch_downloads_once(self):

        class _FakeDownloadCache(ShardCache):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.calls = []

            def _download_file(self, remote, target, expected_size=0):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"x" * int(expected_size))
                self.calls.append(remote)
                return target

        with tempfile.TemporaryDirectory() as td:
            cache = _FakeDownloadCache(
                _source(), CacheSpec(dir=str(Path(td) / "cache"), max_gb=1, min_free_gb=0)
            )
            shard = ShardInfo(
                path=cache.local_path("shards/a.tar"),
                size_bytes=4,
                remote_path="shards/a.tar",
            )
            cache.prefetch(shard)
            cache.close()
            self.assertEqual(cache.calls, ["shards/a.tar"])
            self.assertTrue(shard.path.is_file())
            cache.ensure_local(shard)
            self.assertEqual(cache.calls, ["shards/a.tar"])


class LocalWebDatasetProviderTest(unittest.TestCase):
    def test_modelscope_provider_lifecycle_and_prefetch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shards = []
            for key in ("a", "b"):
                tar_path = root / f"{key}.tar"
                _write_tiny_tar(tar_path, key)
                shards.append(
                    ShardInfo(
                        path=tar_path,
                        size_bytes=tar_path.stat().st_size,
                        remote_path=f"shards/{key}.tar",
                    )
                )
            calls = []

            class FakeProvider:
                def __init__(self, *args, **kwargs):
                    pass

                def ensure_local(self, shard):
                    calls.append("ensure")
                    return shard.path

                def prefetch(self, shard):
                    calls.append("prefetch")

                @contextmanager
                def lease(self, shard):
                    calls.append("lease")
                    yield shard.path

                def close(self):
                    calls.append("close")

            dataset = LocalWebDataset(
                _source(),
                shards,
                MixtureSpec(seed=1, sample_shuffle_buffer=1),
                provider_mode="modelscope",
                cache_spec=CacheSpec(dir=str(root / "cache"), max_gb=1, min_free_gb=0),
                prefetch_shards=1,
            )
            with patch(
                "finetuning.data_pipeline.providers.ModelScopeShardProvider",
                FakeProvider,
            ):
                iterator = iter(dataset)
                sample = next(iterator)
                iterator.close()
            self.assertIn(sample["key"], {"a", "b"})
            self.assertIn("prefetch", calls)
            self.assertIn("ensure", calls)
            self.assertIn("lease", calls)
            self.assertIn("close", calls)


if __name__ == "__main__":
    unittest.main()
