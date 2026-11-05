# coding=utf-8
"""Tests for the process-based pseudo-online simulator service."""

import io
import itertools
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
from torch.utils.data import DataLoader

from finetuning.augmentation.service import (
    SimulationConfig,
    SimulationService,
    augment_clean_sample,
)
from finetuning.augmentation.simulator import NoisePool, resolve_scenario_weights
from finetuning.data_pipeline import build_data_pipeline

SR = 16000


def _write_tiny_tar(path: Path, key: str) -> None:
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(1600, dtype=np.float32), SR, format="WAV", subtype="PCM_16")
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


def _clean_samples(count: int):
    return [
        {
            "audio": np.zeros(1600, dtype=np.float32),
            "target": f"language Chinese<asr_text>测试{index}",
            "prompt": "",
            "source": "test",
            "key": f"k{index}",
            "language": "Chinese",
            "duration": 0.1,
            "sample_seed": 100 + index,
        }
        for index in range(count)
    ]


def _identity(value):
    return value


class _FailingIterable:
    def __init__(self, samples):
        self.samples = samples

    def __iter__(self):
        yield from self.samples
        raise RuntimeError("reader boom")


def _config(root: Path, num_simulators: int = 2) -> SimulationConfig:
    return SimulationConfig(
        spool_dir=str(root / "spool"),
        scenarios=("noise",),
        clean_prob=0.0,
        noise_dir=None,
        sample_rate=SR,
        audio_format="wav",
        max_spool_bytes=1_000_000,
        num_simulators=num_simulators,
        clean_queue_size=4,
        ready_queue_size=8,
        stall_timeout_sec=10.0,
        spool_ttl_sec=0.0,
        start_method="spawn",
    )


class SimulatorServiceTest(unittest.TestCase):
    def test_augment_sample_is_deterministic(self):
        with tempfile.TemporaryDirectory() as td:
            config = _config(Path(td))
            weights = resolve_scenario_weights(("noise",), None)
            pool = NoisePool(None, sample_rate=SR)
            sample = _clean_samples(1)[0]
            first, first_meta = augment_clean_sample(sample, config, pool, weights)
            second, second_meta = augment_clean_sample(sample, config, pool, weights)
            np.testing.assert_allclose(first, second)
            self.assertEqual(first_meta["scenario"], "noise")
            self.assertEqual(first_meta["simulated"], True)
            self.assertGreaterEqual(first_meta["severity"], 0.0)
            self.assertLessEqual(first_meta["severity"], 1.0)

    def test_clean_source_bypasses_simulation(self):
        with tempfile.TemporaryDirectory() as td:
            config = _config(Path(td))
            config.clean_sources = ("clean_src",)
            weights = resolve_scenario_weights(("noise",), None)
            pool = NoisePool(None, sample_rate=SR)
            clean = _clean_samples(1)[0]
            clean["source"] = "clean_src"
            audio, metadata = augment_clean_sample(clean, config, pool, weights)
            np.testing.assert_allclose(audio, clean["audio"])
            self.assertEqual(metadata["scenario"], "clean")
            self.assertFalse(metadata["simulated"])
            self.assertEqual(metadata["clean_reason"], "source")

            degraded = _clean_samples(1)[0]
            degraded["source"] = "robust_src"
            _, degraded_metadata = augment_clean_sample(
                degraded, config, pool, weights
            )
            self.assertTrue(degraded_metadata["simulated"])
            self.assertIsNone(degraded_metadata["clean_reason"])

    def test_service_produces_and_releases_spool(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            service = SimulationService(_clean_samples(5), _config(root))
            service.start()
            samples = list(service.dataset)
            snapshot = service.snapshot()
            service.close()
            self.assertEqual(len(samples), 5)
            self.assertEqual(snapshot["produced"], 5)
            self.assertEqual(snapshot["consumed"], 5)
            self.assertEqual(snapshot["errors"], 0)
            self.assertEqual(snapshot["spool_bytes"], 0)
            self.assertEqual(
                {sample["target"].split("<asr_text>")[1] for sample in samples},
                {f"测试{i}" for i in range(5)},
            )
            for sample in samples:
                self.assertIsInstance(sample["audio"], np.ndarray)
                self.assertTrue(sample["degraded"])
                self.assertEqual(sample["scenario"], "noise")
            self.assertFalse(any(path.is_file() for path in (root / "spool").rglob("*")))

    def test_service_works_with_dataloader_worker(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            service = SimulationService(_clean_samples(4), _config(root, num_simulators=1))
            service.start()
            loader = DataLoader(
                service.dataset,
                batch_size=None,
                num_workers=1,
                collate_fn=_identity,
            )
            samples = list(itertools.islice(iter(loader), 3))
            service.close()
            self.assertEqual(len(samples), 3)
            self.assertTrue(all(isinstance(sample["audio"], np.ndarray) for sample in samples))
            self.assertEqual(service.snapshot()["spool_bytes"], 0)

    def test_reader_failure_raises_in_consumer(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            service = SimulationService(
                _FailingIterable(_clean_samples(2)),
                _config(root, num_simulators=1),
            )
            service.start()
            try:
                with self.assertRaises(RuntimeError):
                    list(service.dataset)
                self.assertGreaterEqual(service.snapshot()["errors"], 1)
            finally:
                service.close()

    def test_service_consumes_data_pipeline_iterable(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shard = root / "shard-00000.tar"
            _write_tiny_tar(shard, "s0")
            pipeline = build_data_pipeline({
                "sources": [{
                    "name": "wd",
                    "backend": "webdataset",
                    "path": str(shard),
                    "split": "train",
                    "language": "Chinese",
                    "fields": {"audio": "wav", "text": "txt", "meta": "json"},
                }],
                "mixture": {"seed": 1, "sample_shuffle_buffer": 1},
            })
            service = SimulationService(pipeline.dataset, _config(root, num_simulators=1))
            service.start()
            samples = list(itertools.islice(service.dataset, 2))
            service.close()
            self.assertEqual(len(samples), 2)
            self.assertTrue(all(sample["source"] == "wd" for sample in samples))
            self.assertTrue(all(isinstance(sample["audio"], np.ndarray) for sample in samples))


if __name__ == "__main__":
    unittest.main()
