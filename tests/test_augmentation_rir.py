# coding=utf-8
"""Tests for real-RIR mixed reverb augmentation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from finetuning.augmentation.service import (
    SimulationConfig,
    SimulationService,
    augment_clean_sample,
)
from finetuning.augmentation.simulator import (
    NoisePool,
    RIRPool,
    apply_scenario,
    resolve_scenario_weights,
)
from finetuning.augmentation.simulator.primitives import add_reverb
from finetuning.data_pipeline.spec import parse_data_config

SR = 16000


def _write_wav(path: Path, wav: np.ndarray, sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(wav, dtype=np.float32), sr, subtype="PCM_16")


def _make_corpus(root: Path):
    corpus = root / "corpus"
    real = corpus / "real_rirs_isotropic_noises"
    real.mkdir(parents=True)
    length = 4000
    rng = np.random.default_rng(0)
    rir = 0.3 * np.exp(-6.0 * np.arange(length) / SR) * rng.standard_normal(length)
    rir[0] += 1.0
    _write_wav(real / "rir_small.wav", rir.astype(np.float32))
    (real / "rir_list").write_text(
        "--rir-id 1 --room-id smallroom1 "
        "RIRS_NOISES/real_rirs_isotropic_noises/rir_small.wav\n",
        encoding="utf-8",
    )
    points = corpus / "pointsource_noises"
    points.mkdir()
    _write_wav(points / "noise-free-sound-0000.wav", np.zeros(1600, dtype=np.float32))
    linked = root / "linked"
    linked.mkdir()
    (linked / "RIRS_NOISES").symlink_to(corpus, target_is_directory=True)
    return linked


def _clean_samples(count: int):
    samples = []
    t = np.arange(SR, dtype=np.float32) / SR
    for index in range(count):
        samples.append(
            {
                "audio": (0.1 * np.sin(2 * np.pi * (220 + index * 20) * t)).astype(
                    np.float32
                ),
                "target": f"language Chinese<asr_text>测试{index}",
                "prompt": "",
                "source": "test",
                "key": f"k{index}",
                "language": "Chinese",
                "duration": 1.0,
                "sample_seed": 100 + index,
            }
        )
    return samples


def _service_config(root: Path, rir_dir: Path) -> SimulationConfig:
    return SimulationConfig(
        spool_dir=str(root / "spool"),
        scenarios=("far_field",),
        clean_prob=0.0,
        noise_dir=None,
        reverb_mode="real",
        rir_dir=str(rir_dir),
        rir_real_prob=1.0,
        sample_rate=SR,
        audio_format="wav",
        max_spool_bytes=1_000_000,
        num_simulators=1,
        clean_queue_size=2,
        ready_queue_size=4,
        stall_timeout_sec=10.0,
        spool_ttl_sec=0.0,
        start_method="spawn",
    )


class RIRPoolTest(unittest.TestCase):
    def test_follows_symlinks_and_excludes_pointsource_noises(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            linked = _make_corpus(root)
            pool = RIRPool(str(linked), sample_rate=SR)
            self.assertEqual(len(pool), 1)
            self.assertEqual(pool.files[0].source, "real")
            self.assertEqual(pool.files[0].category, "small")
            self.assertNotIn("pointsource", str(pool.files[0].path))

    def test_noise_pool_follows_symlinks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            musan = root / "noise_root" / "musan"
            _write_wav(musan / "noise" / "sample.wav", np.zeros(1600, dtype=np.float32))
            (root / "noise_root" / "musan_link").symlink_to(
                musan, target_is_directory=True
            )
            pool = NoisePool(str(root / "noise_root"), sample_rate=SR)
            self.assertEqual(len(pool.files), 1)
            self.assertEqual(pool.files[0].name, "sample.wav")

    def test_real_prob_selects_real_or_simulated(self):
        with tempfile.TemporaryDirectory() as td:
            linked = _make_corpus(Path(td))
            corpus = linked / "RIRS_NOISES"
            simulated = corpus / "simulated_rirs" / "smallroom"
            _write_wav(simulated / "sim.wav", np.ones(160, dtype=np.float32))
            (simulated / "rir_list").write_text(
                "--rir-id 2 --room-id small-Room001 "
                "RIRS_NOISES/simulated_rirs/smallroom/sim.wav\n",
                encoding="utf-8",
            )
            real_pool = RIRPool(str(linked), sample_rate=SR, real_prob=1.0)
            simulated_pool = RIRPool(str(linked), sample_rate=SR, real_prob=0.0)
            real_choice = real_pool._select(np.random.default_rng(0), room_size=0.2)
            simulated_choice = simulated_pool._select(
                np.random.default_rng(0), room_size=0.2
            )
            self.assertTrue(all(info.source == "real" for info in real_choice))
            self.assertTrue(
                all(info.source == "simulated" for info in simulated_choice)
            )

    def test_fallback_scan_without_list_excludes_pointsource_noises(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rirs = root / "rirs"
            _write_wav(rirs / "a.wav", np.ones(160, dtype=np.float32))
            _write_wav(
                rirs / "pointsource_noises" / "noise.wav",
                np.ones(160, dtype=np.float32),
            )
            pool = RIRPool(str(rirs), sample_rate=SR)
            self.assertEqual([info.path.name for info in pool.files], ["a.wav"])

    def test_real_and_synthetic_reverb_differ(self):
        with tempfile.TemporaryDirectory() as td:
            linked = _make_corpus(Path(td))
            pool = RIRPool(str(linked), sample_rate=SR)
            wav = np.sin(2 * np.pi * 440 * np.arange(SR) / SR).astype(np.float32) * 0.1
            common = dict(
                room_size=0.5,
                damping=0.5,
                wet_level=0.5,
                dry_level=0.5,
                rir_pool=pool,
            )
            synthetic = add_reverb(
                wav, SR, np.random.default_rng(0), use_real_rir=False, **common
            )
            real = add_reverb(
                wav, SR, np.random.default_rng(0), use_real_rir=True, **common
            )
            self.assertEqual(synthetic.shape, real.shape)
            self.assertTrue(np.isfinite(real).all())
            self.assertFalse(np.allclose(synthetic, real))
            self.assertLessEqual(float(np.max(np.abs(real))), 1.0)

    def test_mixed_mode_extremes_are_synthetic_or_real(self):
        with tempfile.TemporaryDirectory() as td:
            linked = _make_corpus(Path(td))
            pool = RIRPool(str(linked), sample_rate=SR)
            noise = NoisePool(None)
            wav = np.sin(2 * np.pi * 440 * np.arange(SR) / SR).astype(np.float32) * 0.1

            def run(**kwargs):
                return apply_scenario(
                    wav,
                    SR,
                    "far_field",
                    0.5,
                    np.random.default_rng(0),
                    noise,
                    rir_pool=pool,
                    **kwargs,
                )

            synthetic = run(reverb_mode="synthetic")
            mixed_zero = run(reverb_mode="mixed", rir_mix_prob=0.0)
            mixed_one = run(reverb_mode="mixed", rir_mix_prob=1.0)
            real = run(reverb_mode="real")
            np.testing.assert_allclose(synthetic, mixed_zero)
            np.testing.assert_allclose(real, mixed_one)
            self.assertFalse(np.allclose(synthetic, real))

    def test_service_uses_real_rir(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            linked = _make_corpus(root)
            service = SimulationService(_clean_samples(3), _service_config(root, linked))
            service.start()
            samples = list(service.dataset)
            snapshot = service.snapshot()
            service.close()
            self.assertEqual(len(samples), 3)
            self.assertEqual(snapshot["produced"], 3)
            self.assertEqual(snapshot["consumed"], 3)
            self.assertEqual(snapshot["errors"], 0)
            self.assertTrue(all(sample["degraded"] for sample in samples))
            self.assertTrue(
                all(np.isfinite(sample["audio"]).all() for sample in samples)
            )

    def test_service_rejects_empty_rir_dir(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            empty = root / "empty_rir"
            empty.mkdir()
            service = SimulationService(_clean_samples(1), _service_config(root, empty))
            with self.assertRaises(RuntimeError):
                service.start()
            service.close()

    def test_augment_clean_sample_accepts_rir_pool(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            linked = _make_corpus(root)
            pool = RIRPool(str(linked), sample_rate=SR)
            config = _service_config(root, linked)
            weights = resolve_scenario_weights(("far_field",), None)
            audio, metadata = augment_clean_sample(
                _clean_samples(1)[0], config, NoisePool(None), weights, pool
            )
            self.assertEqual(audio.shape, (SR,))
            self.assertEqual(metadata["scenario"], "far_field")
            self.assertTrue(np.isfinite(audio).all())


class ReverbConfigTest(unittest.TestCase):
    def test_parses_mixed_reverb(self):
        config = parse_data_config(
            {
                "sources": [
                    {"name": "x", "backend": "jsonl", "path": "/tmp/x.jsonl"}
                ],
                "augmentation": {
                    "simulation": {
                        "enabled": True,
                        "spool_dir": "/tmp/spool",
                        "reverb_mode": "mixed",
                        "rir_dir": "/tmp/rir",
                        "rir_cache_files": 16,
                        "rir_real_prob": 0.5,
                        "rir_mix_prob": 0.7,
                    }
                },
            }
        )
        simulation = config.augmentation.simulation
        self.assertEqual(simulation.reverb_mode, "mixed")
        self.assertEqual(simulation.rir_dir, "/tmp/rir")
        self.assertEqual(simulation.rir_cache_files, 16)
        self.assertEqual(simulation.rir_real_prob, 0.5)
        self.assertEqual(simulation.rir_mix_prob, 0.7)

    def test_requires_rir_dir_for_real_or_mixed(self):
        with self.assertRaises(ValueError):
            parse_data_config(
                {
                    "sources": [
                        {"name": "x", "backend": "jsonl", "path": "/tmp/x.jsonl"}
                    ],
                    "augmentation": {
                        "simulation": {
                            "enabled": True,
                            "spool_dir": "/tmp/spool",
                            "reverb_mode": "mixed",
                        }
                    },
                }
            )

    def test_rejects_unknown_reverb_mode(self):
        with self.assertRaises(ValueError):
            parse_data_config(
                {
                    "sources": [
                        {"name": "x", "backend": "jsonl", "path": "/tmp/x.jsonl"}
                    ],
                    "augmentation": {
                        "simulation": {
                            "enabled": True,
                            "spool_dir": "/tmp/spool",
                            "reverb_mode": "unknown",
                            "rir_dir": "/tmp/rir",
                        }
                    },
                }
            )


if __name__ == "__main__":
    unittest.main()
