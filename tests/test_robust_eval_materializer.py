# coding=utf-8
"""Tests for the deterministic robust eval materializer."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.materialize_robust_eval import materialize_robust_eval

SR = 16000


def _write_wav(path: Path, wav: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(wav, dtype=np.float32), SR, subtype="PCM_16")


def _rir_dir(root: Path) -> Path:
    rir_root = root / "rirs"
    real = rir_root / "real_rirs_isotropic_noises"
    real.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rir = 0.3 * np.exp(-6.0 * np.arange(4000) / SR) * rng.standard_normal(4000)
    rir[0] += 1.0
    _write_wav(real / "rir.wav", rir.astype(np.float32))
    (real / "rir_list").write_text(
        "--rir-id 1 --room-id smallroom1 "
        "RIRS_NOISES/real_rirs_isotropic_noises/rir.wav\n",
        encoding="utf-8",
    )
    return rir_root


class RobustEvalMaterializerTest(unittest.TestCase):
    def test_materializes_manifest_with_real_rir(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            clean_audio = root / "clean" / "utt.wav"
            t = np.arange(SR, dtype=np.float32) / SR
            _write_wav(clean_audio, 0.1 * np.sin(2 * np.pi * 440 * t))
            rows = [
                {
                    "audio": str(clean_audio),
                    "text": "language Chinese<asr_text>测试",
                    "duration_sec": 1.0,
                }
            ]
            manifest_path = materialize_robust_eval(
                rows,
                str(root / "out"),
                scenarios=("far_field",),
                severity=0.5,
                copies=2,
                seed=7,
                rir_dir=str(_rir_dir(root)),
                reverb_mode="real",
                audio_format="wav",
            )
            items = [
                json.loads(line)
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(items), 2)
            self.assertEqual({item["scenario"] for item in items}, {"far_field"})
            self.assertEqual({item["severity"] for item in items}, {0.5})
            self.assertTrue(all(item["simulated"] for item in items))
            self.assertEqual({item["text"] for item in items}, {rows[0]["text"]})
            self.assertEqual(
                {item["source_audio"] for item in items}, {str(clean_audio)}
            )
            self.assertEqual(len({item["sample_seed"] for item in items}), 2)
            for item in items:
                path = Path(item["audio"])
                self.assertTrue(path.is_file())
                audio, sample_rate = sf.read(str(path), dtype="float32")
                self.assertEqual(sample_rate, SR)
                self.assertTrue(np.isfinite(audio).all())

    def test_balances_scenarios_across_copies(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            clean_audio = root / "clean" / "utt.wav"
            _write_wav(clean_audio, np.zeros(SR, dtype=np.float32))
            rows = [{"audio": str(clean_audio), "text": "x"}]
            manifest_path = materialize_robust_eval(
                rows,
                str(root / "out"),
                scenarios=("far_field", "echo_reverb"),
                severity=0.5,
                copies=2,
                seed=7,
                rir_dir=str(_rir_dir(root)),
                reverb_mode="real",
                audio_format="wav",
            )
            items = [
                json.loads(line)
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(
                {item["scenario"] for item in items}, {"far_field", "echo_reverb"}
            )

    def test_noise_scenarios_require_noise_dir(self):
        rows = [{"audio": "/tmp/missing.wav", "text": "x"}]
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                materialize_robust_eval(
                    rows,
                    str(Path(td) / "out"),
                    scenarios=("noise",),
                    noise_dir=None,
                )


if __name__ == "__main__":
    unittest.main()
