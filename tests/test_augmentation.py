"""Unit tests for optional online audio augmentation in finetuning."""

import argparse
import unittest

import numpy as np
import torch

from finetuning.noise_augmentation import apply_add_noise
from finetuning.qwen3_asr_sft import (
    AudioAugmentConfig,
    _corpus_character_error_rate,
    apply_specaugment,
    apply_speed_perturbation,
    augment_waveform,
)


class AugmentationTest(unittest.TestCase):
    def test_config_disabled_by_default(self):
        args = argparse.Namespace(augment=0)
        cfg = AudioAugmentConfig.from_args(args)
        self.assertFalse(cfg.enabled)

    def test_speed_perturbation_changes_length(self):
        try:
            import torchaudio  # noqa: F401
        except ImportError:
            self.skipTest("torchaudio not installed")
        sr = 16000
        wav = np.sin(np.linspace(0, 8 * np.pi, sr)).astype(np.float32)
        out = apply_speed_perturbation(wav, sr, 1.1)
        self.assertEqual(out.dtype, np.float32)
        self.assertLess(len(out), len(wav))

    def test_add_noise_keeps_shape(self):
        wav = np.zeros(8000, dtype=np.float32)
        wav[100:200] = 0.5
        out = apply_add_noise(wav, snr_db=10.0)
        self.assertEqual(out.shape, wav.shape)

    def test_noise_can_be_disabled_per_sample(self):
        try:
            import torchaudio  # noqa: F401
        except ImportError:
            self.skipTest("torchaudio not installed")
        cfg = AudioAugmentConfig(
            enabled=True,
            augment_prob=1.0,
            speed_prob=0.0,
            noise_prob=1.0,
        )
        rng = __import__("random").Random(0)
        wav = np.ones(4000, dtype=np.float32) * 0.01
        out = augment_waveform(wav, 16000, cfg, rng, noise_enabled=False)
        np.testing.assert_array_equal(out, wav)

    def test_augment_waveform_with_torchaudio(self):
        try:
            import torchaudio  # noqa: F401
        except ImportError:
            self.skipTest("torchaudio not installed")
        cfg = AudioAugmentConfig(
            enabled=True,
            augment_prob=1.0,
            speed_prob=1.0,
            noise_prob=1.0,
            speed_factors=(1.1,),
        )
        rng = __import__("random").Random(0)
        wav = np.random.randn(4000).astype(np.float32) * 0.01
        out = augment_waveform(wav, 16000, cfg, rng)
        self.assertEqual(out.dtype, np.float32)

    def test_corpus_character_error_rate_ignores_spaces(self):
        cer = _corpus_character_error_rate(["a b", "c"], ["a", "c d"])
        self.assertAlmostEqual(cer, 2.0 / 3.0)

    def test_specaugment_on_mel_batch(self):
        try:
            import torchaudio  # noqa: F401
        except ImportError:
            self.skipTest("torchaudio not installed")
        cfg = AudioAugmentConfig(
            enabled=True,
            specaug_prob=1.0,
            specaug_time_mask_param=10,
            specaug_freq_mask_param=8,
            specaug_num_time_masks=1,
            specaug_num_freq_masks=1,
        )
        feats = torch.ones(2, 128, 100)
        rng = __import__("random").Random(42)
        out = apply_specaugment(feats, cfg, rng)
        self.assertEqual(out.shape, feats.shape)
        self.assertTrue(torch.any(out == 0))


if __name__ == "__main__":
    unittest.main()
