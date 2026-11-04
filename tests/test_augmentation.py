"""Unit tests for collator-side speed perturbation + SpecAugment."""

import random
import unittest

import numpy as np
import torch

from finetuning.data_pipeline import AugmentationSpec, SpecAugmentSpec, SpeedAugmentSpec
from finetuning.qwen3_asr_sft import (
    AudioAugmentConfig,
    _corpus_character_error_rate,
    apply_specaugment,
    apply_speed_perturbation,
    augment_waveform,
)


class AugmentationTest(unittest.TestCase):
    def test_config_can_be_disabled(self):
        cfg = AudioAugmentConfig(enabled=False)
        self.assertFalse(cfg.enabled)

    def test_from_data_config_spec(self):
        spec = AugmentationSpec(
            speed=SpeedAugmentSpec(prob=0.2, factors=(0.9, 1.1)),
            specaug=SpecAugmentSpec(prob=0.3, time_mask_param=12),
        )
        cfg = AudioAugmentConfig.from_spec(spec)
        self.assertTrue(cfg.speed_enabled)
        self.assertEqual(cfg.speed_prob, 0.2)
        self.assertEqual(cfg.speed_factors, (0.9, 1.1))
        self.assertTrue(cfg.specaug_enabled)
        self.assertEqual(cfg.specaug_prob, 0.3)
        self.assertEqual(cfg.specaug_time_mask_param, 12)

    def test_speed_perturbation_changes_length(self):
        sr = 16000
        wav = np.sin(np.linspace(0, 8 * np.pi, sr)).astype(np.float32)
        out = apply_speed_perturbation(wav, sr, 1.1)
        self.assertEqual(out.dtype, np.float32)
        self.assertLess(len(out), len(wav))

    def test_speed_only_waveform_augmentation(self):
        cfg = AudioAugmentConfig(
            enabled=True,
            speed_enabled=True,
            speed_prob=1.0,
            speed_factors=(1.1,),
            specaug_enabled=False,
        )
        wav = np.ones(4000, dtype=np.float32) * 0.01
        out = augment_waveform(wav, 16000, cfg, random.Random(0))
        self.assertEqual(out.dtype, np.float32)
        self.assertLess(len(out), len(wav))

    def test_speed_disabled_returns_input(self):
        cfg = AudioAugmentConfig(
            enabled=True,
            speed_enabled=False,
            specaug_enabled=False,
        )
        wav = np.ones(4000, dtype=np.float32) * 0.01
        out = augment_waveform(wav, 16000, cfg, random.Random(0))
        np.testing.assert_array_equal(out, wav)

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
            speed_enabled=False,
            specaug_enabled=True,
            specaug_prob=1.0,
            specaug_time_mask_param=10,
            specaug_freq_mask_param=8,
            specaug_num_time_masks=1,
            specaug_num_freq_masks=1,
        )
        feats = torch.ones(2, 128, 100)
        out = apply_specaugment(feats, cfg, random.Random(42))
        self.assertEqual(out.shape, feats.shape)
        self.assertTrue(torch.any(out == 0))

    def test_specaugment_disabled_returns_input(self):
        cfg = AudioAugmentConfig(
            enabled=True,
            speed_enabled=False,
            specaug_enabled=False,
        )
        feats = torch.ones(2, 128, 100)
        out = apply_specaugment(feats, cfg, random.Random(42))
        self.assertIs(out, feats)


if __name__ == "__main__":
    unittest.main()
