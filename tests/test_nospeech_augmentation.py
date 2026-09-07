"""Unit tests for finetuning.nospeech_augmentation."""

import random
import unittest

import numpy as np

from finetuning.nospeech_augmentation import (
    NoSpeechAugmentConfig,
    apply_nospeech_augment,
    zero_pad,
)


class NoSpeechAugmentationTest(unittest.TestCase):
    def test_zero_pad_lengths(self):
        wav = np.ones(100, dtype=np.float32)
        out = zero_pad(wav, lead_samples=50, trail_samples=30)
        self.assertEqual(len(out), 180)
        self.assertTrue(np.allclose(out[:50], 0.0))
        self.assertTrue(np.allclose(out[50:150], 1.0))
        self.assertTrue(np.allclose(out[150:], 0.0))

    def test_apply_nospeech_prob_zero_unchanged(self):
        sr = 16000
        wavs = [np.ones(sr, dtype=np.float32)]
        targets = ["language English<asr_text>hello"]
        cfg = NoSpeechAugmentConfig(enabled=True, prob=0.0)
        rng = random.Random(0)
        out_wavs, out_targets = apply_nospeech_augment(
            wavs, targets, [1], rng, cfg, sr
        )
        self.assertEqual(len(out_wavs[0]), sr)
        self.assertEqual(out_targets[0], targets[0])

    def test_apply_nospeech_trailing_pad(self):
        sr = 16000
        wavs = [np.ones(sr, dtype=np.float32)]
        targets = ["language English<asr_text>hello"]
        cfg = NoSpeechAugmentConfig(
            enabled=True,
            prob=1.0,
            pad_min_sec=0.5,
            pad_max_sec=0.5,
        )
        rng = random.Random(1)
        original_choice = rng.choice

        def forced_choice(options):
            if set(options) == {"leading", "trailing"}:
                return "trailing"
            return original_choice(options)

        rng.choice = forced_choice  # type: ignore[method-assign]
        out_wavs, out_targets = apply_nospeech_augment(
            wavs, targets, [1], rng, cfg, sr
        )
        self.assertGreater(len(out_wavs[0]), sr)
        self.assertEqual(out_targets[0], targets[0])

    def test_apply_nospeech_leading_pad(self):
        sr = 16000
        wavs = [np.ones(sr, dtype=np.float32)]
        targets = ["language English<asr_text>hello"]
        cfg = NoSpeechAugmentConfig(
            enabled=True,
            prob=1.0,
            pad_min_sec=0.5,
            pad_max_sec=0.5,
        )
        rng = random.Random(1)
        original_choice = rng.choice

        def forced_choice(options):
            if set(options) == {"leading", "trailing"}:
                return "leading"
            return original_choice(options)

        rng.choice = forced_choice  # type: ignore[method-assign]
        out_wavs, out_targets = apply_nospeech_augment(
            wavs, targets, [1], rng, cfg, sr
        )
        self.assertGreater(len(out_wavs[0]), sr)
        self.assertTrue(np.allclose(out_wavs[0][:8000], 0.0))
        self.assertEqual(out_targets[0], targets[0])

    def test_apply_nospeech_skips_aug_flag_zero(self):
        sr = 16000
        wavs = [np.ones(sr, dtype=np.float32)]
        targets = ["language English<asr_text>hello"]
        cfg = NoSpeechAugmentConfig(enabled=True, prob=1.0)
        out_wavs, out_targets = apply_nospeech_augment(
            wavs, targets, [0], random.Random(0), cfg, sr
        )
        self.assertEqual(len(out_wavs[0]), sr)
        self.assertEqual(out_targets[0], targets[0])

    def test_apply_nospeech_targets_unchanged_by_padding(self):
        """Leading/trailing pads never alter the transcript text."""
        sr = 16000
        wavs = [np.ones(sr, dtype=np.float32), np.ones(sr, dtype=np.float32) * 2.0]
        targets = [
            "language English<asr_text>alpha",
            "language English<asr_text>beta",
        ]
        cfg = NoSpeechAugmentConfig(
            enabled=True,
            prob=1.0,
            pad_min_sec=0.5,
            pad_max_sec=3.0,
        )
        out_wavs, out_targets = apply_nospeech_augment(
            wavs, targets, [1, 1], random.Random(3), cfg, sr
        )
        self.assertEqual(out_targets, targets)
        for out, original in zip(out_wavs, wavs):
            self.assertGreater(len(out), len(original))


if __name__ == "__main__":
    unittest.main()
