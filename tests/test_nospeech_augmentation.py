"""Unit tests for finetuning.nospeech_augmentation."""

import random
import unittest

import numpy as np

from finetuning.nospeech_augmentation import (
    NoSpeechAugmentConfig,
    apply_nospeech_augment,
    extract_asr_text,
    merge_asr_targets,
    try_dual_concat,
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

    def test_merge_asr_targets(self):
        t1 = "language English<asr_text>hello world"
        t2 = "language English<asr_text>foo bar"
        merged = merge_asr_targets(t1, t2)
        self.assertEqual(merged, "language English<asr_text>hello world foo bar")
        self.assertEqual(extract_asr_text(merged), "hello world foo bar")

    def test_try_dual_concat_within_limit(self):
        sr = 16000
        cfg = NoSpeechAugmentConfig(enabled=True, dual_max_speech_sec=30.0)
        wavs = [np.ones(sr, dtype=np.float32), np.ones(sr, dtype=np.float32)]
        targets = [
            "language English<asr_text>one",
            "language English<asr_text>two",
        ]
        rng = random.Random(0)
        result = try_dual_concat(0, wavs, targets, rng, cfg, sr)
        self.assertIsNotNone(result)
        merged_wav, merged_target = result
        self.assertGreater(len(merged_wav), 2 * sr)
        self.assertIn("one two", extract_asr_text(merged_target))

    def test_try_dual_concat_exceeds_limit(self):
        sr = 16000
        cfg = NoSpeechAugmentConfig(enabled=True, dual_max_speech_sec=1.0)
        wavs = [np.ones(sr, dtype=np.float32), np.ones(sr, dtype=np.float32)]
        targets = [
            "language English<asr_text>one",
            "language English<asr_text>two",
        ]
        rng = random.Random(0)
        self.assertIsNone(try_dual_concat(0, wavs, targets, rng, cfg, sr))

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
            if set(options) == {"leading", "trailing", "dual_concat"}:
                return "trailing"
            return original_choice(options)

        rng.choice = forced_choice  # type: ignore[method-assign]
        out_wavs, out_targets = apply_nospeech_augment(
            wavs, targets, [1], rng, cfg, sr
        )
        self.assertGreater(len(out_wavs[0]), sr)
        self.assertEqual(out_targets[0], targets[0])

    def test_apply_nospeech_dual_concat_updates_target(self):
        sr = 16000
        wavs = [
            np.ones(sr, dtype=np.float32),
            np.ones(sr, dtype=np.float32) * 2.0,
        ]
        targets = [
            "language English<asr_text>alpha",
            "language English<asr_text>beta",
        ]
        cfg = NoSpeechAugmentConfig(
            enabled=True,
            prob=1.0,
            pad_min_sec=0.5,
            pad_max_sec=0.5,
            dual_max_speech_sec=30.0,
        )
        rng = random.Random(2)
        original_choice = rng.choice

        def forced_choice(options):
            if set(options) == {"leading", "trailing", "dual_concat"}:
                return "dual_concat"
            return original_choice(options)

        rng.choice = forced_choice  # type: ignore[method-assign]
        out_wavs, out_targets = apply_nospeech_augment(
            wavs, targets, [1, 0], rng, cfg, sr
        )
        self.assertGreater(len(out_wavs[0]), sr)
        self.assertIn("alpha beta", extract_asr_text(out_targets[0]))


if __name__ == "__main__":
    unittest.main()
