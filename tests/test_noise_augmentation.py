"""Unit tests for finetuning.noise_augmentation."""

import os
import random
import tempfile
import unittest

import numpy as np
import soundfile as sf

from finetuning.noise_augmentation import (
    NoiseLibrary,
    apply_add_noise,
    maybe_add_noise,
    mix_noise_at_snr,
    prepare_noise_segment,
)


class NoiseAugmentationTest(unittest.TestCase):
    def test_mix_noise_at_snr_keeps_shape(self):
        wav = np.zeros(8000, dtype=np.float32)
        wav[100:200] = 0.5
        out = mix_noise_at_snr(wav, snr_db=10.0)
        self.assertEqual(out.shape, wav.shape)
        self.assertEqual(out.dtype, np.float32)

    def test_apply_add_noise_alias(self):
        wav = np.random.randn(4000).astype(np.float32) * 0.01
        out = apply_add_noise(wav, snr_db=15.0)
        self.assertEqual(out.shape, wav.shape)

    def test_prepare_noise_segment_tiles_short_noise(self):
        noise = np.ones(100, dtype=np.float32)
        out = prepare_noise_segment(noise, 250)
        self.assertEqual(len(out), 250)
        self.assertTrue(np.allclose(out[:100], 1.0))
        self.assertTrue(np.allclose(out[100:200], 1.0))
        self.assertTrue(np.allclose(out[200:250], 1.0))

    def test_prepare_noise_segment_truncates_long_noise(self):
        noise = np.arange(500, dtype=np.float32)
        out = prepare_noise_segment(noise, 120)
        self.assertEqual(len(out), 120)
        self.assertTrue(np.allclose(out, noise[:120]))

    def test_prepare_noise_segment_empty_returns_empty(self):
        out = prepare_noise_segment(np.array([], dtype=np.float32), 100)
        self.assertEqual(out.size, 0)

    def test_noise_library_from_empty_dir_returns_none(self):
        self.assertIsNone(NoiseLibrary.from_dir(""))
        self.assertIsNone(NoiseLibrary.from_dir("   "))

    def test_noise_library_raises_on_missing_wavs(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                NoiseLibrary.from_dir(td)

    def test_noise_library_sample_segment(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "noise.wav")
            sf.write(path, np.random.randn(1600).astype(np.float32), 16000)
            lib = NoiseLibrary.from_dir(td)
            self.assertIsNotNone(lib)
            assert lib is not None
            self.assertTrue(lib.enabled)
            self.assertEqual(lib.num_files, 1)
            rng = random.Random(0)
            seg = lib.sample_segment(8000, 16000, rng)
            self.assertIsNotNone(seg)
            assert seg is not None
            self.assertEqual(len(seg), 8000)

    def test_maybe_add_noise_prob_zero_unchanged(self):
        wav = np.ones(1000, dtype=np.float32) * 0.1
        rng = random.Random(0)
        out = maybe_add_noise(
            wav, 16000, rng, prob=0.0, snr_min=5.0, snr_max=20.0, library=None
        )
        self.assertTrue(np.array_equal(out, wav))

    def test_maybe_add_noise_prob_one_changes_waveform(self):
        wav = np.ones(1000, dtype=np.float32) * 0.1
        rng = random.Random(0)
        out = maybe_add_noise(
            wav, 16000, rng, prob=1.0, snr_min=5.0, snr_max=20.0, library=None
        )
        self.assertEqual(out.shape, wav.shape)
        self.assertFalse(np.allclose(out, wav))


if __name__ == "__main__":
    unittest.main()
