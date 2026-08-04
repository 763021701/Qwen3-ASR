# coding=utf-8
"""Custom noise-library augmentation for Qwen3-ASR fine-tuning."""

from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import librosa
import numpy as np
import torch


def prepare_noise_segment(noise_wav: np.ndarray, length: int) -> np.ndarray:
    """Tile or truncate a 1-D noise waveform to ``length`` samples."""
    noise = np.asarray(noise_wav, dtype=np.float32)
    if noise.size == 0:
        return noise
    if noise.size < length:
        reps = (length + noise.size - 1) // noise.size
        noise = np.tile(noise, reps)
    return noise[:length]


def mix_noise_at_snr(
    wav: np.ndarray, snr_db: float, noise_wav: Optional[np.ndarray] = None
) -> np.ndarray:
    """Mix noise at the given SNR (dB).

    If ``noise_wav`` is provided (real background noise), it is tiled/truncated
    to the signal length; otherwise synthetic white noise is used.
    """
    signal = torch.from_numpy(wav).float()
    if noise_wav is not None:
        prepared = prepare_noise_segment(noise_wav, signal.numel())
        if prepared.size == 0:
            noise = torch.randn_like(signal)
        else:
            noise = torch.from_numpy(prepared).float()
    else:
        noise = torch.randn_like(signal)
    signal_power = signal.pow(2).mean().clamp(min=1e-10)
    noise_power = noise.pow(2).mean().clamp(min=1e-10)
    snr_linear = 10 ** (snr_db / 10.0)
    scale = torch.sqrt(signal_power / (noise_power * snr_linear))
    mixed = signal + noise * scale
    peak = mixed.abs().max()
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.numpy().astype(np.float32)


def apply_add_noise(
    wav: np.ndarray, snr_db: float, noise_wav: Optional[np.ndarray] = None
) -> np.ndarray:
    """Backward-compatible alias for :func:`mix_noise_at_snr`."""
    return mix_noise_at_snr(wav, snr_db, noise_wav=noise_wav)


@dataclass
class NoiseLibrary:
    """Directory-backed pool of real noise wav files for online augmentation."""

    noise_dir: str = ""
    _files: list[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        noise_dir = (self.noise_dir or "").strip()
        if not noise_dir:
            self._files = []
            return
        noise_files = glob.glob(os.path.join(noise_dir, "**", "*.wav"), recursive=True)
        if not noise_files:
            raise FileNotFoundError(f"No .wav noise files found under {noise_dir!r}")
        self._files = noise_files

    @property
    def enabled(self) -> bool:
        return bool(self._files)

    @property
    def num_files(self) -> int:
        return len(self._files)

    @classmethod
    def from_dir(cls, noise_dir: str) -> Optional["NoiseLibrary"]:
        """Return a library for ``noise_dir``, or ``None`` when unset."""
        noise_dir = (noise_dir or "").strip()
        if not noise_dir:
            return None
        return cls(noise_dir=noise_dir)

    def sample_segment(
        self, num_samples: int, sr: int, rng: random.Random
    ) -> Optional[np.ndarray]:
        """Sample a random noise segment of ``num_samples`` at ``sr`` Hz."""
        if not self._files:
            return None
        path = rng.choice(self._files)
        try:
            noise_wav, _ = librosa.load(path, sr=sr, mono=True)
        except Exception:
            return None
        return prepare_noise_segment(noise_wav, num_samples)


def maybe_add_noise(
    wav: np.ndarray,
    sr: int,
    rng: random.Random,
    *,
    prob: float,
    snr_min: float,
    snr_max: float,
    library: Optional[NoiseLibrary] = None,
) -> np.ndarray:
    """Apply AddNoise augmentation with probability ``prob``."""
    if rng.random() >= prob:
        return wav
    lo, hi = snr_min, snr_max
    snr = rng.uniform(min(lo, hi), max(lo, hi))
    noise_wav = None
    if library is not None and library.enabled:
        noise_wav = library.sample_segment(len(wav), sr, rng)
    return mix_noise_at_snr(wav, snr, noise_wav=noise_wav)
