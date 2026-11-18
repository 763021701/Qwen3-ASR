# coding=utf-8
"""Custom noise-library augmentation for Qwen3-ASR fine-tuning."""

from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import librosa
import numpy as np
import torch

try:
    import pyloudnorm as pyln

    _LUFS_METERS: dict[int, pyln.Meter] = {}

    def _integrated_lufs(wav: np.ndarray, sr: int) -> Optional[float]:
        if len(wav) < int(0.4 * sr) or float(np.max(np.abs(wav))) < 1e-6:
            return None
        meter = _LUFS_METERS.get(sr)
        if meter is None:
            meter = _LUFS_METERS[sr] = pyln.Meter(sr)
        try:
            return float(meter.integrated_loudness(wav.astype(np.float64)))
        except Exception:
            return None

    def normalize_to_lufs(wav: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
        cur = _integrated_lufs(wav, sr)
        if cur is None or not np.isfinite(cur):
            return wav
        out = pyln.normalize.loudness(wav.astype(np.float64), cur, float(target_lufs))
        peak = float(np.max(np.abs(out)))
        if peak > 1.0:
            out = out / peak * 0.99
        return out.astype(np.float32, copy=False)

except ImportError:

    def normalize_to_lufs(wav: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
        rms = float(np.sqrt(np.mean(wav**2))) + 1e-12
        target_rms = 10 ** (float(target_lufs) / 20.0)
        out = wav * (target_rms / rms)
        peak = float(np.max(np.abs(out)))
        if peak > 1.0:
            out = out / peak * 0.99
        return out.astype(np.float32, copy=False)


@dataclass(frozen=True)
class SnrTier:
    """Weighted SNR range sampled uniformly within [snr_min, snr_max]."""

    snr_min: float
    snr_max: float
    weight: float = 1.0


def parse_noise_snr_tiers(spec: str) -> Tuple[SnrTier, ...]:
    """Parse ``min:max[:weight]`` tiers separated by commas."""
    raw = (spec or "").strip()
    if not raw:
        return ()
    tiers: list[SnrTier] = []
    for part in raw.split(","):
        piece = part.strip()
        if not piece:
            continue
        fields = [x.strip() for x in piece.split(":")]
        if len(fields) not in (2, 3):
            raise ValueError(
                f"Invalid noise_snr_tiers entry {piece!r}; expected min:max or min:max:weight"
            )
        lo, hi = float(fields[0]), float(fields[1])
        weight = float(fields[2]) if len(fields) == 3 else 1.0
        if weight <= 0:
            raise ValueError(f"Tier weight must be positive, got {weight} in {piece!r}")
        tiers.append(SnrTier(snr_min=lo, snr_max=hi, weight=weight))
    if not tiers:
        return ()
    return tuple(tiers)


def sample_snr_db(
    rng: random.Random,
    *,
    tiers: Sequence[SnrTier],
    snr_min: float,
    snr_max: float,
) -> float:
    if tiers:
        weights = [t.weight for t in tiers]
        tier = rng.choices(list(tiers), weights=weights, k=1)[0]
        lo, hi = tier.snr_min, tier.snr_max
    else:
        lo, hi = snr_min, snr_max
    return rng.uniform(min(lo, hi), max(lo, hi))


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
    wav: np.ndarray,
    snr_db: float,
    noise_wav: Optional[np.ndarray] = None,
    *,
    sr: Optional[int] = None,
    post_lufs: Optional[float] = None,
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
    out = mixed.numpy().astype(np.float32)
    if post_lufs is not None and sr is not None and sr > 0:
        out = normalize_to_lufs(out, int(sr), float(post_lufs))
    return out


def apply_add_noise(
    wav: np.ndarray, snr_db: float, noise_wav: Optional[np.ndarray] = None
) -> np.ndarray:
    """Backward-compatible alias for :func:`mix_noise_at_snr`."""
    return mix_noise_at_snr(wav, snr_db, noise_wav=noise_wav)


@dataclass
class NoiseLibrary:
    """Directory-backed pool of real noise wav files for online augmentation."""

    noise_dirs: tuple[str, ...] = ()
    _files: list[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        dirs = tuple(d.strip() for d in self.noise_dirs if (d or "").strip())
        self.noise_dirs = dirs
        if not dirs:
            self._files = []
            return
        seen: set[str] = set()
        noise_files: list[str] = []
        for noise_dir in dirs:
            for path in glob.glob(os.path.join(noise_dir, "**", "*.wav"), recursive=True):
                real = os.path.realpath(path)
                if real not in seen:
                    seen.add(real)
                    noise_files.append(path)
        if not noise_files:
            raise FileNotFoundError(f"No .wav noise files found under {list(dirs)!r}")
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
        return cls.from_dirs([noise_dir])

    @classmethod
    def from_dirs(cls, noise_dirs: Sequence[str]) -> Optional["NoiseLibrary"]:
        """Return a library spanning one or more roots, or ``None`` when unset."""
        dirs = tuple(d.strip() for d in noise_dirs if (d or "").strip())
        if not dirs:
            return None
        return cls(noise_dirs=dirs)

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
    snr_tiers: Sequence[SnrTier] = (),
    post_lufs: Optional[float] = None,
) -> np.ndarray:
    """Apply AddNoise augmentation with probability ``prob``."""
    if rng.random() >= prob:
        return wav
    snr = sample_snr_db(rng, tiers=snr_tiers, snr_min=snr_min, snr_max=snr_max)
    noise_wav = None
    if library is not None and library.enabled:
        noise_wav = library.sample_segment(len(wav), sr, rng)
    return mix_noise_at_snr(
        wav, snr, noise_wav=noise_wav, sr=sr, post_lufs=post_lufs
    )
