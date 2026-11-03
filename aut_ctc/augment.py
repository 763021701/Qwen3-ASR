"""Online augmentation (DESIGN.md §8.5): speed perturbation, noise / reverb, SpecAugment on log-mel."""
from __future__ import annotations

from fractions import Fraction
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly


def resample(wav: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return wav
    g = gcd(src_sr, dst_sr)
    return resample_poly(wav, dst_sr // g, src_sr // g).astype(np.float32)


def speed_perturb(wav: np.ndarray, factor: float) -> np.ndarray:
    """Kaldi-style speed change by resampling (tempo and pitch both change)."""
    if factor == 1.0:
        return wav
    frac = Fraction(1 / factor).limit_denominator(100)  # 0.9 -> 10/9: slower and longer
    return resample_poly(wav, frac.numerator, frac.denominator).astype(np.float32)


def add_noise(wav: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    if len(noise) < len(wav):
        noise = np.tile(noise, -(-len(wav) // len(noise)))
    start = rng.integers(0, len(noise) - len(wav) + 1)
    noise = noise[start : start + len(wav)]
    scale = np.sqrt((np.mean(wav**2) + 1e-10) / ((np.mean(noise**2) + 1e-10) * 10 ** (snr_db / 10)))
    return (wav + scale * noise).astype(np.float32)


def add_reverb(wav: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve with an RIR, aligned on its direct-path peak so CTC timing is unchanged; keep the peak level."""
    peak = int(np.argmax(np.abs(rir)))
    out = fftconvolve(wav, rir)[peak : peak + len(wav)]
    return (out * (np.max(np.abs(wav)) / (np.max(np.abs(out)) + 1e-10))).astype(np.float32)


def _read_list(path):
    if not path:
        return []
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _load_mono(path: str, sr: int) -> np.ndarray:
    wav, file_sr = sf.read(path, dtype="float32", always_2d=True)
    return resample(wav.mean(1), file_sr, sr)


class WaveAugment:
    """Speed perturbation on every clip; with probability noise_prob add either noise or reverb (whichever
    lists are given, 50/50 when both)."""

    def __init__(self, speed_perturb=(0.9, 1.0, 1.1), noise_prob=0.3, snr_db=(5, 20), noise_list=None,
                 rir_list=None, sr=16000):
        self.speeds = list(speed_perturb or [1.0])
        self.noise_prob, self.snr_db, self.sr = noise_prob, snr_db, sr
        self.noises, self.rirs = _read_list(noise_list), _read_list(rir_list)

    def __call__(self, wav: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        wav = speed_perturb(wav, float(rng.choice(self.speeds)))
        if (self.noises or self.rirs) and rng.random() < self.noise_prob:
            if self.rirs and (not self.noises or rng.random() < 0.5):
                wav = add_reverb(wav, _load_mono(str(rng.choice(self.rirs)), self.sr))
            else:
                noise = _load_mono(str(rng.choice(self.noises)), self.sr)
                wav = add_noise(wav, noise, rng.uniform(*self.snr_db), rng)
        return wav


def spec_augment(feats, mel_lens, rng: np.random.Generator, freq_masks=2, freq_width=20, time_masks=2,
                 time_width=40):
    """In-place SpecAugment on (B, n_mels, T) log-mel within each utterance's valid frames; masked cells get
    the utterance mean."""
    n_mels = feats.size(1)
    for b in range(feats.size(0)):
        n = int(mel_lens[b])
        x = feats[b, :, :n]
        fill = x.mean()
        for _ in range(freq_masks):
            w = int(rng.integers(0, freq_width + 1))
            f0 = int(rng.integers(0, n_mels - w + 1))
            x[f0 : f0 + w, :] = fill
        for _ in range(time_masks):
            w = min(int(rng.integers(0, time_width + 1)), n)
            t0 = int(rng.integers(0, n - w + 1))
            x[:, t0 : t0 + w] = fill
    return feats
