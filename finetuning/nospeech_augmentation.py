# coding=utf-8
"""No-speech segment augmentation for Qwen3-ASR fine-tuning."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class NoSpeechAugmentConfig:
    """Online augmentation that inserts zero-padded no-speech regions."""

    enabled: bool = False
    prob: float = 0.3
    pad_min_sec: float = 0.5
    pad_max_sec: float = 3.0

    @classmethod
    def from_args(cls, args) -> "NoSpeechAugmentConfig":
        return cls(
            enabled=bool(getattr(args, "nospeech_augment", 0)),
            prob=float(getattr(args, "nospeech_prob", 0.3)),
            pad_min_sec=float(getattr(args, "nospeech_pad_min_sec", 0.5)),
            pad_max_sec=float(getattr(args, "nospeech_pad_max_sec", 3.0)),
        )


def zero_pad(
    wav: np.ndarray, lead_samples: int = 0, trail_samples: int = 0
) -> np.ndarray:
    """Prepend and/or append zero samples."""
    wav = np.asarray(wav, dtype=np.float32)
    if lead_samples <= 0 and trail_samples <= 0:
        return wav
    parts = []
    if lead_samples > 0:
        parts.append(np.zeros(lead_samples, dtype=np.float32))
    parts.append(wav)
    if trail_samples > 0:
        parts.append(np.zeros(trail_samples, dtype=np.float32))
    return np.concatenate(parts)


def _sample_pad_samples(cfg: NoSpeechAugmentConfig, sr: int, rng: random.Random) -> int:
    lo = min(cfg.pad_min_sec, cfg.pad_max_sec)
    hi = max(cfg.pad_min_sec, cfg.pad_max_sec)
    duration = rng.uniform(lo, hi)
    return max(1, int(round(duration * sr)))


def _apply_pad_mode(
    wav: np.ndarray,
    mode: str,
    cfg: NoSpeechAugmentConfig,
    sr: int,
    rng: random.Random,
) -> np.ndarray:
    pad = _sample_pad_samples(cfg, sr, rng)
    if mode == "leading":
        return zero_pad(wav, lead_samples=pad)
    return zero_pad(wav, trail_samples=pad)


def apply_nospeech_augment(
    wavs: List[np.ndarray],
    targets: List[str],
    aug_flags: Sequence[int],
    rng: random.Random,
    cfg: NoSpeechAugmentConfig,
    sr: int,
) -> Tuple[List[np.ndarray], List[str]]:
    """Apply no-speech augmentation to a batch of waveforms and targets."""
    if not cfg.enabled:
        return wavs, targets

    out_wavs = [np.asarray(w, dtype=np.float32) for w in wavs]
    out_targets = list(targets)
    modes = ("leading", "trailing")

    for i, aug in enumerate(aug_flags):
        if int(aug) != 1:
            continue
        if rng.random() >= cfg.prob:
            continue

        out_wavs[i] = _apply_pad_mode(out_wavs[i], rng.choice(modes), cfg, sr, rng)

    return out_wavs, out_targets
