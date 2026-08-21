# coding=utf-8
"""No-speech segment augmentation for Qwen3-ASR fine-tuning."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

_ASR_TEXT_TAG = "<asr_text>"
_LANG_PREFIX_RE = re.compile(r"^(language [^<]*<asr_text>)")


@dataclass
class NoSpeechAugmentConfig:
    """Online augmentation that inserts zero-padded no-speech regions."""

    enabled: bool = False
    prob: float = 0.3
    pad_min_sec: float = 0.5
    pad_max_sec: float = 3.0
    dual_max_speech_sec: float = 30.0

    @classmethod
    def from_args(cls, args) -> "NoSpeechAugmentConfig":
        return cls(
            enabled=bool(getattr(args, "nospeech_augment", 0)),
            prob=float(getattr(args, "nospeech_prob", 0.3)),
            pad_min_sec=float(getattr(args, "nospeech_pad_min_sec", 0.5)),
            pad_max_sec=float(getattr(args, "nospeech_pad_max_sec", 3.0)),
            dual_max_speech_sec=float(
                getattr(args, "nospeech_dual_max_speech_sec", 30.0)
            ),
        )


def extract_asr_text(text: str) -> str:
    """Return transcript after ``<asr_text>``."""
    value = text or ""
    if _ASR_TEXT_TAG in value:
        return value.split(_ASR_TEXT_TAG, 1)[1]
    return value


def language_prefix(text: str) -> str:
    """Return the Qwen3-ASR language prefix used for pairing."""
    match = _LANG_PREFIX_RE.match(text or "")
    return match.group(1) if match else ""


def merge_asr_targets(text1: str, text2: str) -> str:
    """Merge two Qwen3-ASR labels, concatenating asr_text bodies."""
    t1 = text1 or ""
    t2 = text2 or ""
    asr1 = extract_asr_text(t1).strip()
    asr2 = extract_asr_text(t2).strip()
    merged_asr = f"{asr1} {asr2}".strip()
    match1 = _LANG_PREFIX_RE.match(t1)
    match2 = _LANG_PREFIX_RE.match(t2)

    if asr1 and match1:
        prefix = match1.group(1)
    elif asr2 and match2:
        prefix = match2.group(1)
    elif match1:
        prefix = match1.group(1)
    elif match2:
        prefix = match2.group(1)
    else:
        return merged_asr

    return f"{prefix}{merged_asr}"


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


def try_dual_concat(
    index: int,
    wavs: Sequence[np.ndarray],
    targets: Sequence[str],
    rng: random.Random,
    cfg: NoSpeechAugmentConfig,
    sr: int,
    noise_flags: Optional[Sequence[int]] = None,
) -> Optional[Tuple[np.ndarray, str]]:
    """Concatenate wav[index] with another batch item if within duration limit."""
    if len(wavs) < 2:
        return None
    prefix = language_prefix(targets[index])
    noise_flag = int(noise_flags[index]) if noise_flags is not None else None
    candidates = [
        j
        for j in range(len(wavs))
        if j != index
        and language_prefix(targets[j]) == prefix
        and (noise_flags is None or int(noise_flags[j]) == noise_flag)
    ]
    rng.shuffle(candidates)
    wav_i = np.asarray(wavs[index], dtype=np.float32)
    dur_i = len(wav_i) / sr
    for j in candidates:
        wav_j = np.asarray(wavs[j], dtype=np.float32)
        dur_j = len(wav_j) / sr
        if dur_i + dur_j > cfg.dual_max_speech_sec:
            continue
        gap = _sample_pad_samples(cfg, sr, rng)
        merged = np.concatenate(
            [wav_i, np.zeros(gap, dtype=np.float32), wav_j], dtype=np.float32
        )
        merged_target = merge_asr_targets(targets[index], targets[j])
        return merged, merged_target
    return None


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
    noise_flags: Optional[Sequence[int]] = None,
) -> Tuple[List[np.ndarray], List[str]]:
    """Apply no-speech augmentation to a batch of waveforms and targets."""
    if not cfg.enabled:
        return wavs, targets

    out_wavs = [np.asarray(w, dtype=np.float32) for w in wavs]
    out_targets = list(targets)
    modes = ("leading", "trailing", "dual_concat")

    for i, aug in enumerate(aug_flags):
        if int(aug) != 1:
            continue
        if rng.random() >= cfg.prob:
            continue

        mode = rng.choice(modes)
        if mode == "dual_concat":
            dual = try_dual_concat(
                i,
                out_wavs,
                out_targets,
                rng,
                cfg,
                sr,
                noise_flags=noise_flags,
            )
            if dual is not None:
                out_wavs[i], out_targets[i] = dual
                continue
            mode = rng.choice(("leading", "trailing"))

        out_wavs[i] = _apply_pad_mode(out_wavs[i], mode, cfg, sr, rng)

    return out_wavs, out_targets
