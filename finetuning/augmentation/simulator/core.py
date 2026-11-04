# coding=utf-8
"""Runtime helpers around the vendored mega-asr-sim primitives.

The simulator modules in this package are vendored from
https://github.com/NLPvv/mega-asr-sim (MIT, see LICENSE).
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import soundfile as sf

from .primitives import PRIMITIVES
from .scenarios import SCENARIOS, merge_chain
from .severity import resolve_params


class NoisePool:
    """LRU-backed pool of real noise files used by the ``noise`` chains."""

    def __init__(
        self,
        noise_dir: Optional[str] = None,
        sample_rate: int = 16000,
        max_cached_files: int = 64,
    ):
        self.sample_rate = int(sample_rate)
        self.max_cached_files = max(1, int(max_cached_files))
        self.files: List[Path] = []
        if noise_dir:
            root = Path(noise_dir)
            self.files = sorted(
                path
                for pattern in ("*.wav", "*.flac")
                for path in root.rglob(pattern)
                if path.is_file()
            )
        self._cache: "OrderedDict[Path, np.ndarray]" = OrderedDict()

    @property
    def available(self) -> bool:
        return bool(self.files)

    def _load(self, path: Path) -> np.ndarray:
        cached = self._cache.get(path)
        if cached is not None:
            self._cache.move_to_end(path)
            return cached
        wav, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        if wav.size == 0:
            raise ValueError(f"Noise file is empty: {path}")
        if sample_rate != self.sample_rate:
            import librosa

            wav = librosa.resample(
                wav,
                orig_sr=sample_rate,
                target_sr=self.sample_rate,
                res_type="soxr_hq",
            )
        wav = np.asarray(wav, dtype=np.float32)
        self._cache[path] = wav
        if len(self._cache) > self.max_cached_files:
            self._cache.popitem(last=False)
        return wav

    def sample(self, num_samples: int, rng: np.random.Generator) -> np.ndarray:
        if not self.files:
            return rng.standard_normal(num_samples).astype(np.float32)
        path = self.files[int(rng.integers(len(self.files)))]
        wav = self._load(path)
        if len(wav) < num_samples:
            reps = int(np.ceil(num_samples / max(len(wav), 1)))
            wav = np.tile(wav, reps)
        start = (
            int(rng.integers(0, len(wav) - num_samples + 1))
            if len(wav) > num_samples
            else 0
        )
        return wav[start : start + num_samples].copy()


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    wav, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if wav.size == 0:
        raise ValueError(f"Audio is empty: {path}")
    wav = wav.mean(axis=1)
    if sample_rate != target_sr:
        import librosa

        wav = librosa.resample(
            wav, orig_sr=sample_rate, target_sr=target_sr, res_type="soxr_hq"
        )
    return np.asarray(wav, dtype=np.float32)


def apply_scenario(
    wav: np.ndarray,
    sr: int,
    scenario: str,
    severity: float,
    rng: np.random.Generator,
    noise_pool: NoisePool,
    skip_noise: bool = False,
) -> np.ndarray:
    """Apply one mega-asr-sim scenario with a shared global severity."""
    if scenario not in SCENARIOS:
        raise KeyError(f"Unknown scenario: {scenario!r}")
    out = np.asarray(wav, dtype=np.float32)
    for entry in merge_chain(SCENARIOS[scenario]):
        primitive = entry["prim"]
        if skip_noise and primitive == "add_noise":
            continue
        kwargs = resolve_params(entry["params"], float(severity))
        if primitive == "add_noise":
            kwargs["noise_pool"] = noise_pool
        out = PRIMITIVES[primitive](out, sr, rng, **kwargs)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out = out / peak * 0.99
    return out.astype(np.float32, copy=False)


def available_scenarios() -> List[str]:
    return sorted(SCENARIOS)


def resolve_scenario_weights(
    scenarios: Optional[Iterable[str]],
    weights: Optional[Dict[str, float]],
) -> Dict[str, float]:
    if scenarios is None and weights is None:
        selected = list(SCENARIOS)
        raw = {name: 1.0 for name in selected}
    elif weights is not None:
        raw = {str(name): float(value) for name, value in weights.items()}
        unknown = set(raw) - set(SCENARIOS)
        if unknown:
            raise ValueError(f"Unknown scenario weights: {sorted(unknown)}")
    else:
        selected = [str(name) for name in scenarios]
        unknown = set(selected) - set(SCENARIOS)
        if unknown:
            raise ValueError(f"Unknown scenarios: {sorted(unknown)}")
        raw = {name: 1.0 for name in selected}
    if not raw:
        raise ValueError("At least one scenario must be selected.")
    if any(value <= 0 for value in raw.values()):
        raise ValueError("Scenario weights must be positive.")
    total = sum(raw.values())
    return {name: value / total for name, value in raw.items()}
