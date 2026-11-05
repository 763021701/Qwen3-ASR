# coding=utf-8
"""Runtime helpers around the vendored mega-asr-sim primitives.

The simulator modules in this package are vendored from
https://github.com/NLPvv/mega-asr-sim (MIT, see LICENSE).
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import soundfile as sf

from .primitives import PRIMITIVES
from .scenarios import SCENARIOS, merge_chain
from .severity import resolve_params

logger = logging.getLogger(__name__)

_AUDIO_SUFFIXES = {".wav", ".flac"}
_REVERB_MODES = {"synthetic", "real", "mixed"}


def _walk_files(root: str) -> Iterable[Path]:
    """Yield files recursively, following directory symlinks without cycles."""
    seen = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        resolved = Path(dirpath).resolve()
        if resolved in seen:
            dirnames[:] = []
            continue
        seen.add(resolved)
        dirnames.sort()
        for name in sorted(filenames):
            yield Path(dirpath) / name


def _iter_audio_files(root: str) -> List[Path]:
    return [
        path
        for path in _walk_files(root)
        if path.suffix.lower() in _AUDIO_SUFFIXES and path.is_file()
    ]


@dataclass(frozen=True)
class RIRInfo:
    path: Path
    source: str
    category: Optional[str] = None
    room_id: Optional[str] = None


def _source_from_text(text: str) -> str:
    return "simulated" if "simulated" in text.lower() else "real"


def _category_from_text(text: str) -> Optional[str]:
    text = text.lower()
    if "smallroom" in text or "small-room" in text:
        return "small"
    if "mediumroom" in text or "medium-room" in text:
        return "medium"
    if "largeroom" in text or "large-room" in text:
        return "large"
    return None


def _resolve_rir_path(token: str, list_path: Path, root: Path) -> Optional[Path]:
    if os.path.isabs(token):
        candidates = [token]
    else:
        list_dir = os.path.dirname(str(list_path))
        root_str = str(root)
        candidates = [os.path.join(root_str, token), os.path.join(list_dir, token)]
        parts = Path(token).parts
        for index in range(1, len(parts)):
            suffix = os.path.join(*parts[index:])
            candidates.append(os.path.join(list_dir, suffix))
            candidates.append(os.path.join(root_str, suffix))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return Path(candidate)
    return None


def _parse_rir_list(list_path: Path, root: Path) -> Iterable[RIRInfo]:
    for line in list_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        tokens = line.split()
        if not tokens or tokens[0].startswith("#"):
            continue
        room_id = None
        if "--room-id" in tokens:
            index = tokens.index("--room-id")
            if index + 1 < len(tokens):
                room_id = tokens[index + 1]
        path = _resolve_rir_path(tokens[-1], list_path, root)
        if path is None:
            continue
        text = f"{room_id or ''} {path}"
        yield RIRInfo(
            path=path,
            source=_source_from_text(text),
            category=_category_from_text(text),
            room_id=room_id,
        )


def _prepare_real_rir(rir: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
    """Align the direct path to t=0 and remove it for dry/wet mixing."""
    rir = np.asarray(rir, dtype=np.float32)
    if rir.ndim != 1 or rir.size == 0 or not np.isfinite(rir).all():
        return None
    peak_index = int(np.argmax(np.abs(rir)))
    if peak_index > 0:
        rir = np.concatenate([rir[peak_index:], np.zeros(peak_index, dtype=np.float32)])
    fade_len = max(1, int(0.001 * sample_rate))
    fade = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    rir = rir.copy()
    rir[:fade_len] *= fade[: len(rir[:fade_len])]
    magnitude = np.abs(rir)
    peak = float(magnitude.max()) if magnitude.size else 0.0
    if peak <= 1e-8:
        return None
    active = np.flatnonzero(magnitude > peak * 1e-3)
    if active.size:
        end = min(len(rir), int(active[-1]) + max(1, int(0.01 * sample_rate)))
        rir = rir[:end]
    energy = float(np.sqrt(np.sum(rir.astype(np.float64) ** 2)))
    if energy <= 1e-8:
        return None
    return (rir / energy).astype(np.float32)


class RIRPool:
    """LRU-backed pool of real/simulated room impulse responses."""

    def __init__(
        self,
        rir_dir: Optional[str] = None,
        sample_rate: int = 16000,
        max_cached_files: int = 128,
        real_prob: float = 1.0,
    ):
        self.sample_rate = int(sample_rate)
        self.max_cached_files = max(1, int(max_cached_files))
        self.real_prob = float(real_prob)
        if not 0.0 <= self.real_prob <= 1.0:
            raise ValueError("real_prob must be in [0, 1].")
        self.files: List[RIRInfo] = self._index_rirs(rir_dir) if rir_dir else []
        if rir_dir and not self.files:
            logger.warning("No RIR files found under %s.", rir_dir)
        self._by_source: Dict[str, List[RIRInfo]] = {}
        self._by_category: Dict[str, List[RIRInfo]] = {}
        self._by_source_category: Dict[tuple, List[RIRInfo]] = {}
        for info in self.files:
            self._by_source.setdefault(info.source, []).append(info)
            if info.category is not None:
                self._by_category.setdefault(info.category, []).append(info)
                self._by_source_category.setdefault(
                    (info.source, info.category), []
                ).append(info)
        self._cache: "OrderedDict[Path, np.ndarray]" = OrderedDict()

    @staticmethod
    def _index_rirs(rir_dir: str) -> List[RIRInfo]:
        root = Path(rir_dir)
        entries: List[RIRInfo] = []
        seen = set()
        lists = [path for path in _walk_files(str(root)) if path.name == "rir_list"]
        for list_path in lists:
            for info in _parse_rir_list(list_path, root):
                if info.path not in seen:
                    seen.add(info.path)
                    entries.append(info)
        if not entries:
            for path in _iter_audio_files(str(root)):
                if "pointsource_noises" in path.parts:
                    continue
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                entries.append(
                    RIRInfo(
                        path=resolved,
                        source=_source_from_text(str(path)),
                        category=_category_from_text(str(path)),
                    )
                )
        entries.sort(key=lambda info: str(info.path))
        return entries

    @property
    def available(self) -> bool:
        return bool(self.files)

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _category_for_room_size(room_size: Optional[float]) -> Optional[str]:
        if room_size is None:
            return None
        value = float(room_size)
        if value < 1.0 / 3.0:
            return "small"
        if value < 2.0 / 3.0:
            return "medium"
        return "large"

    def _select(self, rng: np.random.Generator, room_size: Optional[float]) -> List[RIRInfo]:
        sources = sorted(self._by_source)
        if len(sources) == 1:
            source = sources[0]
        elif "real" in sources and "simulated" in sources:
            source = "real" if rng.random() < self.real_prob else "simulated"
        else:
            source = sources[0]
        category = self._category_for_room_size(room_size)

        candidates = self._by_source_category.get((source, category))
        if candidates:
            return candidates
        candidates = self._by_source.get(source)
        if candidates:
            return candidates
        candidates = self._by_category.get(category)
        return candidates or self.files

    def _load(self, path: Path) -> np.ndarray:
        cached = self._cache.get(path)
        if cached is not None:
            self._cache.move_to_end(path)
            return cached
        wav, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        if wav.size == 0:
            raise ValueError(f"RIR file is empty: {path}")
        if sample_rate != self.sample_rate:
            import librosa

            wav = np.stack(
                [
                    librosa.resample(
                        wav[:, channel],
                        orig_sr=sample_rate,
                        target_sr=self.sample_rate,
                        res_type="soxr_hq",
                    )
                    for channel in range(wav.shape[1])
                ],
                axis=1,
            )
        wav = np.asarray(wav, dtype=np.float32)
        self._cache[path] = wav
        if len(self._cache) > self.max_cached_files:
            self._cache.popitem(last=False)
        return wav

    def sample(
        self, rng: np.random.Generator, room_size: Optional[float] = None
    ) -> Optional[np.ndarray]:
        if not self.files:
            return None
        candidates = self._select(rng, room_size)
        if not candidates:
            return None
        info = candidates[int(rng.integers(len(candidates)))]
        rirs = self._load(info.path)
        channel = int(rng.integers(rirs.shape[1]))
        return _prepare_real_rir(rirs[:, channel], self.sample_rate)


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
        self.files: List[Path] = _iter_audio_files(noise_dir) if noise_dir else []
        if noise_dir and not self.files:
            logger.warning(
                "No noise files found under %s; noise scenarios will use white noise.",
                noise_dir,
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
    rir_pool: Optional[RIRPool] = None,
    reverb_mode: str = "synthetic",
    rir_mix_prob: float = 1.0,
) -> np.ndarray:
    """Apply one mega-asr-sim scenario with a shared global severity."""
    if scenario not in SCENARIOS:
        raise KeyError(f"Unknown scenario: {scenario!r}")
    if reverb_mode not in _REVERB_MODES:
        raise ValueError(f"reverb_mode must be one of {sorted(_REVERB_MODES)}.")
    out = np.asarray(wav, dtype=np.float32)
    for entry in merge_chain(SCENARIOS[scenario]):
        primitive = entry["prim"]
        if skip_noise and primitive == "add_noise":
            continue
        kwargs = resolve_params(entry["params"], float(severity))
        if primitive == "add_noise":
            kwargs["noise_pool"] = noise_pool
        elif primitive == "add_reverb":
            use_real = False
            if rir_pool is not None and reverb_mode != "synthetic":
                if reverb_mode == "real" or rir_mix_prob >= 1.0:
                    use_real = True
                elif rir_mix_prob > 0.0:
                    use_real = rng.random() < rir_mix_prob
            kwargs["rir_pool"] = rir_pool if use_real else None
            kwargs["use_real_rir"] = use_real
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
