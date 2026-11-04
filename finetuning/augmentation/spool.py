# coding=utf-8
"""Bounded, atomic, cross-process spool for streamed augmented audio."""

from __future__ import annotations

import json
import multiprocessing
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class SpoolItem:
    audio_path: str
    meta_path: str
    size_bytes: int
    token: str


class BoundedSpool:
    """Disk spool with a hard byte cap and cross-process backpressure.

    Producers reserve bytes before writing and block when the cap is reached.
    Consumers call :meth:`release` after loading an item, which frees space and
    wakes blocked producers.
    """

    def __init__(
        self,
        root: str,
        max_bytes: int,
        *,
        context: Optional[Any] = None,
        stop_event: Optional[Any] = None,
    ):
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive.")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)
        self._ctx = context or multiprocessing.get_context("spawn")
        self._bytes = self._ctx.Value("q", 0)
        self._condition = self._ctx.Condition()
        self._stop_event = stop_event or self._ctx.Event()

    @property
    def bytes_used(self) -> int:
        with self._condition:
            return int(self._bytes.value)

    def _reserve(self, required: int) -> None:
        if required > self.max_bytes:
            raise ValueError(
                f"Spool item requires {required} bytes but max_bytes={self.max_bytes}."
            )
        with self._condition:
            while int(self._bytes.value) + required > self.max_bytes:
                if self._stop_event.is_set():
                    raise RuntimeError("Spool is closing.")
                self._condition.wait(timeout=0.5)
            self._bytes.value = int(self._bytes.value) + required

    def _adjust(self, delta: int) -> None:
        with self._condition:
            value = max(0, int(self._bytes.value) + int(delta))
            self._bytes.value = value
            self._condition.notify_all()

    def write(
        self,
        audio: np.ndarray,
        sample_rate: int,
        metadata: Dict[str, Any],
        audio_format: str = "wav",
    ) -> SpoolItem:
        audio_format = audio_format.lower().lstrip(".")
        if audio_format not in {"wav", "flac"}:
            raise ValueError("audio_format must be 'wav' or 'flac'.")
        wav = np.asarray(audio, dtype=np.float32)
        if wav.ndim != 1:
            raise ValueError(f"Expected mono audio, got shape {wav.shape}.")
        wav = np.clip(wav, -1.0, 1.0)
        token = uuid.uuid4().hex
        audio_path = self.root / f"{token}.{audio_format}"
        meta_path = self.root / f"{token}.json"
        tmp_audio = self.root / f"{token}.{audio_format}.tmp"
        tmp_meta = self.root / f"{token}.json.tmp"
        reserved = int(wav.size * 2 + 8192)
        self._reserve(reserved)
        try:
            sf.write(
                str(tmp_audio),
                wav,
                int(sample_rate),
                format="WAV" if audio_format == "wav" else "FLAC",
                subtype="PCM_16",
            )
            payload = dict(metadata)
            payload["audio"] = str(audio_path)
            with tmp_meta.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
            os.replace(tmp_audio, audio_path)
            os.replace(tmp_meta, meta_path)
            actual = audio_path.stat().st_size + meta_path.stat().st_size
            self._adjust(actual - reserved)
            return SpoolItem(
                audio_path=str(audio_path),
                meta_path=str(meta_path),
                size_bytes=int(actual),
                token=token,
            )
        except Exception:
            for path in (tmp_audio, tmp_meta, audio_path, meta_path):
                try:
                    path.unlink()
                except OSError:
                    pass
            self._adjust(-reserved)
            raise

    def read_metadata(self, item: SpoolItem) -> Dict[str, Any]:
        with open(item.meta_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def read_audio(self, item: SpoolItem, target_sr: int = 16000) -> np.ndarray:
        wav, sample_rate = sf.read(
            item.audio_path, dtype="float32", always_2d=True
        )
        if wav.size == 0:
            raise ValueError(f"Spool audio is empty: {item.audio_path}")
        wav = wav.mean(axis=1)
        if sample_rate != target_sr:
            import librosa

            wav = librosa.resample(
                wav,
                orig_sr=sample_rate,
                target_sr=target_sr,
                res_type="soxr_hq",
            )
        return np.asarray(wav, dtype=np.float32)

    def release(self, item: SpoolItem) -> None:
        for path in (item.audio_path, item.meta_path):
            try:
                Path(path).unlink()
            except OSError:
                pass
        self._adjust(-int(item.size_bytes))

    def cleanup_stale(self, ttl_sec: float) -> int:
        if ttl_sec <= 0:
            return 0
        cutoff = time.time() - float(ttl_sec)
        removed = 0
        for path in self.root.iterdir():
            if not path.is_file():
                continue
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError:
                continue
        with self._condition:
            self._bytes.value = sum(
                path.stat().st_size
                for path in self.root.iterdir()
                if path.is_file()
            )
            self._condition.notify_all()
        return removed

    def clear(self) -> None:
        for path in self.root.iterdir():
            if path.is_file():
                try:
                    path.unlink()
                except OSError:
                    continue
        with self._condition:
            self._bytes.value = 0
            self._condition.notify_all()

    def close(self) -> None:
        self.clear()
        try:
            self.root.rmdir()
        except OSError:
            pass
