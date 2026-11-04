# coding=utf-8
"""Process-based pseudo-online augmentation service with bounded spooling."""

from __future__ import annotations

import logging
import multiprocessing
import queue
import time
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

from .simulator import (
    NoisePool,
    apply_scenario,
    load_audio,
    resolve_scenario_weights,
    sample_severity,
)
from .spool import BoundedSpool

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for pseudo-online mega-asr-sim augmentation."""

    spool_dir: str
    scenarios: Optional[tuple] = None
    scenario_weights: Optional[Dict[str, float]] = None
    severity_mapping: str = "linear"
    severity_min: float = 0.0
    severity_max: float = 1.0
    clean_prob: float = 0.0
    noise_dir: Optional[str] = None
    noise_cache_files: int = 64
    sample_rate: int = 16000
    audio_format: str = "wav"
    max_spool_bytes: int = 10 * 1024**3
    num_simulators: int = 4
    clean_queue_size: int = 64
    ready_queue_size: int = 256
    stall_timeout_sec: float = 300.0
    spool_ttl_sec: float = 3600.0
    delete_spool_on_close: bool = True
    seed: int = 42
    run_id: Optional[str] = None
    start_method: str = "spawn"
    sim_version: str = "mega-asr-sim-v1"

    def __post_init__(self) -> None:
        if not self.spool_dir:
            raise ValueError("spool_dir is required.")
        if self.max_spool_bytes <= 0:
            raise ValueError("max_spool_bytes must be positive.")
        if self.num_simulators < 1:
            raise ValueError("num_simulators must be >= 1.")
        if self.clean_queue_size < 1 or self.ready_queue_size < 1:
            raise ValueError("queue sizes must be >= 1.")
        if not 0.0 <= self.clean_prob <= 1.0:
            raise ValueError("clean_prob must be in [0, 1].")
        if not 0.0 <= self.severity_min <= self.severity_max <= 1.0:
            raise ValueError("severity_min/max must satisfy 0 <= min <= max <= 1.")
        if self.audio_format.lower().lstrip(".") not in {"wav", "flac"}:
            raise ValueError("audio_format must be 'wav' or 'flac'.")
        # Validate the scenario selection early.
        resolve_scenario_weights(self.scenarios, self.scenario_weights)

    @classmethod
    def from_spec(cls, spec: Any, sample_rate: int = 16000) -> "SimulationConfig":
        return cls(
            spool_dir=spec.spool_dir or "",
            scenarios=spec.scenarios,
            scenario_weights=spec.scenario_weights,
            severity_mapping=spec.severity_mapping,
            severity_min=spec.severity_min,
            severity_max=spec.severity_max,
            clean_prob=spec.clean_prob,
            noise_dir=spec.noise_dir,
            noise_cache_files=spec.noise_cache_files,
            sample_rate=int(spec.sample_rate or sample_rate),
            audio_format=spec.audio_format,
            max_spool_bytes=spec.max_spool_bytes,
            num_simulators=spec.num_simulators,
            clean_queue_size=spec.clean_queue_size,
            ready_queue_size=spec.ready_queue_size,
            stall_timeout_sec=spec.stall_timeout_sec,
            spool_ttl_sec=spec.spool_ttl_sec,
            delete_spool_on_close=spec.delete_spool_on_close,
            seed=spec.seed,
            run_id=spec.run_id,
            start_method=spec.start_method,
            sim_version=spec.sim_version,
        )


class _Stats:
    def __init__(self, ctx):
        self.produced = ctx.Value("q", 0)
        self.consumed = ctx.Value("q", 0)
        self.errors = ctx.Value("q", 0)
        self.started_at = ctx.Value("d", time.time())

    def inc(self, counter) -> None:
        with counter.get_lock():
            counter.value = int(counter.value) + 1

    def snapshot(self, service) -> Dict[str, Any]:
        elapsed = max(1e-6, time.time() - float(self.started_at.value))
        produced = int(self.produced.value)
        consumed = int(self.consumed.value)
        return {
            "produced": produced,
            "consumed": consumed,
            "errors": int(self.errors.value),
            "produced_per_sec": produced / elapsed,
            "consumed_per_sec": consumed / elapsed,
            "spool_bytes": service.spool.bytes_used,
            "clean_queue": _qsize(service.clean_queue),
            "ready_queue": _qsize(service.ready_queue),
        }


def _qsize(q) -> int:
    try:
        return int(q.qsize())
    except (NotImplementedError, OSError, ValueError):
        return -1


def _sample_seed(sample: Dict[str, Any], default: int) -> int:
    value = sample.get("sample_seed")
    if value is not None:
        try:
            return int(value) & ((1 << 63) - 1)
        except (TypeError, ValueError):
            pass
    key = f"{sample.get('source', '')}:{sample.get('key', '')}"
    return (zlib.crc32(key.encode("utf-8")) + int(default)) & ((1 << 63) - 1)


def _clean_audio(sample: Dict[str, Any], sample_rate: int) -> np.ndarray:
    audio = sample.get("audio")
    if isinstance(audio, np.ndarray):
        wav = np.asarray(audio, dtype=np.float32)
        if wav.ndim != 1:
            raise ValueError(f"Expected mono audio, got {wav.shape}.")
        return wav
    if isinstance(audio, str):
        return load_audio(audio, target_sr=sample_rate)
    raise TypeError(f"Unsupported audio field type: {type(audio).__name__}")


def augment_clean_sample(
    sample: Dict[str, Any],
    config: SimulationConfig,
    noise_pool: NoisePool,
    scenario_weights: Dict[str, float],
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Apply one deterministic mega-asr-sim scenario to a clean sample."""
    seed = _sample_seed(sample, config.seed)
    rng = np.random.default_rng(seed)
    wav = _clean_audio(sample, config.sample_rate)
    metadata = {
        key: value
        for key, value in sample.items()
        if key != "audio" and isinstance(value, (str, int, float, bool, type(None)))
    }
    if rng.random() < config.clean_prob:
        scenario = "clean"
        severity = 0.0
        simulated = False
    else:
        names = list(scenario_weights)
        probabilities = np.asarray([scenario_weights[name] for name in names], dtype=np.float64)
        scenario = str(rng.choice(names, p=probabilities))
        severity = config.severity_min + (
            config.severity_max - config.severity_min
        ) * sample_severity(rng, config.severity_mapping)
        simulated = True
        wav = apply_scenario(
            wav,
            config.sample_rate,
            scenario,
            severity,
            rng,
            noise_pool,
            skip_noise=not bool(int(sample.get("noise_aug", 1))),
        )
    metadata.update(
        {
            "scenario": scenario,
            "severity": float(severity),
            "simulated": bool(simulated),
            "sim_version": config.sim_version,
            "sample_seed": int(seed),
        }
    )
    return np.asarray(wav, dtype=np.float32), metadata


def simulator_worker(
    clean_queue,
    ready_queue,
    spool: BoundedSpool,
    config: SimulationConfig,
    stop_event,
    stats: _Stats,
    active_simulators,
    producer_done,
) -> None:
    try:
        noise_pool = NoisePool(
            config.noise_dir,
            sample_rate=config.sample_rate,
            max_cached_files=config.noise_cache_files,
        )
        scenario_weights = resolve_scenario_weights(
            config.scenarios, config.scenario_weights
        )
        while not stop_event.is_set():
            try:
                sample = clean_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if sample is None:
                break
            try:
                audio, metadata = augment_clean_sample(
                    sample, config, noise_pool, scenario_weights
                )
                item = spool.write(
                    audio, config.sample_rate, metadata, config.audio_format
                )
                while not stop_event.is_set():
                    try:
                        ready_queue.put(item, timeout=0.5)
                        break
                    except queue.Full:
                        continue
                else:
                    spool.release(item)
                    continue
                stats.inc(stats.produced)
            except Exception as exc:
                stats.inc(stats.errors)
                logger.exception("Simulator failed: %s", exc)
    finally:
        with active_simulators.get_lock():
            active_simulators.value = max(0, int(active_simulators.value) - 1)
            last = int(active_simulators.value) == 0
        if last:
            try:
                ready_queue.put(None, timeout=1.0)
            except queue.Full:
                pass
            producer_done.set()


def reader_worker(
    clean_iterable: Iterable[Dict[str, Any]],
    clean_queue,
    stop_event,
    num_simulators: int,
) -> None:
    try:
        for sample in clean_iterable:
            if stop_event.is_set():
                break
            while not stop_event.is_set():
                try:
                    clean_queue.put(sample, timeout=0.5)
                    break
                except queue.Full:
                    continue
    except BaseException as exc:
        logger.exception("Clean-data reader failed: %s", exc)
        stop_event.set()
    finally:
        for _ in range(num_simulators):
            try:
                clean_queue.put(None, timeout=0.1)
            except queue.Full:
                break


class SimulationService:
    """Reader process -> simulator processes -> bounded spool -> training dataset.

    ``clean_iterable`` is passed to a ``spawn`` process, so it must be picklable
    (e.g. a data-pipeline iterable, tuple, or list; not a generator/lambda).
    """

    def __init__(
        self,
        clean_iterable: Iterable[Dict[str, Any]],
        config: SimulationConfig,
    ):
        self.clean_iterable = clean_iterable
        self.config = config
        self.ctx = multiprocessing.get_context(config.start_method)
        self.stop_event = self.ctx.Event()
        self.producer_done = self.ctx.Event()
        self.active_simulators = self.ctx.Value(
            "i", int(config.num_simulators)
        )
        self.clean_queue = self.ctx.Queue(maxsize=config.clean_queue_size)
        self.ready_queue = self.ctx.Queue(maxsize=config.ready_queue_size)
        self.stats = _Stats(self.ctx)
        run_id = config.run_id or (
            time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        )
        spool_root = Path(config.spool_dir) / run_id
        self._cleanup_stale_runs(Path(config.spool_dir))
        self.spool = BoundedSpool(
            str(spool_root),
            max_bytes=config.max_spool_bytes,
            context=self.ctx,
            stop_event=self.stop_event,
        )
        self.processes = []
        self.reader_process: Optional[Any] = None
        self._started = False

    def _cleanup_stale_runs(self, base: Path) -> None:
        ttl = float(self.config.spool_ttl_sec)
        if ttl <= 0 or not base.is_dir():
            return
        cutoff = time.time() - ttl
        for path in base.iterdir():
            if not path.is_dir():
                continue
            try:
                if path.stat().st_mtime < cutoff:
                    for child in path.iterdir():
                        if child.is_file():
                            child.unlink()
                    path.rmdir()
            except OSError:
                continue

    def start(self) -> None:
        if self._started:
            return
        self.stats.started_at.value = time.time()
        try:
            for _ in range(self.config.num_simulators):
                process = self.ctx.Process(
                    target=simulator_worker,
                    args=(
                        self.clean_queue,
                        self.ready_queue,
                        self.spool,
                        self.config,
                        self.stop_event,
                        self.stats,
                        self.active_simulators,
                        self.producer_done,
                    ),
                    daemon=True,
                )
                process.start()
                self.processes.append(process)
            self.reader_process = self.ctx.Process(
                target=reader_worker,
                args=(
                    self.clean_iterable,
                    self.clean_queue,
                    self.stop_event,
                    self.config.num_simulators,
                ),
                daemon=True,
            )
            self.reader_process.start()
        except Exception:
            self.stop_event.set()
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=2.0)
            if self.reader_process is not None and self.reader_process.is_alive():
                self.reader_process.terminate()
                self.reader_process.join(timeout=2.0)
            self.spool.close()
            raise
        self._started = True

    @property
    def dataset(self):
        from .consumer import SpoolDataset

        return SpoolDataset(
            ready_queue=self.ready_queue,
            spool=self.spool,
            stats=self.stats,
            stop_event=self.stop_event,
            producer_done=self.producer_done,
            sample_rate=self.config.sample_rate,
            stall_timeout_sec=self.config.stall_timeout_sec,
        )

    def snapshot(self) -> Dict[str, Any]:
        return self.stats.snapshot(self)

    def close(self) -> None:
        if not self._started and not self.processes and self.reader_process is None:
            if self.config.delete_spool_on_close:
                self.spool.close()
            return
        self.stop_event.set()
        try:
            self.ready_queue.put(None, timeout=0.2)
        except queue.Full:
            pass
        for _ in range(self.config.num_simulators):
            try:
                self.clean_queue.put(None, timeout=0.1)
            except queue.Full:
                break
        if self.reader_process is not None:
            self.reader_process.join(timeout=5.0)
            if self.reader_process.is_alive():
                self.reader_process.terminate()
                self.reader_process.join(timeout=2.0)
        for process in self.processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
        if self.config.delete_spool_on_close:
            self.spool.close()
        for q in (self.clean_queue, self.ready_queue):
            try:
                q.close()
                q.cancel_join_thread()
            except Exception:
                pass
        self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
