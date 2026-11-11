# coding=utf-8
"""Training-side consumer for the bounded pseudo-online spool."""

from __future__ import annotations

import logging
import queue
import random
import time
from typing import Any, Dict, Iterator

from torch.utils.data import IterableDataset

from .spool import BoundedSpool

logger = logging.getLogger(__name__)


class SpoolDataset(IterableDataset):
    """Read augmented spool items, delete them, and yield training samples."""

    def __init__(
        self,
        ready_queue,
        spool: BoundedSpool,
        stats,
        stop_event,
        producer_done,
        error_queue=None,
        error_flag=None,
        sample_rate: int = 16000,
        stall_timeout_sec: float = 300.0,
        stats_log_interval_sec: float = 0.0,
        window_size: int = 256,
    ):
        self.ready_queue = ready_queue
        self.spool = spool
        self.stats = stats
        self.stop_event = stop_event
        self.producer_done = producer_done
        self.error_queue = error_queue
        self.error_flag = error_flag
        self.sample_rate = int(sample_rate)
        self.stall_timeout_sec = float(stall_timeout_sec)
        self.stats_log_interval_sec = float(stats_log_interval_sec)
        # Sort a small window by duration so consecutive DataLoader batches
        # contain similar-length audio and waste less padding on the GPU.
        self.window_size = max(1, int(window_size))

    def _raise_if_error(self) -> None:
        if self.error_flag is None or not int(self.error_flag.value):
            return
        message = "details unavailable"
        if self.error_queue is not None:
            try:
                message = self.error_queue.get(timeout=1.0)
            except queue.Empty:
                pass
        raise RuntimeError(f"Augmentation reader failed: {message}")

    def _log_stats_if_due(self, last_log_at: float, now: float) -> float:
        if (
            self.stats_log_interval_sec <= 0
            or now - last_log_at < self.stats_log_interval_sec
        ):
            return last_log_at
        logger.info(
            "augmentation stats: produced=%d consumed=%d consumed_hours=%.2f "
            "errors=%d spool=%.2fMB ready_queue=%d",
            int(self.stats.produced.value),
            int(self.stats.consumed.value),
            float(self.stats.consumed_hours.value),
            int(self.stats.errors.value),
            self.spool.bytes_used / 1e6,
            self._queue_size(),
        )
        return now

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        last_item_at = time.time()
        last_log_at = last_item_at
        window = []
        done = False
        while True:
            while len(window) < self.window_size:
                self._raise_if_error()
                if self.stop_event.is_set() and self._queue_empty():
                    self._raise_if_error()
                    done = True
                    break
                try:
                    item = self.ready_queue.get(timeout=0.5)
                except queue.Empty:
                    self._raise_if_error()
                    if self.stop_event.is_set():
                        self._raise_if_error()
                        done = True
                        break
                    if self.producer_done.is_set() and self._queue_empty():
                        done = True
                        break
                    if time.time() - last_item_at > self.stall_timeout_sec:
                        raise RuntimeError(
                            "Augmentation service stalled: no spooled samples for "
                            f"{self.stall_timeout_sec:.1f}s "
                            f"(produced={int(self.stats.produced.value)}, "
                            f"consumed={int(self.stats.consumed.value)}, "
                            f"errors={int(self.stats.errors.value)}, "
                            f"spool_bytes={self.spool.bytes_used})."
                        )
                    last_log_at = self._log_stats_if_due(last_log_at, time.time())
                    continue
                if item is None:
                    self._raise_if_error()
                    done = True
                    break
                last_item_at = time.time()
                last_log_at = self._log_stats_if_due(last_log_at, last_item_at)
                try:
                    metadata = self.spool.read_metadata(item)
                except Exception as exc:
                    self.stats.inc(self.stats.errors)
                    logger.warning(
                        "Skipping unreadable spool metadata %s: %s",
                        item.meta_path,
                        exc,
                    )
                    self.spool.release(item)
                    continue
                window.append(
                    (float(metadata.get("duration") or 0.0), item, metadata)
                )
            if not window:
                return
            window.sort(key=lambda entry: entry[0])
            # Keep some randomness: shuffle inside duration-adjacent chunks.
            chunk_size = max(1, self.window_size // 8)
            for start in range(0, len(window), chunk_size):
                chunk = window[start : start + chunk_size]
                random.shuffle(chunk)
                window[start : start + chunk_size] = chunk
            for duration, item, metadata in window:
                try:
                    audio = self.spool.read_audio(item, target_sr=self.sample_rate)
                    sample = dict(metadata)
                    sample["audio"] = audio
                    sample["degraded"] = bool(metadata.get("simulated", True))
                    sample.setdefault("aug", 1)
                    # The simulator already handled additive noise; keep the
                    # current SFT collator from adding it a second time.
                    sample["noise_aug"] = 0
                except Exception as exc:
                    self.stats.inc(self.stats.errors)
                    logger.warning(
                        "Skipping unreadable spool item %s: %s", item.audio_path, exc
                    )
                    self.spool.release(item)
                    continue
                self.spool.release(item)
                self.stats.inc(self.stats.consumed)
                if duration > 0:
                    with self.stats.consumed_hours.get_lock():
                        self.stats.consumed_hours.value += duration / 3600.0
                yield sample
            window = []
            if done:
                return

    def _queue_size(self) -> int:
        try:
            return int(self.ready_queue.qsize())
        except (NotImplementedError, OSError, ValueError):
            return -1

    def _queue_empty(self) -> bool:
        size = self._queue_size()
        return True if size < 0 else size == 0
