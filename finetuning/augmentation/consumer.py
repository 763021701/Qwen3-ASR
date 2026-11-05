# coding=utf-8
"""Training-side consumer for the bounded pseudo-online spool."""

from __future__ import annotations

import logging
import queue
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
            "augmentation stats: produced=%d consumed=%d errors=%d "
            "spool=%.2fMB ready_queue=%d",
            int(self.stats.produced.value),
            int(self.stats.consumed.value),
            int(self.stats.errors.value),
            self.spool.bytes_used / 1e6,
            self._queue_size(),
        )
        return now

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        last_item_at = time.time()
        last_log_at = last_item_at
        while True:
            self._raise_if_error()
            if self.stop_event.is_set() and self._queue_empty():
                self._raise_if_error()
                return
            try:
                item = self.ready_queue.get(timeout=0.5)
            except queue.Empty:
                self._raise_if_error()
                if self.stop_event.is_set():
                    self._raise_if_error()
                    return
                if self.producer_done.is_set() and self._queue_empty():
                    return
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
                return
            last_item_at = time.time()
            last_log_at = self._log_stats_if_due(last_log_at, last_item_at)
            try:
                audio = self.spool.read_audio(item, target_sr=self.sample_rate)
                metadata = self.spool.read_metadata(item)
                sample = dict(metadata)
                sample["audio"] = audio
                sample["degraded"] = bool(metadata.get("simulated", True))
                sample.setdefault("aug", 1)
                # The simulator already handled additive noise; keep the
                # current SFT collator from adding it a second time.
                sample["noise_aug"] = 0
            except Exception as exc:
                self.stats.inc(self.stats.errors)
                logger.warning("Skipping unreadable spool item %s: %s", item.audio_path, exc)
                self.spool.release(item)
                continue
            self.spool.release(item)
            self.stats.inc(self.stats.consumed)
            yield sample

    def _queue_size(self) -> int:
        try:
            return int(self.ready_queue.qsize())
        except (NotImplementedError, OSError, ValueError):
            return -1

    def _queue_empty(self) -> bool:
        size = self._queue_size()
        return True if size < 0 else size == 0
