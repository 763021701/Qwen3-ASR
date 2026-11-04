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
        sample_rate: int = 16000,
        stall_timeout_sec: float = 300.0,
    ):
        self.ready_queue = ready_queue
        self.spool = spool
        self.stats = stats
        self.stop_event = stop_event
        self.producer_done = producer_done
        self.sample_rate = int(sample_rate)
        self.stall_timeout_sec = float(stall_timeout_sec)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        last_item_at = time.time()
        while True:
            if self.stop_event.is_set() and self._queue_empty():
                return
            try:
                item = self.ready_queue.get(timeout=0.5)
            except queue.Empty:
                if self.stop_event.is_set():
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
                continue
            if item is None:
                return
            last_item_at = time.time()
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

    def _queue_empty(self) -> bool:
        try:
            return int(self.ready_queue.qsize()) == 0
        except (NotImplementedError, OSError, ValueError):
            return True
