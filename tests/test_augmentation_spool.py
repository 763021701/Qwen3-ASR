# coding=utf-8
"""Tests for the bounded augmentation spool."""

import multiprocessing
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from finetuning.augmentation.spool import BoundedSpool

SR = 16000


def _spool(root: Path, max_bytes: int) -> BoundedSpool:
    return BoundedSpool(
        str(root / "spool"),
        max_bytes=max_bytes,
        context=multiprocessing.get_context("spawn"),
    )


class BoundedSpoolTest(unittest.TestCase):
    def test_write_read_release(self):
        with tempfile.TemporaryDirectory() as td:
            spool = _spool(Path(td), 1_000_000)
            item = spool.write(
                np.zeros(1600, dtype=np.float32),
                SR,
                {"target": "language Chinese<asr_text>你好", "key": "k1"},
                audio_format="wav",
            )
            self.assertTrue(Path(item.audio_path).is_file())
            self.assertTrue(Path(item.meta_path).is_file())
            self.assertGreater(spool.bytes_used, 0)
            metadata = spool.read_metadata(item)
            self.assertEqual(metadata["key"], "k1")
            self.assertEqual(metadata["audio"], item.audio_path)
            audio = spool.read_audio(item, target_sr=SR)
            self.assertEqual(audio.shape, (1600,))
            spool.release(item)
            self.assertEqual(spool.bytes_used, 0)
            self.assertFalse(Path(item.audio_path).exists())
            self.assertFalse(Path(item.meta_path).exists())
            spool.close()

    def test_blocks_when_capacity_is_full(self):
        with tempfile.TemporaryDirectory() as td:
            # One reserved item is ~11KB (audio + safety margin), so 12KB fits
            # one writer but blocks a second until the first is released.
            spool = _spool(Path(td), 12_000)
            first = spool.write(
                np.zeros(1600, dtype=np.float32),
                SR,
                {"key": "first"},
                audio_format="wav",
            )
            completed = []

            def _second_write():
                completed.append(
                    spool.write(
                        np.zeros(1600, dtype=np.float32),
                        SR,
                        {"key": "second"},
                        audio_format="wav",
                    )
                )

            thread = threading.Thread(target=_second_write)
            thread.start()
            time.sleep(0.3)
            self.assertTrue(thread.is_alive(), "second write should block on the byte cap")
            spool.release(first)
            thread.join(timeout=3.0)
            self.assertFalse(thread.is_alive())
            self.assertEqual(len(completed), 1)
            spool.release(completed[0])
            self.assertEqual(spool.bytes_used, 0)
            spool.close()

    def test_oversized_item_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            spool = _spool(Path(td), 1024)
            with self.assertRaises(ValueError):
                spool.write(np.zeros(1600, dtype=np.float32), SR, {"key": "x"})
            spool.close()

    def test_stale_cleanup_recomputes_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            spool = _spool(Path(td), 1_000_000)
            item = spool.write(np.zeros(800, dtype=np.float32), SR, {"key": "old"})
            old = time.time() - 7200
            for path in (item.audio_path, item.meta_path):
                import os

                os.utime(path, (old, old))
            removed = spool.cleanup_stale(ttl_sec=3600)
            self.assertEqual(removed, 2)
            self.assertEqual(spool.bytes_used, 0)
            spool.close()


if __name__ == "__main__":
    unittest.main()
