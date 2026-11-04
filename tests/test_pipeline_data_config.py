# coding=utf-8
"""Tests for wiring data_config into the SFT pipeline and trainer."""

import io
import unittest
from contextlib import redirect_stdout

from tools.qwen3_asr_pipeline import stage_train


class PipelineDataConfigTest(unittest.TestCase):
    def test_stage_train_passes_data_config_and_max_steps(self):
        config = {
            "dataset": {"dev_jsonl": "data/robust/dev.jsonl"},
            "training": {
                "data_config": "configs/robust/data_local.yaml",
                "output_dir": "outputs/test_data_config",
                "max_steps": 10,
                "batch_size": 2,
            },
            "runtime": {"launcher": "python", "dry_run": True},
        }
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            stage_train(config, dry_run=True)
        command = buffer.getvalue()
        self.assertIn("--data_config", command)
        self.assertIn("configs/robust/data_local.yaml", command)
        self.assertIn("--max_steps 10", command)
        self.assertIn("--eval_file", command)
        self.assertNotIn("--train_file", command)

    def test_stage_train_requires_data_config(self):
        config = {
            "dataset": {},
            "training": {"output_dir": "outputs/test_data_config"},
            "runtime": {"launcher": "python", "dry_run": True},
        }
        with self.assertRaises(ValueError):
            stage_train(config, dry_run=True)

    def test_stage_train_lora_uses_data_config(self):
        config = {
            "dataset": {},
            "training": {
                "data_config": "configs/robust/data_local.yaml",
                "output_dir": "outputs/test_data_config",
                "use_lora": 1,
                "max_steps": 5,
            },
            "runtime": {"launcher": "python", "dry_run": True},
        }
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            stage_train(config, dry_run=True)
        command = buffer.getvalue()
        self.assertIn("qwen3_asr_sft_lora.py", command)
        self.assertIn("--data_config", command)
        self.assertIn("--max_steps 5", command)
        self.assertNotIn("--train_file", command)


if __name__ == "__main__":
    unittest.main()
