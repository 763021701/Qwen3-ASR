"""Pipeline passes augmentation flags through to the training command."""

import os
import tempfile
import unittest
from unittest.mock import patch

from tools.qwen3_asr_pipeline import load_config, stage_train


class PipelineAugmentFlagsTest(unittest.TestCase):
    @patch("tools.qwen3_asr_pipeline.run_command")
    def test_stage_train_dry_run_includes_augment_flags(self, mock_run):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = os.path.join(td, "config.yaml")
            with open(cfg_path, "w", encoding="utf-8") as f:
                f.write(
                    """
dataset:
  language: Uyghur
  output_dir: data/example
  train_jsonl: data/example/train.jsonl
  dev_jsonl: data/example/dev.jsonl
training:
  output_dir: outputs/example
  strip_target_brackets: 1
  augment: 1
  speed_factors: "0.9,1.1"
  noise_prob: 0.3
  noise_dir: data/noise_from_seg
runtime:
  dry_run: true
"""
                )
            config = load_config(cfg_path)
            stage_train(config, dry_run=True)
            self.assertEqual(config["training"]["augment"], 1)
            self.assertEqual(config["training"]["speed_factors"], "0.9,1.1")
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            self.assertIn("--noise_dir", cmd)
            idx = cmd.index("--noise_dir")
            self.assertEqual(cmd[idx + 1], "data/noise_from_seg")
            self.assertIn("--strip_target_brackets", cmd)
            idx = cmd.index("--strip_target_brackets")
            self.assertEqual(cmd[idx + 1], "1")


if __name__ == "__main__":
    unittest.main()
