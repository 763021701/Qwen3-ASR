"""Pipeline keeps augmentation in the data config, not training CLI flags."""

import os
import tempfile
import unittest
from unittest.mock import patch

from tools.qwen3_asr_pipeline import load_config, stage_train


class PipelineAugmentFlagsTest(unittest.TestCase):
    @patch("tools.qwen3_asr_pipeline.run_command")
    def test_stage_train_uses_data_config_for_augmentation(self, mock_run):
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
  data_config: configs/data_configs/example.yaml
  output_dir: outputs/example
  strip_target_brackets: 1
runtime:
  dry_run: true
"""
                )
            config = load_config(cfg_path)
            stage_train(config, dry_run=True)
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            self.assertIn("--data_config", cmd)
            self.assertIn("--strip_target_brackets", cmd)
            for legacy in (
                "--augment",
                "--speed_prob",
                "--noise_dir",
                "--specaug_prob",
                "--nospeech_augment",
            ):
                self.assertNotIn(legacy, cmd)


if __name__ == "__main__":
    unittest.main()
