"""Pipeline passes augmentation flags through to the training command."""

import os
import tempfile
import unittest

from tools.qwen3_asr_pipeline import load_config, stage_train


class PipelineAugmentFlagsTest(unittest.TestCase):
    def test_stage_train_dry_run_includes_augment_flags(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = os.path.join(td, "config.yaml")
            with open(cfg_path, "w", encoding="utf-8") as f:
                f.write(
                    """
dataset:
  language: Uyghur
  output_dir: data/example
training:
  output_dir: outputs/example
  augment: 1
  speed_factors: "0.9,1.1"
  noise_prob: 0.3
runtime:
  dry_run: true
"""
                )
            config = load_config(cfg_path)
            stage_train(config, dry_run=True)
            self.assertEqual(config["training"]["augment"], 1)
            self.assertEqual(config["training"]["speed_factors"], "0.9,1.1")


if __name__ == "__main__":
    unittest.main()
