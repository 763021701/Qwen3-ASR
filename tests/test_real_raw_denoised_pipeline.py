import os
import tempfile
import unittest
from unittest.mock import patch

from tools.qwen3_asr_pipeline import stage_prepare


class RealRawDenoisedPipelineTest(unittest.TestCase):
    @patch("tools.qwen3_asr_pipeline.run_command")
    def test_prepare_command_contains_both_manifests(self, mock_run):
        with tempfile.TemporaryDirectory() as td:
            config = {
                "dataset": {
                    "source_type": "real_raw_denoised",
                    "language": "None",
                    "raw_jsonl": "raw.jsonl",
                    "denoised_jsonl": "denoised.jsonl",
                    "test_source": "test.jsonl",
                    "output_dir": "data/example",
                    "train_jsonl": "data/example/train.jsonl",
                    "dev_jsonl": "data/example/dev.jsonl",
                    "test_jsonl": "data/example/test.jsonl",
                    "dev_fraction": 0.2,
                    "split_seed": 42,
                    "check_audio": 1,
                },
                "training": {"output_dir": "outputs/example"},
            }
            stage_prepare(config, dry_run=True)

            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            self.assertTrue(cmd[-1] == "1")
            self.assertIn("prepare_real_raw_denoised_sft.py", cmd[1])
            self.assertEqual(cmd[cmd.index("--raw_jsonl") + 1], os.path.abspath("raw.jsonl"))
            self.assertEqual(
                cmd[cmd.index("--denoised_jsonl") + 1],
                os.path.abspath("denoised.jsonl"),
            )
            self.assertEqual(cmd[cmd.index("--test_source") + 1], os.path.abspath("test.jsonl"))


if __name__ == "__main__":
    unittest.main()
