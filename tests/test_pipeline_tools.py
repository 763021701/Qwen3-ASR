import json
import os
import tempfile
import unittest

from tools.qwen3_asr_pipeline import find_latest_checkpoint, load_config, manifest_paths
from tools.validate_qwen3_asr_jsonl import validate_jsonl


class PipelineToolsTest(unittest.TestCase):
    def test_validate_jsonl_accepts_valid_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            audio = os.path.join(td, "a.wav")
            manifest = os.path.join(td, "train.jsonl")
            open(audio, "wb").close()
            with open(manifest, "w", encoding="utf-8") as f:
                f.write(json.dumps({"audio": audio, "text": "language Uyghur<asr_text>hello"}) + "\n")

            report = validate_jsonl(manifest, expected_language="Uyghur", check_audio=True)

            self.assertTrue(report.valid)
            self.assertEqual(report.total_lines, 1)
            self.assertEqual(report.valid_records, 1)
            self.assertEqual(report.issues, [])

    def test_validate_jsonl_rejects_bad_language_and_missing_audio(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = os.path.join(td, "train.jsonl")
            with open(manifest, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "audio": os.path.join(td, "missing.wav"),
                            "text": "language Klingon<asr_text>hello",
                        }
                    )
                    + "\n"
                )

            report = validate_jsonl(manifest, expected_language="Uyghur", check_audio=True)
            codes = {issue.code for issue in report.issues}

            self.assertFalse(report.valid)
            self.assertIn("audio_not_found", codes)
            self.assertIn("language_mismatch", codes)
            self.assertIn("unsupported_language", codes)

    def test_find_latest_checkpoint_uses_numeric_step(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "checkpoint-9"))
            os.makedirs(os.path.join(td, "checkpoint-100"))
            os.makedirs(os.path.join(td, "checkpoint-20"))
            os.makedirs(os.path.join(td, "not-a-checkpoint"))

            self.assertEqual(find_latest_checkpoint(td), os.path.join(td, "checkpoint-100"))

    def test_load_config_and_manifest_defaults(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = os.path.join(td, "config.yaml")
            with open(cfg, "w", encoding="utf-8") as f:
                f.write(
                    """
dataset:
  source_type: common_voice
  language: Uyghur
  output_dir: data/example
training:
  output_dir: outputs/example
runtime:
  dry_run: true
"""
                )

            config = load_config(cfg)
            paths = manifest_paths(config)

            self.assertEqual(config["dataset"]["language"], "Uyghur")
            self.assertTrue(paths["train"].endswith(os.path.join("data", "example", "train.jsonl")))
            self.assertTrue(paths["dev"].endswith(os.path.join("data", "example", "dev.jsonl")))
            self.assertTrue(paths["test"].endswith(os.path.join("data", "example", "test.jsonl")))


if __name__ == "__main__":
    unittest.main()

