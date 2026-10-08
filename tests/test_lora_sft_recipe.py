"""LoRA must honor the data and evaluation settings used by the SFT baseline."""

import importlib
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from finetuning import qwen3_asr_sft_lora as lora


class LoraSftRecipeTest(unittest.TestCase):
    def test_accepts_sft_selection_and_label_options(self):
        argv = [
            "lora", "--use_lora", "1", "--lora_scope", "all",
            "--lora_r", "16", "--lora_alpha", "32", "--lora_dropout", "0.05",
            "--strip_target_brackets", "1", "--save_best_metric", "cer",
            "--wer_eval_samples", "0", "--wer_batch_size", "2",
            "--wer_max_new_tokens", "512", "--extra_eval_set", "radiology=dev.csv",
        ]
        with patch.object(sys, "argv", argv):
            args = lora.parse_args()
        self.assertEqual(args.save_best_metric, "cer")
        self.assertEqual(args.strip_target_brackets, 1)
        self.assertEqual(args.wer_eval_samples, 0)
        self.assertEqual(args.wer_batch_size, 2)
        self.assertEqual(args.wer_max_new_tokens, 512)
        self.assertEqual(args.extra_eval_set, ["radiology=dev.csv"])

    def test_optimizer_honors_the_same_training_arguments_as_sft(self):
        from transformers import Trainer, TrainingArguments

        with tempfile.TemporaryDirectory() as output:
            args = TrainingArguments(output_dir=output, use_cpu=True, report_to="none")
            baseline = Trainer(model=torch.nn.Linear(2, 2), args=args).create_optimizer()
            adapter = lora.CastFloatInputsTrainer(
                model=torch.nn.Linear(2, 2), args=args
            ).create_optimizer()
        self.assertEqual(type(adapter), type(baseline))
        for key in ("betas", "eps", "fused"):
            self.assertEqual(adapter.defaults[key], baseline.defaults[key])

    def test_preprocess_preserves_per_sample_noise_flag(self):
        processor = SimpleNamespace(apply_chat_template=lambda *a, **k: ["prefix"])
        preprocess = lora.make_preprocess_fn_prefix_only(processor)
        for flag in (0, 1):
            with self.subTest(noise_aug=flag):
                row = preprocess({"audio": "sample.wav", "text": "text", "noise_aug": flag})
                self.assertEqual(row["noise_aug"], flag)
        self.assertEqual(preprocess({"audio": "sample.wav", "text": "text"})["noise_aug"], 1)

    def _collate(self, features, **kwargs):
        processor = SimpleNamespace(tokenizer=SimpleNamespace(eos_token="EOS"))
        collator = lora.DataCollatorForQwen3ASRFinetuning(processor, **kwargs)
        data_module = importlib.import_module(collator.__class__.__module__)
        n = len(features)
        inputs = {
            "input_ids": torch.tensor([[0, 1, 2, 3, 4]] * n),
            "attention_mask": torch.tensor([[0, 1, 1, 1, 1]] * n),
        }
        prefix = {"attention_mask": torch.tensor([[1, 1]] * n)}
        with patch.object(data_module, "load_audio", return_value=np.zeros(8, np.float32)), \
             patch.object(data_module, "augment_waveform", side_effect=lambda wav, *a, **k: wav) as augment, \
             patch.object(data_module, "build_processor_batch_inputs", return_value=(inputs, prefix)) as build:
            batch = collator(features)
        return batch, augment, build

    def test_collator_gates_noise_without_disabling_other_augmentation(self):
        cfg = SimpleNamespace(enabled=True, nospeech=SimpleNamespace(enabled=False))
        features = [
            {"audio": "a.wav", "target": "a", "prefix_text": "P", "aug": 1, "noise_aug": 0},
            {"audio": "b.wav", "target": "b", "prefix_text": "P", "aug": 1, "noise_aug": 1},
            {"audio": "c.wav", "target": "c", "prefix_text": "P", "aug": 0, "noise_aug": 1},
        ]
        batch, augment, _ = self._collate(features, augment=cfg)
        self.assertEqual(augment.call_count, 2)
        self.assertEqual([c.kwargs["noise_enabled"] for c in augment.call_args_list], [False, True])
        self.assertEqual(batch["labels"].tolist(), [[-100, -100, -100, 3, 4]] * 3)

    def test_collator_strips_bracket_glyphs_but_keeps_contents(self):
        features = [{
            "audio": "a.wav", "prefix_text": "P",
            "target": "language None<asr_text>grade (2) [high] {risk}",
        }]
        _, _, build = self._collate(features, strip_target_brackets=True)
        self.assertEqual(build.call_args.args[1], ["Planguage None<asr_text>grade 2 high riskEOS"])


if __name__ == "__main__":
    unittest.main()
