"""Unit tests for finetuning.module_freeze and LoRA scope resolution."""

import argparse
import re
import unittest

import torch.nn as nn

from finetuning.module_freeze import (
    PARTS,
    count_part_parameters,
    parse_parts,
    set_part_freeze,
)


class _FakeThinker(nn.Module):
    """Mimics the Qwen3-ASR thinker structure used by module_freeze."""

    def __init__(self):
        super().__init__()
        audio_tower = nn.Module()
        audio_tower.conv2d1 = nn.Conv2d(1, 2, 3, padding=1)
        audio_tower.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
        audio_tower.ln_post = nn.LayerNorm(4)
        audio_tower.conv_out = nn.Linear(4, 8)
        audio_tower.proj1 = nn.Linear(4, 4)
        audio_tower.proj2 = nn.Linear(4, 8)
        self.audio_tower = audio_tower
        self.model = nn.Module()
        self.model.embed = nn.Embedding(10, 4)
        self.model.decoder = nn.ModuleList([nn.Linear(4, 4)])
        self.lm_head = nn.Linear(4, 10)


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.thinker = _FakeThinker()


def _make_model() -> _FakeModel:
    model = _FakeModel()
    model.train()  # start in train mode, like a fresh training run
    return model


class ParsePartsTest(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(parse_parts(None), ())
        self.assertEqual(parse_parts(""), ())
        self.assertEqual(parse_parts(" , "), ())

    def test_valid(self):
        self.assertEqual(parse_parts("encoder,aligner"), ("encoder", "aligner"))
        self.assertEqual(parse_parts("llm"), ("llm",))

    def test_dedup_and_whitespace(self):
        self.assertEqual(parse_parts(" llm , llm , encoder "), ("llm", "encoder"))

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            parse_parts("encoder,audio")


class PartFreezeTest(unittest.TestCase):
    def _counts(self, model):
        counts = count_part_parameters(model)
        self.assertEqual(set(counts), set(PARTS))
        return counts

    def test_no_freeze_all_trainable(self):
        model = _make_model()
        counts = self._counts(model)
        for part in PARTS:
            total, trainable = counts[part]
            self.assertGreater(total, 0)
            self.assertEqual(trainable, total)

    def test_freeze_encoder_only(self):
        model = _make_model()
        set_part_freeze(model, ("encoder",))
        at = model.thinker.audio_tower
        for name in ("conv2d1", "layers", "ln_post"):
            for p in getattr(at, name).parameters():
                self.assertFalse(p.requires_grad, name)
            self.assertFalse(getattr(at, name).training, name)
        for name in ("conv_out", "proj1", "proj2"):
            for p in getattr(at, name).parameters():
                self.assertTrue(p.requires_grad, name)
            self.assertTrue(getattr(at, name).training, name)
        for module in (model.thinker.model, model.thinker.lm_head):
            for p in module.parameters():
                self.assertTrue(p.requires_grad)
        counts = self._counts(model)
        self.assertEqual(counts["encoder"][1], 0)
        self.assertEqual(counts["aligner"][1], counts["aligner"][0])
        self.assertEqual(counts["llm"][1], counts["llm"][0])

    def test_freeze_aligner_and_llm(self):
        model = _make_model()
        set_part_freeze(model, ("aligner", "llm"))
        at = model.thinker.audio_tower
        for name in ("conv2d1", "layers", "ln_post"):
            for p in getattr(at, name).parameters():
                self.assertTrue(p.requires_grad, name)
        for name in ("conv_out", "proj1", "proj2"):
            for p in getattr(at, name).parameters():
                self.assertFalse(p.requires_grad, name)
            self.assertFalse(getattr(at, name).training, name)
        for module in (model.thinker.model, model.thinker.lm_head):
            for p in module.parameters():
                self.assertFalse(p.requires_grad)
        counts = self._counts(model)
        self.assertEqual(counts["encoder"][1], counts["encoder"][0])
        self.assertEqual(counts["aligner"][1], 0)
        self.assertEqual(counts["llm"][1], 0)

    def test_freeze_all(self):
        model = _make_model()
        set_part_freeze(model, PARTS)
        counts = self._counts(model)
        for part in PARTS:
            self.assertEqual(counts[part][1], 0)

    def test_toggle_off_restores_trainable(self):
        # set_part_freeze only enforces the frozen state (requires_grad + eval);
        # restoring train() mode on unfrozen parts is the Trainer's job.
        model = _make_model()
        set_part_freeze(model, ("encoder",))
        set_part_freeze(model, ())
        counts = self._counts(model)
        for part in PARTS:
            self.assertEqual(counts[part][1], counts[part][0])


class LoraScopeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from finetuning.qwen3_asr_sft_lora import LORA_TARGETS, _resolve_lora_scope

        cls.LORA_TARGETS = LORA_TARGETS
        cls.resolve = staticmethod(_resolve_lora_scope)

    def test_legacy_single_scope_unchanged(self):
        for scope in self.LORA_TARGETS:
            self.assertEqual(self.resolve(scope), self.LORA_TARGETS[scope])

    def test_combination_is_union(self):
        pattern = self.resolve("aligner,llm")
        rx = re.compile(pattern)
        self.assertIsNotNone(rx.fullmatch("audio_tower.proj1"))
        self.assertIsNotNone(rx.fullmatch("model.layers.0.self_attn.q_proj"))
        self.assertIsNone(rx.fullmatch("audio_tower.layers.0.self_attn.q_proj"))

    def test_invalid_part_raises(self):
        with self.assertRaises(ValueError):
            self.resolve("aligner,bogus")

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            self.resolve("")


class GrpoFrozenPartsTest(unittest.TestCase):
    def test_default_keeps_llm_only(self):
        from finetuning.qwen3_asr_grpo import _resolve_frozen_parts

        self.assertEqual(
            _resolve_frozen_parts(argparse.Namespace(freeze_modules=None)),
            ["encoder", "aligner"],
        )

    def test_explicit_spec(self):
        from finetuning.qwen3_asr_grpo import _resolve_frozen_parts

        self.assertEqual(
            _resolve_frozen_parts(argparse.Namespace(freeze_modules="encoder")),
            ["encoder"],
        )
        self.assertEqual(
            _resolve_frozen_parts(argparse.Namespace(freeze_modules="")),
            [],
        )


if __name__ == "__main__":
    unittest.main()
