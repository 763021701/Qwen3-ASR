"""Verify single-mel collator path matches dual processor() outputs."""

import unittest

import numpy as np
import torch
from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor

MODEL_PATH = (
    "/root/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/"
    "7278e1e70fe206f11671096ffdd38061171dd6e5"
)
SAMPLE_AUDIOS = [
    "/root/autodl-tmp/workspace/dataset/SwitchLingua_audio/Cantonese/2640_0.m4a",
    "/root/autodl-tmp/workspace/dataset/SwitchLingua_audio/Cantonese/2651_3.m4a",
    "/root/autodl-tmp/workspace/dataset/SwitchLingua_audio/Cantonese/58_1.m4a",
]
SAMPLE_TEXTS = [
    "language English,Cantonese<asr_text>嗰個新smartphone出咗喇，真係好exciting!",
    "language English,Cantonese<asr_text>聽說今次聖誕裝飾會好靚, I'm excited to see how the city lights up!",
    "language English,Cantonese<asr_text>你有冇試過嗰啲 character meet-and-greets？我竟然見到Mickey Mouse, it was like a dream come true!",
]

from finetuning.processor_collate import build_processor_batch_inputs
from finetuning.qwen3_asr_sft import load_audio, make_preprocess_fn_prefix_only
def collate_dual_processor(processor, prefix_texts, targets, audios):
    """Current collator path: processor() called twice (mel computed twice)."""
    eos = processor.tokenizer.eos_token or ""
    full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]

    full_inputs = processor(
        text=full_texts,
        audio=audios,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    prefix_inputs = processor(
        text=prefix_texts,
        audio=audios,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )

    prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
    labels = full_inputs["input_ids"].clone()
    attn = full_inputs["attention_mask"]
    starts = (attn == 1).long().argmax(dim=1).tolist()
    for i, (pl, st) in enumerate(zip(prefix_lens, starts)):
        labels[i, st : st + pl] = -100
    labels[attn == 0] = -100
    full_inputs["labels"] = labels
    return full_inputs, prefix_inputs


def collate_single_mel(processor, prefix_texts, targets, audios):
    """Proposed path: feature_extractor once, tokenizer twice."""
    eos = processor.tokenizer.eos_token or ""
    full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]
    full_inputs, prefix_tok = build_processor_batch_inputs(
        processor, full_texts, prefix_texts, audios
    )

    prefix_lens = prefix_tok["attention_mask"].sum(dim=1).tolist()
    labels = full_inputs["input_ids"].clone()
    attn = full_inputs["attention_mask"]
    starts = (attn == 1).long().argmax(dim=1).tolist()
    for i, (pl, st) in enumerate(zip(prefix_lens, starts)):
        labels[i, st : st + pl] = -100
    labels[attn == 0] = -100
    full_inputs["labels"] = labels
    return full_inputs, prefix_tok


class ProcessorSingleMelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = Qwen3ASRProcessor.from_pretrained(
            MODEL_PATH, fix_mistral_regex=True
        )
        cls.preprocess_fn = make_preprocess_fn_prefix_only(cls.processor)

    def _build_batch(self, n=3, use_synthetic=False):
        if use_synthetic:
            audios = []
            prefix_texts = []
            targets = []
            for i in range(n):
                sr = 16000
                dur = 0.5 + i * 1.2
                t = np.linspace(0, dur, int(sr * dur), endpoint=False)
                wav = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
                ex = {"audio": f"synthetic_{i}", "text": SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]}
                row = type(self).preprocess_fn(ex)
                audios.append(wav)
                prefix_texts.append(row["prefix_text"])
                targets.append(row["target"])
            return prefix_texts, targets, audios

        rows = []
        for audio_path, text in zip(SAMPLE_AUDIOS[:n], SAMPLE_TEXTS[:n]):
            rows.append(type(self).preprocess_fn({"audio": audio_path, "text": text}))
        prefix_texts = [r["prefix_text"] for r in rows]
        targets = [r["target"] for r in rows]
        audios = [load_audio(r["audio"]) for r in rows]
        return prefix_texts, targets, audios

    def _assert_equivalent(self, dual_full, dual_prefix, single_full, single_prefix):
        for key in ("input_ids", "attention_mask", "labels"):
            self.assertTrue(
                torch.equal(dual_full[key], single_full[key]),
                msg=f"full_inputs mismatch on {key}",
            )
        self.assertTrue(
            torch.allclose(
                dual_full["input_features"].float(),
                single_full["input_features"].float(),
            ),
            msg="input_features mismatch",
        )
        self.assertTrue(
            torch.equal(
                dual_full["feature_attention_mask"],
                single_full["feature_attention_mask"],
            ),
            msg="feature_attention_mask mismatch",
        )
        self.assertTrue(
            torch.equal(dual_prefix["input_ids"], single_prefix["input_ids"]),
            msg="prefix input_ids mismatch",
        )
        self.assertTrue(
            torch.equal(
                dual_prefix["attention_mask"], single_prefix["attention_mask"]
            ),
            msg="prefix attention_mask mismatch",
        )

    def test_real_audio_batch(self):
        prefix_texts, targets, audios = self._build_batch(n=3, use_synthetic=False)
        dual_full, dual_prefix = collate_dual_processor(
            self.processor, prefix_texts, targets, audios
        )
        single_full, single_prefix = collate_single_mel(
            self.processor, prefix_texts, targets, audios
        )
        self._assert_equivalent(dual_full, dual_prefix, single_full, single_prefix)

    def test_synthetic_varying_lengths(self):
        prefix_texts, targets, audios = self._build_batch(n=3, use_synthetic=True)
        dual_full, dual_prefix = collate_dual_processor(
            self.processor, prefix_texts, targets, audios
        )
        single_full, single_prefix = collate_single_mel(
            self.processor, prefix_texts, targets, audios
        )
        self._assert_equivalent(dual_full, dual_prefix, single_full, single_prefix)

if __name__ == "__main__":
    unittest.main()
