#!/usr/bin/env python3
# coding=utf-8
"""Shared-forward CTC + LLM inference: audio_tower runs ONCE per chunk.

One `audio_tower(..., return_ctc_hidden=True)` call yields both:
  - hidden_states[0]  (pre-proj)  -> CTC decoder -> CTC text
  - last_hidden_state (post-proj) -> LLM (via precomputed `audio_features`) -> LLM text

The LLM generate path is given `audio_features` (precomputed post-proj), so it
skips its internal `get_audio_features` (no second audio_tower pass). Requires
the model-code changes in modeling_qwen3_asr.py:
  get_ctc_logits(ctc_hidden_states=...), forward(audio_features=...),
  prepare_inputs_for_generation(audio_features=...), generate kwargs routing.

`--compare` also runs the separate paths (CTC via get_ctc_logits(input_features)
+ LLM via generate(input_features=...)) and counts audio_tower forwards to prove
shared = 1/chunk vs separate = 2/chunk, with identical outputs.
"""
import argparse
import os
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from infer_qwen3_asr_ctc import (  # noqa: E402
    default_ctc_vocab_path,
    load_audio,
    load_ctc_checkpoint,
)
from qwen_asr.inference.utils import SAMPLE_RATE, split_audio_into_chunks  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser("Shared-forward CTC+LLM inference (audio_tower once)")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--audio", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="auto")
    p.add_argument("--ctc_vocab_path", type=str, default="")
    p.add_argument("--funasr_path", type=str, default="")
    p.add_argument("--chunk_sec", type=float, default=25.0)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--context", type=str, default="")
    p.add_argument("--language", type=str, default="",
                   help="Force output language (e.g. Chinese). Empty for auto-detect "
                        "(emits a `language X<asr_text>` prefix per chunk).")
    p.add_argument("--compare", action="store_true",
                   help="Also run separate CTC + separate LLM, count audio_tower calls, diff outputs")
    return p.parse_args()


def _model_device_dtype(model):
    dev = getattr(model, "device", None)
    if dev is None:
        dev = next(model.parameters()).device
    return dev, getattr(model, "dtype", torch.float32)


def _build_prompt(processor, context, force_language=None):
    msgs = [
        {"role": "system", "content": context or ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    base = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    if force_language:
        base = base + f"language {force_language}{'<asr_text>'}"
    return base


def _featurize(processor, prompt, wav, device, dtype):
    inputs = processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True)
    inputs["input_ids"] = inputs["input_ids"].to(device)
    inputs["attention_mask"] = inputs["attention_mask"].to(device)
    inputs["feature_attention_mask"] = inputs["feature_attention_mask"].to(device)
    inputs["input_features"] = inputs["input_features"].to(device=device, dtype=dtype)
    return inputs


def _decode_llm(processor, sequences, prompt_len):
    return processor.batch_decode(
        sequences[:, prompt_len:], skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


class _TowerCounter:
    def __init__(self, tower):
        self.tower = tower
        self.orig = tower.forward
        self.n = 0

    def _wrapped(self, *args, **kwargs):
        self.n += 1
        return self.orig(*args, **kwargs)

    def install(self):
        self.tower.forward = self._wrapped

    def reset(self):
        self.n = 0

    def restore(self):
        self.tower.forward = self.orig


@torch.no_grad()
def shared_forward_chunk(model, processor, ctc_tokenizer, thinker, wav, context,
                         max_new_tokens, device, dtype, force_language=None):
    """ONE audio_tower call -> (ctc_text, llm_text)."""
    prompt = _build_prompt(processor, context, force_language)
    inputs = _featurize(processor, prompt, wav, device, dtype)
    feat_mask = inputs["feature_attention_mask"]
    flen = feat_mask.sum(dim=1)[0]
    input_feature = inputs["input_features"][0]

    # --- the single shared audio_tower forward ---
    audio_output = thinker.audio_tower(
        input_feature[:, :flen], feature_lens=flen.unsqueeze(0), return_ctc_hidden=True,
    )
    last_hidden = audio_output.last_hidden_state      # post-proj  -> LLM
    ctc_hidden = audio_output.hidden_states[0]        # pre-proj   -> CTC

    # CTC branch (no audio_tower; uses precomputed hidden)
    ctc_logits, ctc_input_lengths = thinker.get_ctc_logits(ctc_hidden_states=[ctc_hidden])
    ctc_res = thinker.decode_ctc_logits(ctc_logits, ctc_input_lengths, tokenizer=ctc_tokenizer, return_timestamps=False)
    ctc_text = (ctc_res[0].get("ctc_text") or "").strip() if ctc_res else ""

    # LLM branch (no audio_tower; audio_features precomputed)
    prompt_len = inputs["input_ids"].shape[1]
    out = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        feature_attention_mask=feat_mask,
        input_features=inputs["input_features"],
        audio_features=last_hidden,
        max_new_tokens=max_new_tokens,
    )
    llm_text = _decode_llm(processor, out.sequences, prompt_len)
    return ctc_text, llm_text


@torch.no_grad()
def separate_forward_chunk(model, processor, ctc_tokenizer, thinker, wav, context,
                           max_new_tokens, device, dtype, force_language=None):
    """Two audio_tower calls (CTC path + LLM path) -> (ctc_text, llm_text)."""
    prompt = _build_prompt(processor, context, force_language)
    inputs = _featurize(processor, prompt, wav, device, dtype)
    feat_mask = inputs["feature_attention_mask"]

    # CTC path: get_ctc_logits runs audio_tower internally
    ctc_logits, ctc_input_lengths = thinker.get_ctc_logits(
        input_features=inputs["input_features"], feature_attention_mask=feat_mask)
    ctc_res = thinker.decode_ctc_logits(ctc_logits, ctc_input_lengths, tokenizer=ctc_tokenizer, return_timestamps=False)
    ctc_text = (ctc_res[0].get("ctc_text") or "").strip() if ctc_res else ""

    # LLM path: generate(input_features=...) runs audio_tower internally (no audio_features)
    prompt_len = inputs["input_ids"].shape[1]
    out = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        feature_attention_mask=feat_mask,
        input_features=inputs["input_features"],
        max_new_tokens=max_new_tokens,
    )
    llm_text = _decode_llm(processor, out.sequences, prompt_len)
    return ctc_text, llm_text


def main():
    args = parse_args()
    if not os.path.exists(args.audio):
        raise FileNotFoundError(f"Audio file not found: {args.audio}")
    vocab_path = args.ctc_vocab_path or default_ctc_vocab_path()
    model, processor, ctc_tokenizer = load_ctc_checkpoint(
        checkpoint=args.checkpoint, vocab_path=vocab_path,
        funasr_path=args.funasr_path, device=args.device, dtype_name=args.dtype,
    )
    thinker = model.thinker if hasattr(model, "thinker") else model
    device, dtype = _model_device_dtype(model)

    wav = load_audio(args.audio)
    duration = len(wav) / SAMPLE_RATE
    parts = split_audio_into_chunks(wav, SAMPLE_RATE, args.chunk_sec)
    print(f"=== {args.audio}  ({duration:.2f}s)  {len(parts)} chunk(s) ===")

    counter = _TowerCounter(thinker.audio_tower)
    counter.install()

    # --- shared forward ---
    counter.reset()
    shared_ctc, shared_llm = [], []
    for i, (cwav, off) in enumerate(parts):
        ct, lt = shared_forward_chunk(model, processor, ctc_tokenizer, thinker, cwav,
                                      args.context, args.max_new_tokens, device, dtype,
                                      args.language or None)
        shared_ctc.append(ct)
        shared_llm.append(lt)
    shared_n = counter.n
    shared_ctc_text = "".join(shared_ctc)
    shared_llm_text = "".join(shared_llm)
    print(f"\n[shared]  audio_tower_calls={shared_n}  (expect {len(parts)})  "
          f"ctc_chars={len(shared_ctc_text)} llm_chars={len(shared_llm_text)}")
    for i, (ct, lt) in enumerate(zip(shared_ctc, shared_llm)):
        print(f"  chunk {i}: ctc={len(ct)}ch llm={len(lt)}ch")
    print(f"[shared CTC] {shared_ctc_text}")
    print(f"[shared LLM] {shared_llm_text}")

    # --- separate forward (proof of 2x calls + identical outputs) ---
    if args.compare:
        counter.reset()
        sep_ctc, sep_llm = [], []
        for i, (cwav, off) in enumerate(parts):
            ct, lt = separate_forward_chunk(model, processor, ctc_tokenizer, thinker, cwav,
                                            args.context, args.max_new_tokens, device, dtype,
                                            args.language or None)
            sep_ctc.append(ct)
            sep_llm.append(lt)
        sep_n = counter.n
        sep_ctc_text = "".join(sep_ctc)
        sep_llm_text = "".join(sep_llm)
        print(f"\n[separate] audio_tower_calls={sep_n}  (expect {2 * len(parts)})  "
              f"ctc_chars={len(sep_ctc_text)} llm_chars={len(sep_llm_text)}")
        print(f"[separate CTC] {sep_ctc_text}")
        print(f"[separate LLM] {sep_llm_text}")
        print(f"\n[verify] CTC match: {shared_ctc_text == sep_ctc_text}")
        print(f"[verify] LLM match: {shared_llm_text == sep_llm_text}")
        print(f"[verify] audio_tower calls: shared={shared_n} separate={sep_n} "
              f"(shared should be half of separate)")

    counter.restore()


if __name__ == "__main__":
    main()
