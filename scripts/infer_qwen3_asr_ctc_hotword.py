#!/usr/bin/env python3
# coding=utf-8
"""CTC hotword retrieval -> LLM context biasing (whole-audio, ONE audio_tower).

Whole audio is featurized via ``processor(text, audio, truncation=False)`` (the
Qwen3-ASR processor disables WhisperFeatureExtractor's 30s truncation in
``__call__``; calling ``feature_extractor`` directly does NOT and silently
truncates to 30s -- that was the misdiagnosed "~25s CTC ceiling"). Then:

  1. ONE ``audio_tower(..., return_ctc_hidden=True)`` -> pre-proj (CTC) + post-proj (LLM).
  2. CTC decoder on the pre-proj:
       - normal length (<= --ctc_slice_threshold_sec): whole pre-proj, one CTC pass;
       - very long: pre-proj sliced into <=--ctc_slice_sec segments, CTC per segment,
         log-probs concatenated (bounds the CTC decoder's O(N^2) attention memory).
  3. Retrieve hot-words via ctc_rag_hw.CTCRagRetrieverFromLogProbs on the (possibly
     concatenated) log-probs.
  4. ONE LLM generate with post-proj as precomputed ``audio_features`` and the
     retrieved hot-words injected as context (space/nano_style/...).

``--compare`` also runs the LLM with empty context (no hot-words).
"""
import argparse
import os
import sys
import time
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
from qwen_asr.inference.utils import SAMPLE_RATE  # noqa: E402
from ctc_rag_hw import CTCRagRetrieverFromLogProbs  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser("CTC hotword retrieval -> LLM context biasing (whole-audio)")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--audio", type=str, required=True)
    p.add_argument("--hotwords", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="auto")
    p.add_argument("--ctc_vocab_path", type=str, default="")
    p.add_argument("--funasr_path", type=str, default="")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--language", type=str, default="")
    p.add_argument("--ctc_topk", type=int, default=30)
    p.add_argument("--max_hotwords", type=int, default=32)
    p.add_argument("--context_format", choices=["space", "comma", "structured", "nano_style"],
                   default="nano_style")
    p.add_argument("--ctc_slice_threshold_sec", type=float, default=60.0,
                   help="Audio longer than this slices pre-proj for the CTC decoder "
                        "(bounds O(N^2) attention memory). <= this -> whole CTC.")
    p.add_argument("--ctc_slice_sec", type=float, default=25.0,
                   help="CTC decoder segment length when slicing.")
    p.add_argument("--compare", action="store_true",
                   help="Also run LLM with empty context (no hot-words).")
    return p.parse_args()


def _model_device_dtype(model):
    dev = getattr(model, "device", None)
    if dev is None:
        dev = next(model.parameters()).device
    return dev, getattr(model, "dtype", torch.float32)


def _format_hotword_context(hotwords, fmt="space"):
    if not hotwords:
        return ""
    if fmt == "space":
        return " ".join(hotwords)
    if fmt == "comma":
        return ", ".join(hotwords)
    if fmt == "structured":
        return f"请结合以下热词进行语音转写。\n热词列表：[{', '.join(hotwords)}]"
    if fmt == "nano_style":
        return ("请结合上下文信息，更加准确地完成语音转写任务。"
                "如果没有相关信息，我们会留空。\n\n"
                f"**上下文信息：**\n\n热词列表：[{', '.join(hotwords)}]")
    return " ".join(hotwords)


def _build_prompt(processor, context, force_language=None):
    msgs = [
        {"role": "system", "content": context or ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    base = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    if force_language:
        base = base + f"language {force_language}{'<asr_text>'}"
    return base


@torch.no_grad()
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

    retriever = CTCRagRetrieverFromLogProbs(
        blank_id=thinker.ctc_head.blank_id, ctc_tokenizer=ctc_tokenizer)
    n_hw = retriever.load_hotwords(args.hotwords)

    wav = load_audio(args.audio)
    duration = len(wav) / SAMPLE_RATE
    force_language = args.language or None

    # ---- 1. featurize whole audio (truncation=False) + ONE audio_tower ----
    prompt0 = _build_prompt(processor, "", force_language)
    inputs0 = processor(text=[prompt0], audio=[wav], return_tensors="pt",
                        padding=True, truncation=False)
    feat_mask = inputs0["feature_attention_mask"].to(device)
    input_features = inputs0["input_features"].to(device=device, dtype=dtype)
    flen = feat_mask.sum(dim=1)[0]

    t0 = time.perf_counter()
    audio_output = thinker.audio_tower(
        input_features[0][:, :flen], feature_lens=flen.unsqueeze(0), return_ctc_hidden=True)
    t_tower = time.perf_counter() - t0
    post_proj = audio_output.last_hidden_state   # [T, dim] -> LLM
    pre_proj = audio_output.hidden_states[0]     # [T, 1024] -> CTC
    T = pre_proj.size(0)
    fps = T / duration if duration > 0 else 13.0

    # ---- 2. CTC log-probs (whole or sliced) ----
    t0 = time.perf_counter()
    if duration <= args.ctc_slice_threshold_sec:
        ctc_mode = f"whole ({T} frames)"
        logits, lens = thinker.get_ctc_logits(ctc_hidden_states=[pre_proj])
        log_probs = logits[0, :lens[0], :]
    else:
        seg = max(1, int(args.ctc_slice_sec * fps))
        slices = [pre_proj[i:i + seg] for i in range(0, T, seg)]
        ctc_mode = f"sliced {len(slices)}x~{args.ctc_slice_sec}s ({[s.size(0) for s in slices]} frames)"
        parts = []
        for s in slices:
            lg, ln = thinker.get_ctc_logits(ctc_hidden_states=[s])
            parts.append(lg[0, :ln[0], :])
        log_probs = torch.cat(parts, dim=0)
    t_ctc = time.perf_counter() - t0

    # ---- 3. retrieve hot-words ----
    t0 = time.perf_counter()
    rag = retriever.retrieve_from_log_probs(
        log_probs, max_hotwords=args.max_hotwords, ctc_topk=args.ctc_topk)
    t_rag = time.perf_counter() - t0
    context = _format_hotword_context(rag.retrieved_hotwords, args.context_format)

    # ---- 4. ONE LLM generate (whole audio, precomputed audio_features) ----
    def gen(ctx):
        prompt = _build_prompt(processor, ctx, force_language)
        feat = processor(text=[prompt], audio=[wav], return_tensors="pt",
                         padding=True, truncation=False)
        ids = feat["input_ids"].to(device)
        am = feat["attention_mask"].to(device)
        fm = feat["feature_attention_mask"].to(device)
        ifeat = feat["input_features"].to(device=device, dtype=dtype)
        prompt_len = ids.size(1)
        out = model.generate(
            input_ids=ids, attention_mask=am, feature_attention_mask=fm,
            input_features=ifeat, audio_features=post_proj,
            max_new_tokens=args.max_new_tokens)
        return processor.batch_decode(
            out.sequences[:, prompt_len:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]

    t0 = time.perf_counter()
    biased = gen(context)
    t_llm_biased = time.perf_counter() - t0
    baseline, t_llm_base = None, 0.0
    if args.compare:
        t0 = time.perf_counter()
        baseline = gen("")
        t_llm_base = time.perf_counter() - t0

    # ---- report ----
    print(f"=== {args.audio}  ({duration:.1f}s)  hotwords={n_hw} ===")
    print(f"[ctc mode] {ctc_mode}  fps={fps:.1f}")
    print(f"[time] audio_tower={t_tower:.2f}s  ctc={t_ctc:.2f}s  rag={t_rag:.2f}s  "
          f"llm_biased={t_llm_biased:.2f}s" + (f"  llm_baseline={t_llm_base:.2f}s" if args.compare else ""))
    print(f"[ctc greedy] {rag.greedy_text}")
    print(f"[hotwords] ({len(rag.retrieved_hotwords)}) {rag.retrieved_hotwords}")
    print(f"[context] {context!r}")
    print(f"[biased]   chars={len(biased)}")
    print(biased)
    if baseline is not None:
        print(f"[baseline] chars={len(baseline)}")
        print(baseline)


if __name__ == "__main__":
    main()
