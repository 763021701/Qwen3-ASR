#!/usr/bin/env python3
# coding=utf-8
"""
Qwen3-ASR + RAG hotword inference demo.

Uses Fun-ASR-Nano's CTC encoder to produce rough transcription, retrieves
relevant hotwords via PhonemeCorrector (FastRAG + AccuRAG), then injects
them as context into Qwen3-ASR for final recognition.

Usage:
    python hotword_inference.py audio.wav --hotwords hot.txt

    python hotword_inference.py audio.wav \
        --hotwords hot.txt \
        --nano_model FunAudioLLM/Fun-ASR-Nano-2512 \
        --nano_remote_code ../../model.py \
        --qwen3_model Qwen/Qwen3-ASR-1.7B \
        --max_hotwords 30 --top_k 50

    # Also supports a plain run (no hotwords) for comparison:
    python hotword_inference.py audio.wav --no_hotword
"""

import argparse
import os
import sys
import time

import torch


def count_hotwords(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip() and not line.strip().startswith("#"))


def print_separator(title: str = "", width: int = 70):
    if title:
        print(f"\n{'─' * width}")
        print(f"  {title}")
        print(f"{'─' * width}")
    else:
        print("─" * width)


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR + RAG Hotword Inference Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("audio", type=str, help="Path to audio file")
    parser.add_argument(
        "--hotwords", type=str, default=None,
        help="Hotword file (one per line) or comma-separated list",
    )
    parser.add_argument(
        "--qwen3_model", type=str, default="Qwen/Qwen3-ASR-1.7B",
        help="Qwen3-ASR model name or local path (default: Qwen/Qwen3-ASR-1.7B)",
    )
    parser.add_argument(
        "--nano_model", type=str, default="FunAudioLLM/Fun-ASR-Nano-2512",
        help="Fun-ASR-Nano model name or local path (default: FunAudioLLM/Fun-ASR-Nano-2512)",
    )
    parser.add_argument(
        "--nano_remote_code", type=str, default=None,
        help="Path to Fun-ASR model.py (default: auto-detect from Fun-ASR project)",
    )
    parser.add_argument("--language", type=str, default=None, help="Force language (e.g. Chinese, English)")
    parser.add_argument("--top_k", type=int, default=30, help="PhonemeCorrector candidate count")
    parser.add_argument("--max_hotwords", type=int, default=30, help="Max hotwords injected into context")
    parser.add_argument("--threshold", type=float, default=0.7, help="PhonemeCorrector match threshold")
    parser.add_argument("--similar_threshold", type=float, default=0.5, help="PhonemeCorrector similar threshold")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    parser.add_argument("--no_hotword", action="store_true", help="Run vanilla transcription (no RAG hotword)")
    parser.add_argument("--parallel", action="store_true", help="Run CTC retrieval and Qwen3-ASR audio encoding in parallel")
    parser.add_argument("--context", type=str, default="", help="Manual context string (vanilla mode only)")
    args = parser.parse_args()

    fun_asr_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if fun_asr_root not in sys.path:
        sys.path.insert(0, fun_asr_root)

    qwen_asr_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if qwen_asr_root not in sys.path:
        sys.path.insert(0, qwen_asr_root)

    from qwen_asr import Qwen3ASRModel, CTCHotwordRetriever

    hw_count = 0
    if args.hotwords and os.path.isfile(args.hotwords):
        hw_count = count_hotwords(args.hotwords)

    print("=" * 70)
    print("  Qwen3-ASR + RAG Hotword Inference Demo")
    print("=" * 70)
    print(f"  Audio:          {args.audio}")
    print(f"  Qwen3 model:    {args.qwen3_model}")
    if not args.no_hotword and args.hotwords:
        print(f"  Nano model:     {args.nano_model}")
        print(f"  Hotwords:       {args.hotwords} ({hw_count} hotwords)" if hw_count else f"  Hotwords:       {args.hotwords}")
        print(f"  top_k:          {args.top_k}")
        print(f"  max_hotwords:   {args.max_hotwords}")
        print(f"  threshold:      {args.threshold}")
    else:
        print(f"  Mode:           vanilla (no RAG hotword)")
        if args.context:
            print(f"  Context:        {args.context}")
    print(f"  Language:       {args.language or 'auto-detect'}")
    print(f"  Device:         {args.device}")
    print()

    # --- Load Qwen3-ASR ---
    print_separator("Step 1: Loading Qwen3-ASR model")
    t0 = time.time()
    asr = Qwen3ASRModel.from_pretrained(
        args.qwen3_model,
        dtype=torch.bfloat16,
        device_map=args.device,
        max_inference_batch_size=32,
        max_new_tokens=512,
    )
    print(f"  Qwen3-ASR loaded in {time.time() - t0:.1f}s")

    # --- Hotword mode ---
    if not args.no_hotword and args.hotwords:
        print_separator("Step 2: Loading CTC Hotword Retriever")
        t0 = time.time()

        nano_remote_code = args.nano_remote_code
        if nano_remote_code is None:
            candidate = os.path.join(fun_asr_root, "model.py")
            if os.path.isfile(candidate):
                nano_remote_code = candidate

        retriever = CTCHotwordRetriever(
            nano_model=args.nano_model,
            nano_remote_code=nano_remote_code,
            device=args.device,
            ctc_only=True,
            threshold=args.threshold,
            similar_threshold=args.similar_threshold,
        )
        n_loaded = retriever.load_hotwords(args.hotwords)
        print(f"  CTC retriever loaded in {time.time() - t0:.1f}s ({n_loaded} hotwords)")

        # ---- Serial: CTC + retrieval (for breakdown timing) ----
        print_separator("Step 3: CTC encode + decode + hotword retrieval")

        # 3a. CTC audio encoding + greedy decode
        t_ctc_start = time.time()
        ctc_text = retriever.ctc_decode(args.audio)
        t_ctc_decode = time.time() - t_ctc_start

        # 3b. Full retrieve (includes 2nd CTC decode + PhonemeCorrector)
        t_retrieve_start = time.time()
        rr = retriever.retrieve(args.audio, top_k=args.top_k, max_hotwords=args.max_hotwords)
        t_retrieve = time.time() - t_retrieve_start

        t_hotword_total = t_ctc_decode + t_retrieve

        print(f"  CTC text:           {rr.ctc_text}")
        print(f"  CTC decode time:    {t_ctc_decode:.3f}s  (Nano encoder + CTC decoder)")
        print(f"  Retrieval time:     {t_retrieve:.3f}s  (CTC decode + PhonemeCorrector)")
        print(f"  Hotword total:      {t_hotword_total:.3f}s")
        print(f"  Retrieved:          {len(rr.retrieved_hotwords)} hotwords")

        if rr.retrieved_hotwords:
            print()
            for i, hw in enumerate(rr.retrieved_hotwords, 1):
                score = rr.hotword_scores.get(hw, 0.0)
                print(f"    {i:3d}. {hw:<20s} score={score:.4f}")

        if rr.details:
            fast_raw = rr.details.get("fast_raw", [])
            accu_raw = rr.details.get("accu_raw", [])
            print(f"\n  FastRAG candidates: {len(fast_raw)}")
            print(f"  AccuRAG re-ranked:  {len(accu_raw)}")

        print(f"\n  Context string:  \"{rr.context_string}\"")

        # ---- Serial ASR (for comparison baseline) ----
        print_separator("Step 4: Running Qwen3-ASR with hotword context (serial)")
        t_asr_start = time.time()
        results_serial = asr.transcribe_vanilla(
            audio=args.audio,
            context=rr.context_string,
            language=args.language,
            return_time_stamps=False,
        )
        t_asr = time.time() - t_asr_start

        r = results_serial[0]
        print(f"  Language:        {r.language}")
        print(f"  Text:            {r.text}")
        print(f"  Qwen3-ASR time:  {t_asr:.3f}s  (audio encoder + LLM generate)")
        print(f"  Context used:    \"{rr.context_string}\"")

        t_serial_e2e = t_hotword_total + t_asr

        # ---- Parallel run (if requested) ----
        if args.parallel:
            print_separator("Step 5: Parallel hotword transcription")
            t_par_start = time.time()
            results_par = asr.transcribe_hotword(
                audio=args.audio,
                hotword_retriever=retriever,
                language=args.language,
                return_time_stamps=False,
                top_k=args.top_k,
                max_hotwords=args.max_hotwords,
                parallel=True,
            )
            t_par_e2e = time.time() - t_par_start

            rp = results_par[0]
            print(f"  Language:        {rp.language}")
            print(f"  Text:            {rp.text}")
            print(f"  Parallel e2e:    {t_par_e2e:.3f}s")
            print(f"  Context used:    \"{rp.context_used}\"")

        # ---- Timing summary ----
        print_separator("Timing Summary")
        print(f"  [CTC decode]       {t_ctc_decode:7.3f}s  |  Fun-ASR-Nano encoder + CTC greedy")
        print(f"  [Hotword retrieve] {t_retrieve:7.3f}s  |  CTC decode (2nd) + PhonemeCorrector")
        print(f"  [Qwen3-ASR]        {t_asr:7.3f}s  |  Audio encoder + LLM generate")
        print(f"  {'─' * 50}")
        print(f"  [Serial e2e]       {t_serial_e2e:7.3f}s")
        if args.parallel:
            saved = t_serial_e2e - t_par_e2e
            pct = saved / t_serial_e2e * 100 if t_serial_e2e > 0 else 0
            print(f"  [Parallel e2e]     {t_par_e2e:7.3f}s  (saved {saved:.3f}s / {pct:.0f}%)")

    else:
        # --- Vanilla mode ---
        print_separator("Step 2: Running Qwen3-ASR (vanilla)")
        t0 = time.time()
        results = asr.transcribe(
            audio=args.audio,
            context=args.context,
            language=args.language,
            return_time_stamps=False,
        )
        asr_time = time.time() - t0

        r = results[0]
        print(f"  Language:        {r.language}")
        print(f"  Text:            {r.text}")
        print(f"  ASR time:        {asr_time:.2f}s")

    # --- Summary ---
    print()
    print("=" * 70)
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
