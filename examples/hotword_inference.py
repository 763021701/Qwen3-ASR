#!/usr/bin/env python3
# coding=utf-8
"""
Qwen3-ASR + RAG hotword inference demo.

Uses the external CTC-RAG hotword retriever library to retrieve relevant
hotwords, then injects them as context into Qwen3-ASR for final recognition.

Usage:
    python hotword_inference.py audio.wav --hotwords hot.txt

    python hotword_inference.py audio.wav \
        --hotwords hot.txt \
        --qwen3_model Qwen/Qwen3-ASR-1.7B \
        --max_hotwords 30 --ctc_topk 30

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
        help="Hotword file (one per line) or inline multi-line text",
    )
    parser.add_argument(
        "--qwen3_model", type=str, default="Qwen/Qwen3-ASR-1.7B",
        help="Qwen3-ASR model name or local path (default: Qwen/Qwen3-ASR-1.7B)",
    )
    parser.add_argument("--language", type=str, default=None, help="Force language (e.g. Chinese, English)")
    parser.add_argument("--ctc_topk", type=int, default=30, help="CTC-RAG candidate count")
    parser.add_argument("--max_hotwords", type=int, default=30, help="Max hotwords injected into context")
    parser.add_argument("--threshold", type=float, default=0.7, help="PhonemeCorrector match threshold")
    parser.add_argument("--similar_threshold", type=float, default=0.6, help="PhonemeCorrector similar threshold")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    parser.add_argument("--no_hotword", action="store_true", help="Run vanilla transcription (no RAG hotword)")
    parser.add_argument("--parallel", action="store_true", help="Run CTC retrieval and Qwen3-ASR audio encoding in parallel")
    parser.add_argument("--context", type=str, default="", help="Manual context string (vanilla mode only)")
    args = parser.parse_args()

    qwen_asr_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if qwen_asr_root not in sys.path:
        sys.path.insert(0, qwen_asr_root)

    from qwen_asr import Qwen3ASRModel, CTCRagRetriever

    hw_count = 0
    if args.hotwords and os.path.isfile(args.hotwords):
        hw_count = count_hotwords(args.hotwords)

    print("=" * 70)
    print("  Qwen3-ASR + RAG Hotword Inference Demo")
    print("=" * 70)
    print(f"  Audio:          {args.audio}")
    print(f"  Qwen3 model:    {args.qwen3_model}")
    if not args.no_hotword and args.hotwords:
        print(f"  Hotwords:       {args.hotwords} ({hw_count} hotwords)" if hw_count else f"  Hotwords:       {args.hotwords}")
        print(f"  ctc_topk:       {args.ctc_topk}")
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
        print_separator("Step 2: Loading CTC-RAG Retriever")
        t0 = time.time()
        retriever = CTCRagRetriever(
            device=args.device,
            ctc_only=True,
            threshold=args.threshold,
            similar_threshold=args.similar_threshold,
        )
        n_loaded = retriever.load_hotwords(args.hotwords)
        print(f"  CTC-RAG retriever loaded in {time.time() - t0:.1f}s ({n_loaded} hotwords)")

        # ---- Retrieval (for breakdown timing) ----
        print_separator("Step 3: CTC-RAG retrieval")

        t_retrieve_start = time.time()
        rr = retriever.retrieve(
            args.audio,
            ctc_topk=args.ctc_topk,
            max_hotwords=args.max_hotwords,
        )
        t_retrieve = time.time() - t_retrieve_start

        context_str = Qwen3ASRModel.format_hotword_context(rr.retrieved_hotwords)

        print(f"  Greedy text:        {rr.greedy_text}")
        print(f"  Integrated text:    {rr.integrated_text}")
        print(f"  Retrieval time:     {t_retrieve:.3f}s")
        print(f"  Retrieved:          {len(rr.retrieved_hotwords)} hotwords")

        if rr.retrieved_hotwords:
            print()
            for i, hw in enumerate(rr.retrieved_hotwords, 1):
                score = rr.hotword_scores.get(hw, 0.0)
                print(f"    {i:3d}. {hw:<20s} score={score:.4f}")

        if rr.details:
            audio_timings = rr.details.get("audio_timings", {})
            rag_timings = rr.details.get("timings", {})
            if audio_timings:
                print("\n  Audio timings:")
                for name, value in audio_timings.items():
                    print(f"    {name:<20s} {value * 1000.0:9.2f} ms")
            if rag_timings:
                print("\n  RAG timings:")
                for name, value in rag_timings.items():
                    print(f"    {name:<20s} {value * 1000.0:9.2f} ms")

        print(f"\n  Context string:  \"{context_str}\"")

        # ---- Serial ASR (for comparison baseline) ----
        print_separator("Step 4: Running Qwen3-ASR with hotword context (serial)")
        t_asr_start = time.time()
        results_serial = asr.transcribe_vanilla(
            audio=args.audio,
            context=context_str,
            language=args.language,
            return_time_stamps=False,
        )
        t_asr = time.time() - t_asr_start

        r = results_serial[0]
        print(f"  Language:        {r.language}")
        print(f"  Text:            {r.text}")
        print(f"  Qwen3-ASR time:  {t_asr:.3f}s  (audio encoder + LLM generate)")
        print(f"  Context used:    \"{context_str}\"")

        t_serial_e2e = t_retrieve + t_asr

        # ---- Parallel run (if requested) ----
        if args.parallel:
            print_separator("Step 5: Parallel hotword transcription")
            t_par_start = time.time()
            results_par = asr.transcribe_hotword(
                audio=args.audio,
                rag_retriever=retriever,
                language=args.language,
                return_time_stamps=False,
                ctc_topk=args.ctc_topk,
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
        print(f"  [CTC-RAG retrieve] {t_retrieve:7.3f}s")
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
