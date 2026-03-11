#!/usr/bin/env python3
# coding=utf-8
"""
Simple inference script for Qwen3-ASR.

Usage:
    # Use HuggingFace model name
    python simple_inference.py audio.wav --model 1.7B
    python simple_inference.py audio.wav --model 0.6B --backend transformers
    
    # Use local model path
    python simple_inference.py audio.wav --model /path/to/Qwen3-ASR-1.7B
"""

import argparse
import os
import torch
from qwen_asr import Qwen3ASRModel


def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR Simple Inference")
    parser.add_argument("audio", type=str, help="Path to audio file (or URL)")
    parser.add_argument(
        "--model",
        type=str,
        default="1.7B",
        help="Model: '0.6B', '1.7B', or local path (default: 1.7B)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["vllm", "transformers"],
        help="Backend: vllm or transformers (default: vllm)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Force language (e.g., 'Chinese', 'English'). If not set, auto-detect.",
    )
    parser.add_argument(
        "--context",
        type=str,
        default="",
        help="Context string (hotwords, scene description, etc.)",
    )
    args = parser.parse_args()

    # Build model name or use local path
    if os.path.exists(args.model) or "/" in args.model or "\\" in args.model:
        # User provided a local path
        model_name = args.model
    else:
        # User provided a size shorthand like "0.6B" or "1.7B"
        model_name = f"Qwen/Qwen3-ASR-{args.model}"
    
    print(f"Loading model: {model_name} (backend: {args.backend})")

    # Initialize model
    if args.backend == "vllm":
        model = Qwen3ASRModel.LLM(
            model=model_name,
            gpu_memory_utilization=0.8,
            max_inference_batch_size=32,
            max_new_tokens=1024,
        )
    else:  # transformers
        model = Qwen3ASRModel.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            max_inference_batch_size=32,
            max_new_tokens=512,
        )

    print(f"Transcribing: {args.audio}")

    # Run inference
    results = model.transcribe(
        audio=args.audio,
        context=args.context,
        language=args.language,
        return_time_stamps=False,
    )

    # Print results
    print("\n" + "=" * 60)
    print(f"Language: {results[0].language}")
    print(f"Text: {results[0].text}")
    print("=" * 60)


if __name__ == "__main__":
    main()
