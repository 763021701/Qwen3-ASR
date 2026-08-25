#!/usr/bin/env python3
# coding=utf-8
"""
Forced alignment demo for Qwen3-ForcedAligner-0.6B.

Aligns a reference transcript against an audio file and prints per-token
timestamps (one token = one CJK character or one space-separated word).
Timestamp resolution is 80 ms.

Usage:
    # Transcript from a UTF-8 text file (default language: Chinese)
    python simple_align.py audio.wav --text-file transcript.txt

    # Inline transcript, explicit language
    python simple_align.py audio.wav --text "甚至出现交易几乎停滞的情况。" --language Chinese

    # Local model path
    python simple_align.py audio.wav --text-file transcript.txt \
        --model /path/to/Qwen3-ForcedAligner-0.6B
"""

import argparse

import torch

from qwen_asr import Qwen3ForcedAligner


DEFAULT_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
SUPPORTED_LANGUAGES = [
    "Chinese", "Cantonese", "English", "German", "Spanish", "French",
    "Italian", "Portuguese", "Russian", "Korean", "Japanese",
]


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-ForcedAligner demo: align an audio file against a transcript."
    )
    parser.add_argument("audio", help="Path to audio file (or URL)")
    parser.add_argument("--text", type=str, default=None, help="Inline transcript text")
    parser.add_argument(
        "--text-file",
        type=str,
        default=None,
        help="Path to a UTF-8 text file containing the transcript",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="Chinese",
        help="Transcript language (default: %(default)s). Supported: "
        + ", ".join(SUPPORTED_LANGUAGES),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="HF repo id or local model path (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="device_map for the model (default: cuda:0 if CUDA is available, else cpu)",
    )
    args = parser.parse_args()

    if not args.text and not args.text_file:
        parser.error("provide the transcript via --text or --text-file")
    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
    else:
        text = args.text.strip()
    if not text:
        parser.error("transcript is empty")

    device_map = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading aligner: {args.model} (device_map={device_map})")
    aligner = Qwen3ForcedAligner.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device_map
    )

    print(f"Audio    : {args.audio}")
    print(f"Language : {args.language}")
    print(f"Transcript: {text}")

    results = aligner.align(audio=args.audio, text=text, language=args.language)
    items = results[0].items
    last_end = max((it.end_time for it in items), default=0.0)

    print(f"\nAligned {len(items)} tokens, span 0.000 -> {last_end:.3f} s\n")
    print(f"{'idx':>4}  {'text':<8} {'start(s)':>9} {'end(s)':>9}")
    for i, it in enumerate(items):
        print(f"{i:>4}  {it.text:<8} {it.start_time:>9.3f} {it.end_time:>9.3f}")


if __name__ == "__main__":
    main()
