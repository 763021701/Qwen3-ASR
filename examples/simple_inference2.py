#!/usr/bin/env python3
# coding=utf-8
"""
Simple inference for a **local fine-tuned** Qwen3-ASR checkpoint (Transformers backend).

Supports **single language** or **comma-separated code-switch** `language` specs, same as
training/inference elsewhere (normalized via `normalize_language_spec`).

Usage:
    python examples/simple_inference2.py path/to/audio.wav \\
        --model outputs/switchlingua_cantonese_en_sft/checkpoint-455 \\
        --language English,Cantonese

    # Auto language (no prompt forcing); model emits its own language tag in output
    python examples/simple_inference2.py path/to/audio.wav --model /path/to/checkpoint-455

Checkpoint directory must be loadable by `Qwen3ASRModel.from_pretrained` (weights + tokenizer +
processor assets). If loading fails, copy `preprocessor_config.json` and `chat_template.json` from
the base `Qwen/Qwen3-ASR-*` tree into the checkpoint folder (see `.cursor/skills/.../reference.md`).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import normalize_language_spec, validate_language_spec


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR simple inference on a fine-tuned checkpoint (Transformers, multi-language spec)."
    )
    parser.add_argument("audio", type=str, help="Path to audio file (or URL)")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Local checkpoint directory (fine-tuned), e.g. outputs/.../checkpoint-455",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="",
        help="Optional forced language spec: one name or comma-separated (e.g. English,Cantonese). "
        "Empty means no forcing (model may emit language metadata in text).",
    )
    parser.add_argument(
        "--context",
        type=str,
        default="",
        help="Context string (hotwords, scene description, etc.)",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="cuda:0",
        help="Transformers device_map (default: cuda:0).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Generation cap (default: 512).",
    )
    parser.add_argument(
        "--max_inference_batch_size",
        type=int,
        default=32,
        help="Batch size limit inside transcribe (default: 32).",
    )
    args = parser.parse_args()

    model_path = os.path.abspath(args.model)
    if not os.path.isdir(model_path):
        raise SystemExit(f"Model path is not a directory: {model_path}")

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    lang_arg: str | None = None
    if args.language.strip():
        lang_arg = normalize_language_spec(args.language.strip())
        validate_language_spec(lang_arg)

    print(f"Loading checkpoint: {model_path} (dtype={dtype}, device_map={args.device_map!r})")
    model = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=args.device_map,
        max_inference_batch_size=args.max_inference_batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"Transcribing: {args.audio}")
    if lang_arg:
        print(f"Forced language spec: {lang_arg!r}")

    results = model.transcribe(
        audio=args.audio,
        context=args.context,
        language=lang_arg,
        return_time_stamps=False,
    )

    print("\n" + "=" * 60)
    print(f"Language: {results[0].language}")
    print(f"Text: {results[0].text}")
    print("=" * 60)


if __name__ == "__main__":
    main()
