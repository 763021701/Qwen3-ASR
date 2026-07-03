#!/usr/bin/env python3
# coding=utf-8
import argparse
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qwen_asr import Qwen3ASRModel


def parse_args():
    parser = argparse.ArgumentParser("Run Qwen3-ASR inference on one audio file")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-ASR-1.7B",
                        help="HuggingFace repo id or local directory.")
    parser.add_argument("--audio", type=str, required=True,
                        help="Input audio: local path / URL / base64 data url.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--language", type=str, default="",
                        help="Force output language (e.g. Chinese, English). Empty for auto-detect.")
    parser.add_argument("--context", type=str, default="",
                        help="Optional context string fed into the prompt.")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--return_time_stamps", type=int, default=0, choices=[0, 1],
                        help="If 1, produce forced-alignment timestamps (requires --forced_aligner).")
    parser.add_argument("--forced_aligner", type=str, default="",
                        help="Forced aligner model path/repo id. Required when --return_time_stamps=1.")
    return parser.parse_args()


def resolve_dtype(dtype_name: str):
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_model(args):
    model_kwargs = {"dtype": resolve_dtype(args.dtype)}
    if args.device != "cpu":
        model_kwargs["device_map"] = args.device

    forced_aligner = args.forced_aligner or None
    if args.return_time_stamps == 1 and not forced_aligner:
        raise ValueError("--return_time_stamps=1 requires --forced_aligner to be provided.")

    return Qwen3ASRModel.from_pretrained(
        args.model_path,
        forced_aligner=forced_aligner,
        forced_aligner_kwargs=dict(model_kwargs) if forced_aligner else None,
        max_inference_batch_size=-1,
        max_new_tokens=args.max_new_tokens,
        **model_kwargs,
    )


def infer_one(asr, audio, language, context, return_time_stamps):
    results = asr.transcribe(
        audio=audio,
        context=context,
        language=language or None,
        return_time_stamps=return_time_stamps,
    )
    return results[0] if results else None


def main():
    args = parse_args()
    asr = load_model(args)
    result = infer_one(
        asr,
        audio=args.audio,
        language=args.language,
        context=args.context,
        return_time_stamps=bool(args.return_time_stamps),
    )
    if result is None:
        print("")
        return
    print(f"[language] {result.language}")
    print(f"[text] {result.text}")
    if result.time_stamps is not None and len(result.time_stamps) > 0:
        head = result.time_stamps[0]
        tail = result.time_stamps[-1]
        print(f"[ts_first] {head.text!r} {head.start_time}->{head.end_time} s")
        print(f"[ts_last ] {tail.text!r} {tail.start_time}->{tail.end_time} s")


if __name__ == "__main__":
    main()
