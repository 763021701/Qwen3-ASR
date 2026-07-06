#!/usr/bin/env python3
# coding=utf-8
import argparse
import sys
import time
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import (
    SAMPLE_RATE,
    normalize_audios,
    split_audio_into_chunks,
)


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
    parser.add_argument("--mode", choices=["full", "chunk"], default="full",
                        help="full = one AR pass on the whole audio (default, no split since "
                             "MAX_ASR_INPUT_SECONDS=1200); chunk = split into <=chunk_sec pieces "
                             "via the built-in low-energy-boundary splitter, AR per chunk, concat text.")
    parser.add_argument("--chunk_sec", type=float, default=25.0,
                        help="Target max chunk duration in seconds (chunk mode only).")
    parser.add_argument("--compare", action="store_true",
                        help="Run both full and chunk modes, print results + timing side-by-side.")
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


def run_full(asr, audio, language, context, return_time_stamps):
    t0 = time.perf_counter()
    res = infer_one(asr, audio, language, context, return_time_stamps)
    return res, time.perf_counter() - t0


def run_chunked(asr, audio_path, chunk_sec, language, context, verbose=False):
    """Split via built-in low-energy-boundary splitter, AR per chunk, concat text."""
    wav = normalize_audios(audio_path)[0]
    parts = split_audio_into_chunks(wav, SAMPLE_RATE, chunk_sec)
    texts, detail = [], []
    t0 = time.perf_counter()
    for i, (cwav, offset) in enumerate(parts):
        res = infer_one(asr, (cwav, SAMPLE_RATE), language, context, False)
        txt = res.text if res else ""
        texts.append(txt)
        dur = len(cwav) / SAMPLE_RATE
        detail.append((i, offset, dur, len(txt)))
        if verbose:
            print(f"  [chunk {i}] offset={offset:.2f}s dur={dur:.2f}s chars={len(txt)}")
    return "".join(texts), time.perf_counter() - t0, detail


def main():
    args = parse_args()
    asr = load_model(args)
    rts = bool(args.return_time_stamps)

    if args.compare or args.mode == "full":
        full_res, full_dt = run_full(asr, args.audio, args.language, args.context, rts)
        full_text = full_res.text if full_res else ""
        print(f"[full]  time={full_dt:.2f}s chars={len(full_text)} "
              f"lang={full_res.language if full_res else '?'}")
        if not args.compare:
            if full_res is not None:
                print(f"[language] {full_res.language}")
                print(f"[text] {full_res.text}")
                if full_res.time_stamps and len(full_res.time_stamps) > 0:
                    head, tail = full_res.time_stamps[0], full_res.time_stamps[-1]
                    print(f"[ts_first] {head.text!r} {head.start_time}->{head.end_time} s")
                    print(f"[ts_last ] {tail.text!r} {tail.start_time}->{tail.end_time} s")
            else:
                print("")
            return

    if args.compare or args.mode == "chunk":
        chunk_text, chunk_dt, detail = run_chunked(
            asr, args.audio, args.chunk_sec, args.language, args.context, verbose=True)
        print(f"\n[chunk] chunk_sec={args.chunk_sec}s time={chunk_dt:.2f}s "
              f"chars={len(chunk_text)} nchunks={len(detail)}")
        print(chunk_text)


if __name__ == "__main__":
    main()
