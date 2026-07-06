#!/usr/bin/env python3
# coding=utf-8
"""Chunked CTC inference for long audio.

Two modes:
  --mode fixed   : fixed-length <=chunk_sec chunks with overlap_sec overlap,
                   frame-level midpoint stitch (each chunk owns the middle of
                   the overlap; head/tail overlap/2 dropped).
  --mode silence : FSMN-VAD splits at silence; each VAD segment <=max_sec is
                   decoded whole (boundary at silence -> no word-cut); any
                   segment >max_sec falls back to the fixed-length+overlap
                   sub-chunking within that segment. Segment outputs are
                   concatenated (silence between segments is dropped, no
                   overlap needed between VAD segments).

Both recover the tail that single-pass loses after ~25s.
"""
import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from infer_qwen3_asr_ctc import (  # noqa: E402
    default_ctc_vocab_path,
    infer_one,
    load_audio,
    load_ctc_checkpoint,
    maybe_add_funasr_path,
)

SR = 16000


def parse_args():
    p = argparse.ArgumentParser("Chunked CTC inference for long audio")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--audio", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="auto")
    p.add_argument("--ctc_vocab_path", type=str, default="")
    p.add_argument("--funasr_path", type=str, default="")
    p.add_argument("--mode", choices=["fixed", "silence"], default="silence")
    p.add_argument("--chunk_sec", type=float, default=25.0,
                   help="max chunk length (fixed mode) / max VAD segment before "
                        "sub-chunk fallback (silence mode)")
    p.add_argument("--overlap_sec", type=float, default=1.0)
    p.add_argument("--compare", action="store_true",
                   help="Also run single-pass and print side-by-side")
    return p.parse_args()


def _model_device_dtype(model):
    dev = getattr(model, "device", None)
    if dev is None:
        try:
            dev = next(model.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")
    return dev, getattr(model, "dtype", torch.float32)


def _thinker_blank(model):
    thinker = model.thinker if hasattr(model, "thinker") else model
    return thinker, thinker.ctc_head.blank_id


def _collapse(frame_ids, blank_id):
    """CTC greedy collapse: drop blanks, merge consecutive repeats."""
    token_ids = []
    prev = None
    for tid in frame_ids:
        if tid == blank_id:
            prev = None
            continue
        if tid != prev:
            token_ids.append(tid)
            prev = tid
    return token_ids


def _featurize(processor, audio, device, dtype):
    inputs = processor.feature_extractor(
        [audio], sampling_rate=SR, padding=True,
        return_attention_mask=True, return_tensors="pt",
    )
    feats = inputs["input_features"].to(device=device, dtype=dtype)
    mask = inputs["attention_mask"].to(device=device)
    return feats, mask


@torch.no_grad()
def _decode_single(model, processor, audio, device, dtype, thinker, blank_id):
    """Single-pass CTC on an audio array -> collapsed token_ids."""
    feats, mask = _featurize(processor, audio, device, dtype)
    ctc_logits, ctc_input_lengths = thinker.get_ctc_logits(
        feats, feature_attention_mask=mask)
    pred = ctc_logits.argmax(dim=-1)[0]
    T = ctc_input_lengths[0].item()
    return _collapse(pred[:T].tolist(), blank_id)


@torch.no_grad()
def _chunk_token_ids(model, processor, audio, chunk_sec, overlap_sec,
                     device, dtype, thinker, blank_id, verbose=False, label=""):
    """Fixed-length+overlap frame-level midpoint stitch on an audio array.
    Returns (token_ids, owned_frames_total)."""
    duration = len(audio) / SR
    if duration <= chunk_sec + 1e-3:
        return _decode_single(model, processor, audio, device, dtype, thinker, blank_id), 0

    stride_sec = chunk_sec - overlap_sec
    starts = []
    s = 0.0
    while s < duration - 1e-6:
        starts.append(s)
        if s + chunk_sec >= duration - 1e-6:
            break
        s += stride_sec
    n = len(starts)

    frames_per_sec = None
    overlap_frames = None
    all_frame_ids = []
    owned_total = 0

    for k, start in enumerate(starts):
        end = min(start + chunk_sec, duration)
        chunk_audio = audio[int(start * SR): int(end * SR)]
        feats, mask = _featurize(processor, chunk_audio, device, dtype)
        ctc_logits, ctc_input_lengths = thinker.get_ctc_logits(
            feats, feature_attention_mask=mask)
        pred = ctc_logits.argmax(dim=-1)[0]
        T = ctc_input_lengths[0].item()
        frame_ids = pred[:T].tolist()

        if k == 0 and T > 0:
            frames_per_sec = T / (end - start)
            overlap_frames = max(1, round(overlap_sec * frames_per_sec))
            if verbose:
                print(f"[calib{label}] chunk0 T={T} dur={end - start:.2f}s "
                      f"-> {frames_per_sec:.2f} fps overlap_frames={overlap_frames}")

        half = (overlap_frames // 2) if overlap_frames else 0
        lo = 0 if k == 0 else half
        hi = T if k == n - 1 else max(T - half, lo)
        all_frame_ids.extend(frame_ids[lo:hi])
        owned_total += hi - lo
        if verbose:
            print(f"[chunk{label} {k}] audio [{start:.2f},{end:.2f}]s T={T} "
                  f"owned_frames=[{lo},{hi})")

    return _collapse(all_frame_ids, blank_id), owned_total


def load_vad(funasr_path):
    maybe_add_funasr_path(funasr_path)
    warnings.filterwarnings("ignore")
    logging.disable(logging.INFO)
    from funasr import AutoModel  # noqa: E402
    return AutoModel(model="fsmn-vad", disable_update=True,
                     disable_pbar=True, disable_log=True, device="cpu")


@torch.no_grad()
def infer_fixed(model, processor, ctc_tokenizer, audio_path,
                chunk_sec, overlap_sec, verbose=False):
    audio = load_audio(audio_path)
    duration = len(audio) / SR
    thinker, blank_id = _thinker_blank(model)
    device, dtype = _model_device_dtype(model)
    token_ids, _ = _chunk_token_ids(
        model, processor, audio, chunk_sec, overlap_sec,
        device, dtype, thinker, blank_id, verbose=verbose)
    text = (ctc_tokenizer.decode(token_ids) if token_ids else "").strip()
    return text, duration, [(0.0, duration)]


@torch.no_grad()
def infer_silence(model, processor, ctc_tokenizer, vad_model, audio_path,
                  max_sec, overlap_sec, verbose=False):
    audio = load_audio(audio_path)
    duration = len(audio) / SR
    thinker, blank_id = _thinker_blank(model)
    device, dtype = _model_device_dtype(model)

    res = vad_model.generate(input=audio_path, cache={})
    segs = res[0]["value"] if res and res[0].get("value") else []
    if verbose:
        print(f"[vad] {len(segs)} segment(s) from {duration:.2f}s audio")

    all_token_ids = []
    seg_detail = []
    speech_dur = 0.0
    for s_ms, e_ms in segs:
        s, e = s_ms / 1000.0, e_ms / 1000.0
        seg_audio = audio[int(s * SR): int(e * SR)]
        dur = e - s
        if dur <= max_sec + 1e-3:
            tids = _decode_single(model, processor, seg_audio, device, dtype, thinker, blank_id)
            mode = "single"
        else:
            tids, _ = _chunk_token_ids(
                model, processor, seg_audio, max_sec, overlap_sec,
                device, dtype, thinker, blank_id, verbose=verbose,
                label=f"@[{s:.1f},{e:.1f}]")
            mode = "sub-chunked"
        all_token_ids.extend(tids)
        speech_dur += dur
        seg_detail.append((s, e, dur, mode, len(tids)))
        if verbose:
            print(f"[seg] [{s:.2f},{e:.2f}]s dur={dur:.2f}s {mode} tokens={len(tids)}")

    text = (ctc_tokenizer.decode(all_token_ids) if all_token_ids else "").strip()
    return text, speech_dur, seg_detail


def main():
    args = parse_args()
    if not os.path.exists(args.audio):
        raise FileNotFoundError(f"Audio file not found: {args.audio}")
    vocab_path = args.ctc_vocab_path or default_ctc_vocab_path()
    model, processor, ctc_tokenizer = load_ctc_checkpoint(
        checkpoint=args.checkpoint, vocab_path=vocab_path,
        funasr_path=args.funasr_path, device=args.device, dtype_name=args.dtype,
    )

    duration = len(load_audio(args.audio)) / SR
    print(f"=== {args.audio}  ({duration:.2f}s)  mode={args.mode} ===")

    if args.compare:
        single_text = infer_one(model, processor, ctc_tokenizer, args.audio)
        print(f"\n[single-pass] chars={len(single_text)}")
        print(single_text)

    if args.mode == "fixed":
        text, cov, detail = infer_fixed(
            model, processor, ctc_tokenizer, args.audio,
            args.chunk_sec, args.overlap_sec, verbose=True)
        print(f"\n[fixed] chunk={args.chunk_sec}s overlap={args.overlap_sec}s "
              f"chars={len(text)} covered={cov:.2f}s/{duration:.2f}s")
    else:
        vad_model = load_vad(args.funasr_path)
        text, speech_dur, detail = infer_silence(
            model, processor, ctc_tokenizer, vad_model, args.audio,
            args.chunk_sec, args.overlap_sec, verbose=True)
        print(f"\n[silence] max={args.chunk_sec}s overlap={args.overlap_sec}s "
              f"chars={len(text)} speech={speech_dur:.2f}s/{duration:.2f}s "
              f"segs={len(detail)}")
        for s, e, dur, mode, ntok in detail:
            print(f"  seg [{s:.2f},{e:.2f}]s dur={dur:.2f}s {mode} tokens={ntok}")
    print(text)


if __name__ == "__main__":
    main()
