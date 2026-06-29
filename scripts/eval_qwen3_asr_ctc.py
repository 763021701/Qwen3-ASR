#!/usr/bin/env python3
# coding=utf-8
import argparse
import csv
import json
import math
import os
import re
import time
from typing import Dict, List

import librosa
import torch
from safetensors import safe_open
from tqdm.auto import tqdm

from finetuning.qwen3_asr_sft import (
    default_ctc_vocab_path,
    enable_ctc_training,
    load_ctc_tokenizer,
)
from qwen_asr import Qwen3ASRModel


def parse_args():
    parser = argparse.ArgumentParser("Evaluate Qwen3-ASR CTC checkpoint on metadata.csv")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--ctc_vocab_path", type=str, default="")
    parser.add_argument("--funasr_path", type=str, default="")
    parser.add_argument("--max_samples", type=int, default=0)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip())


def edit_distance(ref: str, hyp: str) -> int:
    if ref == hyp:
        return 0
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    if len(ref) < len(hyp):
        ref, hyp = hyp, ref

    previous = list(range(len(hyp) + 1))
    for i, ref_ch in enumerate(ref, start=1):
        current = [i]
        for j, hyp_ch in enumerate(hyp, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ref_ch != hyp_ch),
                )
            )
        previous = current
    return previous[-1]


def chunked(items: List[Dict[str, str]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_manifest(metadata_csv: str, max_samples: int = 0):
    rows = []
    skipped_missing_audio = 0
    skipped_empty_text = 0
    with open(metadata_csv, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            audio_path = (row.get("audio_path") or "").strip()
            ref_text = (row.get("Transcription") or "").strip()
            if not audio_path or not os.path.exists(audio_path):
                skipped_missing_audio += 1
                continue
            if not ref_text:
                skipped_empty_text += 1
                continue
            rows.append(
                {
                    "audio_path": audio_path,
                    "ref_text": ref_text,
                    "speaker_id": (row.get("SpeakerID") or "").strip(),
                }
            )
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows, skipped_missing_audio, skipped_empty_text


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


def load_ctc_checkpoint(checkpoint: str, vocab_path: str, funasr_path: str, device: str, dtype_name: str):
    model_dtype = resolve_dtype(dtype_name)
    model_kwargs = {"dtype": model_dtype}
    if device != "cpu":
        model_kwargs["device_map"] = device
    wrapper = Qwen3ASRModel.from_pretrained(checkpoint, **model_kwargs)
    ctc_tokenizer = load_ctc_tokenizer(vocab_path, funasr_path)
    enable_ctc_training(wrapper.model, vocab_path)

    decoder_state = {}
    head_state = {}
    weights_path = os.path.join(checkpoint, "model.safetensors")
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith("thinker.ctc_decoder."):
                decoder_state[key[len("thinker.ctc_decoder.") :]] = handle.get_tensor(key)
            elif key.startswith("thinker.ctc_head."):
                head_state[key[len("thinker.ctc_head.") :]] = handle.get_tensor(key)

    thinker = wrapper.model.thinker if hasattr(wrapper.model, "thinker") else wrapper.model
    thinker.ctc_decoder.load_state_dict(decoder_state, strict=True)
    thinker.ctc_head.load_state_dict(head_state, strict=True)
    wrapper.model.eval()
    return wrapper, ctc_tokenizer


def load_audio(path: str, sr: int = 16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


@torch.no_grad()
def infer_batch(wrapper: Qwen3ASRModel, ctc_tokenizer, batch_rows: List[Dict[str, str]]):
    audios = [load_audio(row["audio_path"]) for row in batch_rows]
    inputs = wrapper.processor.feature_extractor(
        audios,
        sampling_rate=16000,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_features = inputs["input_features"].to(device=wrapper.device, dtype=wrapper.dtype)
    feature_attention_mask = inputs["attention_mask"].to(device=wrapper.device)
    return wrapper.model.generate_ctc(
        input_features=input_features,
        feature_attention_mask=feature_attention_mask,
        tokenizer=ctc_tokenizer,
        return_timestamps=False,
    )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    vocab_path = args.ctc_vocab_path or default_ctc_vocab_path()
    rows, skipped_missing_audio, skipped_empty_text = load_manifest(args.metadata_csv, args.max_samples)
    if not rows:
        raise RuntimeError("No valid evaluation rows found.")

    wrapper, ctc_tokenizer = load_ctc_checkpoint(
        checkpoint=args.checkpoint,
        vocab_path=vocab_path,
        funasr_path=args.funasr_path,
        device=args.device,
        dtype_name=args.dtype,
    )

    predictions_path = os.path.join(args.output_dir, "predictions.jsonl")
    summary_path = os.path.join(args.output_dir, "summary.json")

    total_rows = len(rows)
    total_edits = 0
    total_ref_chars = 0
    exact_matches = 0

    started_at = time.time()
    with open(predictions_path, "w", encoding="utf-8") as pred_handle:
        num_batches = math.ceil(total_rows / args.batch_size)
        for batch_rows in tqdm(chunked(rows, args.batch_size), total=num_batches, desc="ctc-eval"):
            batch_outputs = infer_batch(wrapper, ctc_tokenizer, batch_rows)
            for row, output in zip(batch_rows, batch_outputs):
                hyp_text = (output.get("ctc_text") or "").strip()
                ref_norm = normalize_text(row["ref_text"])
                hyp_norm = normalize_text(hyp_text)
                char_errors = edit_distance(ref_norm, hyp_norm)
                ref_chars = len(ref_norm)
                total_edits += char_errors
                total_ref_chars += ref_chars
                exact_matches += int(ref_norm == hyp_norm)
                pred_handle.write(
                    json.dumps(
                        {
                            "audio_path": row["audio_path"],
                            "speaker_id": row["speaker_id"],
                            "ref_text": row["ref_text"],
                            "hyp_text": hyp_text,
                            "ref_norm": ref_norm,
                            "hyp_norm": hyp_norm,
                            "char_errors": char_errors,
                            "ref_chars": ref_chars,
                            "exact_match": ref_norm == hyp_norm,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    elapsed_sec = time.time() - started_at
    summary = {
        "checkpoint": args.checkpoint,
        "metadata_csv": args.metadata_csv,
        "output_dir": args.output_dir,
        "evaluated_rows": total_rows,
        "skipped_missing_audio": skipped_missing_audio,
        "skipped_empty_text": skipped_empty_text,
        "batch_size": args.batch_size,
        "device": args.device,
        "dtype": str(wrapper.dtype).replace("torch.", ""),
        "cer": (total_edits / total_ref_chars) if total_ref_chars > 0 else None,
        "char_errors": total_edits,
        "ref_chars": total_ref_chars,
        "exact_match_rate": exact_matches / total_rows,
        "elapsed_sec": elapsed_sec,
        "predictions_jsonl": predictions_path,
    }
    with open(summary_path, "w", encoding="utf-8") as summary_handle:
        json.dump(summary, summary_handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
