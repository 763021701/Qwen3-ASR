#!/usr/bin/env python3
# coding=utf-8
"""
Evaluate Meta omnilingual-asr on a Qwen3-style jsonl manifest (same protocol as
evaluation/uyghur/eval_uyghur_asr_jsonl.py): NFC normalization, ZWSP stripped,
whitespace collapsed, corpus-level WER/CER via jiwer (or built-in Levenshtein).

Uyghur (Common Voice Arabic orthography) uses fairseq2 language id uig_Arab;
see omnilingual_asr/models/wav2vec2_llama/lang_ids.py.

Dependencies:
  - Install omnilingual-asr (editable recommended), e.g.:
      pip install -e /path/to/omnilingual-asr
  - fairseq2 and other deps per that repo's README
  - pip install jiwer   (optional but recommended; matches Qwen eval script)
  - librosa (for MP3 / formats fairseq2 AudioDecoder does not support; same as Qwen stack)

Example:
  python evaluation/uyghur/baselines/eval_uyghur_asr_omnilingual_jsonl.py \
    --jsonl data/uyghur/common_voice/ug_test_qwen3.jsonl \
    --model_card omniASR_LLM_1B_v2 \
    --omnilingual_lang uig_Arab \
    --max_samples 2000 \
    --batch_size 4 \
    --output_predictions outputs/omniASR_LLM_1B_v2/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from typing import Any, Dict, List, Tuple, Union

from masr_eval_pkg import compute_wer, compute_cer, compute_sentence_wer, compute_sentence_cer
from masr_eval_pkg.normalizers import get_normalizer

import numpy as np
import torch

_ASR_TEXT_TAG = "<asr_text>"

# Wav2Vec2 stack needs enough raw samples; shorter clips collapse to length 1 after striding and crash conv1d.
_WAV2VEC2_MIN_SAMPLES_16K = 16000

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Uyghur ASR eval with omnilingual-asr (same scoring as Qwen3 jsonl eval)."
    )
    p.add_argument("--jsonl", type=str, required=True, help="Manifest: lines with audio + text.")
    p.add_argument(
        "--model_card",
        type=str,
        default="omniASR_LLM_7B_v2",
        help="omnilingual-asr hub model card (e.g. omniASR_LLM_3B_v2, omniASR_CTC_7B_v2).",
    )
    p.add_argument(
        "--omnilingual_lang",
        type=str,
        default="uig_Arab",
        help="fairseq2 language id passed to transcribe() (ignored for pure CTC cards).",
    )
    p.add_argument("--max_samples", type=int, default=0, help="If >0, only first N rows.")
    p.add_argument("--batch_size", type=int, default=4, help="Inference batch size.")
    p.add_argument(
        "--device",
        type=str,
        default="",
        help="cuda, cuda:0, cpu, etc. Empty = cuda if available else cpu.",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="bfloat16, float16, float32, or auto (bf16 on SM>=80 else fp16).",
    )
    p.add_argument(
        "--output_predictions",
        type=str,
        default="",
        help="If set, write jsonl with ref/hyp/errors per line (same schema as Qwen eval).",
    )
    p.add_argument(
        "--audio_input",
        type=str,
        choices=("auto", "path", "librosa"),
        default="auto",
        help="path=native fairseq2 decode (wav/flac only, no min-length pad — short files may crash). "
        "auto and librosa both load with librosa at 16 kHz, pad to a safe minimum, and support mp3; "
        "use either for Common Voice or mixed corpora.",
    )
    p.add_argument(
        "--min_samples_16k",
        type=int,
        default=16000,
        help="Pad 16 kHz waveform to at least max(16000, this value) samples (trailing zeros). "
        "16000≈1s; raises floor only when set higher.",
    )
    return p.parse_args()

def extract_reference_text(label: str) -> str:
    """Strip Qwen3 training prefix like 'language Uyghur<asr_text>...'."""
    s = (label or "").strip()
    if not s:
        return ""
    if _ASR_TEXT_TAG in s:
        return s.split(_ASR_TEXT_TAG, 1)[1].strip()
    return s

def load_manifest(path: str, max_samples: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows

def resolve_torch_dtype(name: str) -> torch.dtype:
    key = str(name or "auto").strip().lower()
    if key == "auto":
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
            return torch.bfloat16
        return torch.float16
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if key not in mapping:
        raise ValueError(f"Unknown dtype {name!r}. Use auto, bfloat16, float16, or float32.")
    return mapping[key]

def load_audio_waveform_dict(path: str, min_samples_16k: int) -> Dict[str, Any]:
    """Load audio as dict for ASRInferencePipeline (bypasses fairseq2 file decoder).

    Resamples to 16 kHz to match model input; pads short clips so the wav2vec2
    front-end never sees degenerate lengths after striding (conv kernel > length).
    """
    import librosa

    floor = max(_WAV2VEC2_MIN_SAMPLES_16K, max(0, int(min_samples_16k)))
    wav, _sr = librosa.load(path, sr=16000, mono=True)
    if wav.size == 0:
        wav = np.zeros(floor, dtype=np.float32)
    elif int(wav.shape[0]) < floor:
        wav = np.pad(wav, (0, floor - int(wav.shape[0])), mode="constant")
    # Time-major: 1D (T,) or (T, C). Shape (1, T) is wrong here — convert_to_mono uses dim1 as channels.
    w = torch.from_numpy(wav.astype("float32", copy=False))
    return {"waveform": w, "sample_rate": 16000}

def build_transcribe_inputs(
    batch_paths: List[str], audio_input: str, min_samples_16k: int
) -> Union[List[str], List[Dict[str, Any]]]:
    """Pipeline requires a homogeneous batch: all paths or all waveform dicts."""
    if audio_input == "path":
        return batch_paths
    # auto == librosa: always decode + pad (avoids mp3 decode errors and short-wav crashes on native path)
    return [load_audio_waveform_dict(p, min_samples_16k) for p in batch_paths]

def main() -> None:
    args = parse_args()
    rows = load_manifest(args.jsonl, args.max_samples)
    if not rows:
        print("No samples loaded.", file=sys.stderr)
        sys.exit(1)

    for i, ex in enumerate(rows):
        if "audio" not in ex or "text" not in ex:
            print(f"Line {i}: need 'audio' and 'text' fields.", file=sys.stderr)
            sys.exit(1)
        ap = ex["audio"]
        if not os.path.isfile(ap):
            print(f"Missing audio file: {ap}", file=sys.stderr)
            sys.exit(1)

    try:
        from omnilingual_asr.models.inference.pipeline import (  # type: ignore[import-not-found]
            ASRInferencePipeline,
        )
    except ImportError as exc:
        print(
            "Failed to import omnilingual_asr. Install the official repo, e.g.\n"
            "  pip install -e /path/to/omnilingual-asr",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    device = (args.device or "").strip()
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = resolve_torch_dtype(args.dtype)
    lang = (args.omnilingual_lang or "").strip() or None

    print(f"Loading omnilingual-asr model_card={args.model_card!r} device={device!r} dtype={dtype} ...")
    pipeline = ASRInferencePipeline(
        model_card=args.model_card,
        device=device,
        dtype=dtype,
    )

    refs_raw = [extract_reference_text(ex["text"]) for ex in rows]
    audios = [ex["audio"] for ex in rows]

    min_s = max(0, int(args.min_samples_16k))
    print(
        f"Transcribing {len(rows)} utterances (omnilingual_lang={lang!r}, audio_input={args.audio_input!r}, "
        f"min_samples_16k={min_s}) ..."
    )
    predictions: List[str] = []
    for start in range(0, len(audios), args.batch_size):
        batch_paths = audios[start : start + args.batch_size]
        batch_inp = build_transcribe_inputs(batch_paths, args.audio_input, min_s)
        if lang:
            langs = [lang] * len(batch_paths)
            texts = pipeline.transcribe(batch_inp, lang=langs, batch_size=len(batch_paths))
        else:
            texts = pipeline.transcribe(batch_inp, batch_size=len(batch_paths))
        predictions.extend(texts)

    if len(predictions) != len(refs_raw):
        print("Internal error: prediction count mismatch.", file=sys.stderr)
        sys.exit(1)

    normalizer = get_normalizer("ug")
    ref_wer = [normalizer.normalize_for_wer(r) for r in refs_raw]
    hyp_wer = [normalizer.normalize_for_wer(h) for h in predictions]
    ref_cer = [normalizer.normalize_for_cer(r) for r in refs_raw]
    hyp_cer = [normalizer.normalize_for_cer(h) for h in predictions]

    wer_result = compute_wer(ref_wer, hyp_wer, per_sample=True)
    cer_result = compute_cer(ref_cer, hyp_cer, per_sample=True)
    wer = wer_result["wer"]
    cer = cer_result["cer"]
    backend = "masr"
    per_sample_wer = wer_result["per_sample"]
    per_sample_cer = cer_result["per_sample"]

    print("")
    print("=== Uyghur ASR metrics (omnilingual-asr) ===")
    print(f"Model card:  {args.model_card}")
    print(f"Lang id:     {lang}")
    print(f"Samples:     {len(rows)}")
    print(f"Scoring:     UyghurNormalizer (NFC, ZWNJ/ZWJ/BOM removed, whitespace collapsed)")
    print(f"Backend:     {backend}")
    print(f"WER:         {wer * 100:.2f}%")
    print(f"CER:         {cer * 100:.2f}%")
    print("")
    print("Note: Same normalization and WER/CER definition as evaluation/uyghur/eval_uyghur_asr_jsonl.py.")

    if args.output_predictions:
        out_path = args.output_predictions
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as wf:
            for ex, rw, hw, pr, wr in zip(rows, ref_wer, hyp_wer, predictions, per_sample_wer):
                rec = {
                    "audio": ex["audio"],
                    "reference_raw": extract_reference_text(ex["text"]),
                    "hypothesis_raw": pr,
                    "reference_norm": rw,
                    "hypothesis_norm": hw,
                    "utterance_word_errors": wr["substitutions"] + wr["deletions"] + wr["insertions"],
                    "reference_word_count": max(wr["n_ref_tokens"], 1),
                }
                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote predictions to {out_path}")

if __name__ == "__main__":
    main()
