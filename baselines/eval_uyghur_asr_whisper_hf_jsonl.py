#!/usr/bin/env python3
# coding=utf-8
"""
Evaluate a HuggingFace Whisper checkpoint on a Qwen3-style jsonl manifest (same protocol as
finetuning/eval_uyghur_asr_jsonl.py): NFC normalization, ZWSP stripped, whitespace collapsed,
corpus-level WER/CER via jiwer (or built-in Levenshtein).

Default model: ixxan/whisper-small-uyghur-common-voice
  https://huggingface.co/ixxan/whisper-small-uyghur-common-voice

Note: openai/whisper-small tokenizer does not list Uyghur for forced_decoder_ids; for this
fine-tuned checkpoint we default to task-only prefix (transcribe). Set --language to a Whisper
supported name (e.g. turkish) only if you intentionally want that prefix (usually not for ug).

Batched eval uses return_attention_mask=True so padded mels are masked (required for correct WER).

Dependencies: transformers, torch, librosa; optional jiwer (recommended).

Example:
  python baselines/eval_uyghur_asr_whisper_hf_jsonl.py \\
    --jsonl data/ug_test_qwen3.jsonl \\
    --model ixxan/whisper-small-uyghur-common-voice \\
    --batch_size 8 \\
    --max_samples 500
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_ASR_TEXT_TAG = "<asr_text>"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Uyghur ASR eval with HF Whisper (same scoring as Qwen3 jsonl eval)."
    )
    p.add_argument("--jsonl", type=str, required=True, help="Manifest: lines with audio + text.")
    p.add_argument(
        "--model",
        type=str,
        default="ixxan/whisper-small-uyghur-common-voice",
        help="HF model id or local path (WhisperForConditionalGeneration).",
    )
    p.add_argument(
        "--language",
        type=str,
        default="",
        help="If non-empty, Whisper forced decoder language (lowercase, must be in tokenizer list). "
        "Empty = task-only prefix (recommended for ixxan Uyghur fine-tune on whisper-small).",
    )
    p.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=("transcribe", "translate"),
        help="Whisper task token for decoder prefix.",
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
        help="float16, bfloat16, float32, or auto (fp16 on GPU else fp32).",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=445,
        help="Max generated tokens. Whisper max_target_positions is 448 for small; decoder prefix "
        "(forced task/language) consumes a few slots, so 448 + prefix exceeds the limit — default 445.",
    )
    p.add_argument("--num_beams", type=int, default=5, help="Beam size (1 = greedy).")
    p.add_argument(
        "--output_predictions",
        type=str,
        default="",
        help="If set, write jsonl with ref/hyp/errors per line (same schema as Qwen eval).",
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


_ZW_RE = re.compile(r"[\u200c\u200d\ufeff]")


def normalize_for_scoring(s: str) -> str:
    """NFC, drop common zero-width chars, collapse whitespace."""
    s = unicodedata.normalize("NFC", (s or "").strip())
    s = _ZW_RE.sub("", s)
    s = " ".join(s.split())
    return s


def _levenshtein_1d(a: Sequence[Any], b: Sequence[Any]) -> int:
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def corpus_wer_cer_fallback(refs: List[str], hyps: List[str]) -> Tuple[float, float]:
    """Corpus-level WER/CER via summed edit distance / summed reference length."""
    w_err, w_den = 0, 0
    c_err, c_den = 0, 0
    for r, h in zip(refs, hyps):
        rw = r.split()
        hw = h.split()
        if len(rw) == 0 and len(hw) == 0:
            pass
        elif len(rw) == 0:
            w_err += len(hw)
            w_den += max(len(hw), 1)
        else:
            w_err += _levenshtein_1d(rw, hw)
            w_den += len(rw)

        rc = list(r)
        hc = list(h)
        if len(rc) == 0 and len(hc) == 0:
            pass
        elif len(rc) == 0:
            c_err += len(hc)
            c_den += max(len(hc), 1)
        else:
            c_err += _levenshtein_1d(rc, hc)
            c_den += len(rc)
    return w_err / max(w_den, 1), c_err / max(c_den, 1)


def corpus_wer_cer_jiwer(refs: List[str], hyps: List[str]) -> Tuple[float, float]:
    import jiwer

    return float(jiwer.wer(refs, hyps)), float(jiwer.cer(refs, hyps))


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


def resolve_torch_dtype(name: str, device: str) -> torch.dtype:
    key = str(name or "auto").strip().lower()
    if key == "auto":
        if device.startswith("cuda") and torch.cuda.is_available():
            return torch.float16
        return torch.float32
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if key not in mapping:
        raise ValueError(f"Unknown dtype {name!r}. Use auto, float16, bfloat16, or float32.")
    return mapping[key]


def forced_decoder_prompt_ids(processor: Any, language: str, task: str) -> List[Tuple[int, int]]:
    lang = (language or "").strip().lower()
    if lang:
        return list(processor.get_decoder_prompt_ids(language=lang, task=task))
    return list(processor.get_decoder_prompt_ids(task=task))


def cap_whisper_max_new_tokens(
    model: Any, forced_ids: List[Tuple[int, int]], requested: int
) -> Tuple[int, int]:
    """Whisper: decoder_input_ids length + max_new_tokens must be <= max_target_positions."""
    max_pos = int(getattr(model.config, "max_target_positions", 448))
    # Prefix length matches Transformers check (task-only ~3 tokens with forced_decoder_ids length 2).
    prefix_reserve = max(3, len(forced_ids) + 1)
    allowed = max(1, max_pos - prefix_reserve)
    eff = max(1, min(int(requested), allowed))
    return eff, max_pos


def load_batch_waveforms(paths: List[str]) -> List[np.ndarray]:
    import librosa

    out: List[np.ndarray] = []
    for p in paths:
        wav, _ = librosa.load(p, sr=16000, mono=True)
        out.append(np.asarray(wav, dtype=np.float32))
    return out


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
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
    except ImportError as exc:
        print("Install transformers: pip install transformers", file=sys.stderr)
        raise SystemExit(1) from exc

    device = (args.device or "").strip()
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = resolve_torch_dtype(args.dtype, device)

    print(f"Loading Whisper processor/model from {args.model!r} (dtype={dtype}) ...")
    processor = WhisperProcessor.from_pretrained(args.model)
    try:
        model = WhisperForConditionalGeneration.from_pretrained(args.model, dtype=dtype)
    except TypeError:
        model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device)
    model.eval()

    try:
        forced_ids = forced_decoder_prompt_ids(processor, args.language, args.task)
    except ValueError as exc:
        print(f"Invalid --language / --task for tokenizer: {exc}", file=sys.stderr)
        sys.exit(1)

    max_new_eff, max_pos = cap_whisper_max_new_tokens(model, forced_ids, args.max_new_tokens)
    if max_new_eff < int(args.max_new_tokens):
        print(
            f"Clamping max_new_tokens: {args.max_new_tokens} -> {max_new_eff} "
            f"(max_target_positions={max_pos}, decoder prefix reserve).",
            file=sys.stderr,
        )

    refs_raw = [extract_reference_text(ex["text"]) for ex in rows]
    audios = [ex["audio"] for ex in rows]

    lang_note = args.language.strip() or "(task-only, no language token)"
    print(
        f"Evaluating {len(rows)} utterances (forced decoder: language={lang_note!r}, task={args.task!r}) ..."
    )

    predictions: List[str] = []
    with torch.inference_mode():
        for start in range(0, len(audios), args.batch_size):
            batch_paths = audios[start : start + args.batch_size]
            waves = load_batch_waveforms(batch_paths)
            # Batched inference must request attention_mask; otherwise padded mel frames are not
            # masked and transcriptions are corrupted (very high WER). See feature_extraction_whisper.py.
            inputs = processor(
                waves,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            )
            input_features = inputs.input_features.to(device, dtype=dtype)
            attn: Optional[torch.Tensor] = None
            if getattr(inputs, "attention_mask", None) is not None:
                attn = inputs.attention_mask.to(device)
            else:
                print(
                    "Warning: processor did not return attention_mask; batched scores may be unreliable.",
                    file=sys.stderr,
                )

            gen_kw: Dict[str, Any] = {
                "forced_decoder_ids": forced_ids,
                "max_new_tokens": max_new_eff,
                "num_beams": int(args.num_beams),
            }
            if attn is not None:
                gen_kw["attention_mask"] = attn

            ids = model.generate(input_features, **gen_kw)
            texts = processor.batch_decode(ids, skip_special_tokens=True)
            predictions.extend(t.strip() for t in texts)

    if len(predictions) != len(refs_raw):
        print("Internal error: prediction count mismatch.", file=sys.stderr)
        sys.exit(1)

    refs = [normalize_for_scoring(r) for r in refs_raw]
    hyps = [normalize_for_scoring(h) for h in predictions]

    try:
        wer, cer = corpus_wer_cer_jiwer(refs, hyps)
        backend = "jiwer"
    except ImportError:
        wer, cer = corpus_wer_cer_fallback(refs, hyps)
        backend = "builtin_levenshtein"

    print("")
    print("=== Uyghur ASR metrics (HuggingFace Whisper) ===")
    print(f"Model:       {args.model}")
    print(f"Samples:     {len(rows)}")
    print(f"Scoring:     NFC, ZWSP removed, whitespace collapsed")
    print(f"Backend:     {backend}")
    print(f"WER:         {wer * 100:.2f}%")
    print(f"CER:         {cer * 100:.2f}%")
    print("")
    print("Note: Same normalization and WER/CER definition as finetuning/eval_uyghur_asr_jsonl.py.")

    if args.output_predictions:
        out_path = args.output_predictions
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as wf:
            for ex, rr, hh, pr in zip(rows, refs, hyps, predictions):
                rw, hw = rr.split(), hh.split()
                if len(rw) == 0:
                    dist = len(hw)
                else:
                    dist = _levenshtein_1d(rw, hw)
                rec = {
                    "audio": ex["audio"],
                    "reference_raw": extract_reference_text(ex["text"]),
                    "hypothesis_raw": pr,
                    "reference_norm": rr,
                    "hypothesis_norm": hh,
                    "utterance_word_errors": dist,
                    "reference_word_count": max(len(rw), 1) if len(rw) == 0 else len(rw),
                }
                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote predictions to {out_path}")


if __name__ == "__main__":
    main()
