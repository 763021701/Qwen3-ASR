#!/usr/bin/env python3
# coding=utf-8
"""
Evaluate Qwen3-ASR on Chinese-English code-switching jsonl manifests.

The scorer follows docs/normalize_label.md for mixed Chinese/English ASR:
Unicode normalization, punctuation removal, lowercase Latin text, Chinese
number ITN to Arabic digits, and script-aware tokenization. Mixed token error
rate (MER) is computed over English word tokens, number tokens, and individual
Han characters, so script boundaries such as "abc中文123" cannot collapse into
one whitespace token.

Normalization and metrics are delegated to MASR_Eval_Pkg (ChineseNormalizer +
compute_cer + compute_corpus_mer). Data-source-specific cleaning (markup,
noise annotations, speaker tags, timestamps) is kept as pre-processing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from typing import Any, Dict, List

import torch

from masr_eval_pkg import compute_cer, compute_corpus_mer, compute_mer
from masr_eval_pkg.normalizers import get_normalizer

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import parse_asr_output

_ASR_TEXT_TAG = "<asr_text>"

# --- Data-source-specific cleaning patterns (kept as pre-processing) ---
_ZW_RE = re.compile(r"[​-‏‪-‮⁠﻿]")
_MARKUP_RE = re.compile(r"<[^>]+>")
_SPEAKER_RE = re.compile(r"\b(?:speaker|spk)\s*\d+\b", re.IGNORECASE)
_TIMESTAMP_RE = re.compile(
    r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?\b|\b\d+(?:\.\d+)?\s*(?:s|sec|secs|seconds)\b",
    re.IGNORECASE,
)
_NOISE_ANNOTATION_RE = re.compile(
    r"\[(?:noise|music|laughter|laugh|silence|inaudible|background|applause|cough|breath)[^\]]*\]",
    re.IGNORECASE,
)
_LETTER_DOT_RE = re.compile(r"(?<=[A-Za-z])\.(?=[A-Za-z])")

_ZH_CONVERT_MAP = {
    "off": "none",
    "to_simplified": "t2s",
    "to_traditional": "s2t",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chinese-English code-switching ASR eval for Qwen3 jsonl.")
    p.add_argument("--jsonl", type=str, required=True, help="Manifest with audio + Qwen3 text fields.")
    p.add_argument("--model", type=str, required=True, help="Checkpoint path or HuggingFace model id.")
    p.add_argument(
        "--language",
        type=str,
        default="None",
        help='Forced inference language. Use "None" for language None<asr_text> prompting.',
    )
    p.add_argument("--max_samples", type=int, default=0, help="If >0, only evaluate the first N rows.")
    p.add_argument("--batch_size", type=int, default=4, help="Inference batch size.")
    p.add_argument("--max_new_tokens", type=int, default=512, help="Generation cap.")
    p.add_argument("--context", type=str, default="", help="Optional context string for all utterances.")
    p.add_argument("--device_map", type=str, default="cuda:0", help="Transformers device_map.")
    p.add_argument(
        "--hanzi_script_norm",
        type=str,
        default="off",
        choices=("off", "to_simplified", "to_traditional"),
        help="Optional Han script normalization before scoring.",
    )
    p.add_argument(
        "--output_predictions",
        type=str,
        default="",
        help="If set, write per-utterance prediction and scoring details as jsonl.",
    )
    return p.parse_args()


def extract_reference_text(label: str) -> str:
    s = (label or "").strip()
    if not s:
        return ""
    if _ASR_TEXT_TAG in s:
        return s.split(_ASR_TEXT_TAG, 1)[1].strip()
    return s


def pre_clean(text: str, hanzi_converter) -> str:
    """Data-source-specific cleaning before MASR normalization.

    Removes zero-width chars, markup tags, noise annotations, speaker labels,
    timestamps, letter-dots, and control characters. Han script conversion
    (if enabled) is applied here so the normalizer does not need OpenCC.
    """
    s = unicodedata.normalize("NFKC", (text or "").strip())
    s = _ZW_RE.sub("", s)
    s = _MARKUP_RE.sub(" ", s)
    s = _NOISE_ANNOTATION_RE.sub(" ", s)
    s = _SPEAKER_RE.sub(" ", s)
    s = _TIMESTAMP_RE.sub(" ", s)
    s = _LETTER_DOT_RE.sub("", s)
    s = "".join(" " if unicodedata.category(ch) in ("Cc", "Cf") else ch for ch in s)
    if hanzi_converter is not None:
        s = hanzi_converter(s)
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
        if not os.path.isfile(ex["audio"]):
            print(f"Missing audio file: {ex['audio']}", file=sys.stderr)
            sys.exit(1)

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    print(f"Loading model from {args.model!r} ...")
    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=args.device_map,
        max_inference_batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    refs_raw = [extract_reference_text(ex["text"]) for ex in rows]
    audios = [ex["audio"] for ex in rows]
    lang = args.language.strip() or None

    print(f"Evaluating {len(rows)} utterances (language={lang!r}) ...")
    predictions: List[str] = []
    for start in range(0, len(audios), args.batch_size):
        outs = model.transcribe(
            audio=audios[start : start + args.batch_size],
            context=args.context,
            language=lang,
            return_time_stamps=False,
        )
        for o in outs:
            _, txt = parse_asr_output(o.text, user_language=lang)
            predictions.append(txt)

    if len(predictions) != len(refs_raw):
        print("Internal error: prediction count mismatch.", file=sys.stderr)
        sys.exit(1)

    # --- Hanzi converter (optional, applied in pre_clean) ---
    hanzi_converter = None
    if args.hanzi_script_norm != "off":
        try:
            from opencc import OpenCC  # type: ignore
        except ImportError:
            raise SystemExit("Install opencc-python-reimplemented for Han script normalization.")
        cc_mode = "t2s" if args.hanzi_script_norm == "to_simplified" else "s2t"
        hanzi_converter = OpenCC(cc_mode).convert

    # --- Pre-clean data-source artifacts ---
    refs_cleaned = [pre_clean(r, hanzi_converter) for r in refs_raw]
    hyps_cleaned = [pre_clean(h, None) for h in predictions]

    # --- Normalization via MASR_Eval_Pkg ChineseNormalizer ---
    zh_convert = _ZH_CONVERT_MAP[args.hanzi_script_norm]
    normalizer = get_normalizer(
        "zh",
        zh_convert=zh_convert,
        number_normalize="to_arabic",
    )

    # CER: space-free character-level
    ref_cer = [normalizer.normalize_for_cer(r) for r in refs_cleaned]
    hyp_cer = [normalizer.normalize_for_cer(h) for h in hyps_cleaned]
    cer_result = compute_cer(ref_cer, hyp_cer, per_sample=True)

    # MER: mixed-token error rate
    ref_mer = [normalizer.normalize_for_mer(r) for r in refs_cleaned]
    hyp_mer = [normalizer.normalize_for_mer(h) for h in hyps_cleaned]
    mer_corpus = compute_corpus_mer(ref_mer, hyp_mer, text_preprocessed=True)

    # Per-sample MER for output
    per_sample_mer = [
        compute_mer(rm, hm, text_preprocessed=True)
        for rm, hm in zip(ref_mer, hyp_mer)
    ]
    per_sample_cer_list = cer_result["per_sample"]

    # Sentence accuracy from MER token match
    exact_matches = sum(
        1 for sm in per_sample_mer if sm.ref_tokens == sm.hyp_tokens
    )

    mer_val = mer_corpus.mer
    cer_val = cer_result["cer"]
    token_errors = mer_corpus.total_substitutions + mer_corpus.total_deletions + mer_corpus.total_insertions
    char_errors = cer_result["substitutions"] + cer_result["deletions"] + cer_result["insertions"]

    print("")
    print("=== Chinese-English ASR metrics ===")
    print(f"Samples:             {len(rows)}")
    print(
        "Scoring:             pre_clean + ChineseNormalizer (NFKC, number ITN, punctuation removal), "
        "tokens = English words + numbers + individual Han chars"
    )
    print(f"Hanzi script norm:   {args.hanzi_script_norm}")
    print(f"MER:                 {mer_val * 100:.2f}%")
    print(f"CER:                 {cer_val * 100:.2f}%")
    print(f"Sentence Accuracy:   {exact_matches / max(len(ref_mer), 1) * 100:.2f}%")
    print(f"Token errors:        {token_errors} / {mer_corpus.total_n_ref_tokens}")
    print(f"Char errors:         {char_errors} / {cer_result['n_ref_chars']}")

    if args.output_predictions:
        parent = os.path.dirname(args.output_predictions)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.output_predictions, "w", encoding="utf-8") as wf:
            for ex, rr, hh, rm, hm, sm, sc in zip(
                rows,
                refs_raw,
                predictions,
                ref_mer,
                hyp_mer,
                per_sample_mer,
                per_sample_cer_list,
            ):
                rec = {
                    "audio": ex["audio"],
                    "reference_raw": rr,
                    "hypothesis_raw": hh,
                    "reference_norm": rm,
                    "hypothesis_norm": hm,
                    "reference_tokens": sm.ref_tokens,
                    "hypothesis_tokens": sm.hyp_tokens,
                    "utterance_token_errors": sm.substitutions + sm.deletions + sm.insertions,
                    "reference_token_count": max(sm.n_ref_tokens, 1),
                    "utterance_char_errors": sc["substitutions"] + sc["deletions"] + sc["insertions"],
                    "reference_char_count": max(sc["n_ref_chars"], 1),
                    "exact_match": sm.ref_tokens == sm.hyp_tokens,
                }
                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote predictions to {args.output_predictions}")


if __name__ == "__main__":
    main()
