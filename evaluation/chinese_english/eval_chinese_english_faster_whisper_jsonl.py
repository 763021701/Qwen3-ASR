#!/usr/bin/env python3
# coding=utf-8
"""
Evaluate faster-whisper large-v3 on Chinese-English code-switching jsonl manifests.

The scoring path intentionally mirrors
`evaluation/chinese_english/eval_chinese_english_asr_jsonl.py`: Qwen3-style
labels are stripped after `<asr_text>`, then references and hypotheses are scored
with the same mixed-token MER, character CER, and sentence accuracy protocol.

Normalization and metrics are delegated to MASR_Eval_Pkg (ChineseNormalizer +
compute_cer + compute_corpus_mer).

Example:
  conda run -n scrape_hear python evaluation/chinese_english/eval_chinese_english_faster_whisper_jsonl.py \\
    --jsonl data/raw_test_eval/test_qwen3.jsonl \\
    --model /root/.cache/huggingface/hub/models--Systran--faster-whisper-large-v3 \\
    --output_predictions outputs/raw_test_eval/faster_whisper_large_v3_predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

from masr_eval_pkg import compute_cer, compute_corpus_mer, compute_mer
from masr_eval_pkg.normalizers import get_normalizer

_ASR_TEXT_TAG = "<asr_text>"
_DEFAULT_JSONL = "data/raw_test_eval/test_qwen3.jsonl"
_DEFAULT_MODEL = "/root/.cache/huggingface/hub/models--Systran--faster-whisper-large-v3"

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
    p = argparse.ArgumentParser(description="Chinese-English ASR eval for faster-whisper on Qwen3 jsonl.")
    p.add_argument("--jsonl", type=str, default=_DEFAULT_JSONL, help="Manifest with audio + Qwen3 text fields.")
    p.add_argument(
        "--model",
        type=str,
        default=_DEFAULT_MODEL,
        help="faster-whisper model id, local CTranslate2 model dir, or HF cache model dir.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='CTranslate2 device, e.g. "cuda", "cuda:0", "cpu", or "auto".',
    )
    p.add_argument(
        "--compute_type",
        type=str,
        default="float16",
        help='CTranslate2 compute type, e.g. "float16", "int8_float16", "int8", or "default".',
    )
    p.add_argument(
        "--language",
        type=str,
        default="",
        help='Optional Whisper language code such as "en", "zh", or "yue". Empty = auto detect.',
    )
    p.add_argument("--task", type=str, default="transcribe", choices=("transcribe", "translate"))
    p.add_argument("--beam_size", type=int, default=5)
    p.add_argument("--best_of", type=int, default=5)
    p.add_argument("--max_samples", type=int, default=0, help="If >0, only evaluate the first N rows.")
    p.add_argument(
        "--condition_on_previous_text",
        action="store_true",
        help="Forward previous segment text within each audio file. Default is off for independent clips.",
    )
    p.add_argument(
        "--vad_filter",
        action="store_true",
        help="Enable faster-whisper VAD filtering before decoding.",
    )
    p.add_argument("--initial_prompt", type=str, default="", help="Optional initial prompt passed to Whisper.")
    p.add_argument(
        "--local_files_only",
        type=int,
        default=1,
        choices=(0, 1),
        help="When loading a Hugging Face model id, require cached files only.",
    )
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
    """Data-source-specific cleaning before MASR normalization."""
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


def resolve_model_path(model: str) -> str:
    path = Path(model).expanduser()
    if not path.exists():
        return model
    if (path / "model.bin").is_file():
        return str(path)

    snapshots = path / "snapshots"
    if snapshots.is_dir():
        candidates = [p for p in snapshots.iterdir() if p.is_dir() and (p / "model.bin").is_file()]
        if not candidates:
            return str(path)
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(candidates[0])

    return str(path)


def normalize_device(device: str) -> tuple:
    value = (device or "auto").strip()
    if value.startswith("cuda:"):
        return "cuda", int(value.split(":", 1)[1])
    return value, 0


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

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SystemExit("Install faster-whisper in this environment before running this script.") from exc

    model_path = resolve_model_path(args.model)
    device, device_index = normalize_device(args.device)
    language = args.language.strip() or None
    initial_prompt = args.initial_prompt or None

    print(f"Loading faster-whisper model from {model_path!r} ...")
    model = WhisperModel(
        model_path,
        device=device,
        device_index=device_index,
        compute_type=args.compute_type,
        local_files_only=bool(args.local_files_only),
    )

    refs_raw = [extract_reference_text(ex["text"]) for ex in rows]
    predictions: List[str] = []
    detected_languages: List[str] = []
    language_probabilities: List[float] = []

    print(f"Evaluating {len(rows)} utterances (language={language!r}) ...")
    for idx, ex in enumerate(rows, start=1):
        segments, info = model.transcribe(
            ex["audio"],
            language=language,
            task=args.task,
            beam_size=args.beam_size,
            best_of=args.best_of,
            vad_filter=args.vad_filter,
            initial_prompt=initial_prompt,
            condition_on_previous_text=args.condition_on_previous_text,
        )
        hyp = "".join(segment.text for segment in segments).strip()
        predictions.append(hyp)
        detected_languages.append(getattr(info, "language", "") or "")
        language_probabilities.append(float(getattr(info, "language_probability", 0.0) or 0.0))
        if idx % 10 == 0:
            print(f"[{idx}/{len(rows)}] {os.path.basename(ex['audio'])}: {hyp[:80]}...")

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
    print("=== Faster-Whisper Chinese-English ASR metrics ===")
    print(f"Samples:             {len(rows)}")
    print(
        "Scoring:             pre_clean + ChineseNormalizer (NFKC, number ITN, punctuation removal), "
        "tokens = English words + numbers + individual Han chars"
    )
    print(f"Model:               {model_path}")
    print(f"Device:              {args.device}")
    print(f"Compute type:        {args.compute_type}")
    print(f"Whisper language:    {language!r}")
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
            for ex, rr, hh, rm, hm, sm, sc, lang_det, lang_prob in zip(
                rows,
                refs_raw,
                predictions,
                ref_mer,
                hyp_mer,
                per_sample_mer,
                per_sample_cer_list,
                detected_languages,
                language_probabilities,
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
                    "detected_language": lang_det,
                    "language_probability": lang_prob,
                }
                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote predictions to {args.output_predictions}")


if __name__ == "__main__":
    main()
