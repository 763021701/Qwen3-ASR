#!/usr/bin/env python3
# coding=utf-8
"""
Evaluate a Qwen3-ASR checkpoint on a Qwen3-style jsonl manifest (e.g. ug_dev_qwen3.jsonl).

For Uyghur (Arabic script, space-separated orthography in Common Voice):
  - WER (word error rate) is the usual headline metric and matches most ASR papers.
  - CER (character / Unicode-scalar error rate) is a useful secondary metric for script detail.

Install (recommended): pip install jiwer
Without jiwer, a small Levenshtein fallback is used (same corpus-level definition).

If reference and hypothesis use different Uyghur scripts (Arabic vs Latin), optionally align the
hypothesis with umsc before scoring (default: enabled when umsc is installed):
  pip install umsc
  # disable: add --no_umsc_script_align

Example:
  python evaluation/uyghur/eval_uyghur_asr_jsonl.py \
    --jsonl data/uyghur/common_voice/ug_test_qwen3.jsonl \
    --model /root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5 \
    --language Uyghur \
    --max_samples 2000 \
    --batch_size 8 \
    --output_predictions outputs/qwen3_asr/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import torch

from masr_eval_pkg import compute_wer, compute_cer, compute_sentence_wer, compute_sentence_cer
from masr_eval_pkg.normalizers import get_normalizer

from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import parse_asr_output

_ASR_TEXT_TAG = "<asr_text>"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Uyghur (or similar) ASR eval: WER + CER on Qwen3 jsonl.")
    p.add_argument("--jsonl", type=str, required=True, help="Manifest: lines with audio + text (Qwen3 label).")
    p.add_argument("--model", type=str, required=True, help="Checkpoint or HF model id.")
    p.add_argument(
        "--language",
        type=str,
        default="Uyghur",
        help="Force this language at inference (must match training label, e.g. Uyghur).",
    )
    p.add_argument("--max_samples", type=int, default=0, help="If >0, only first N rows.")
    p.add_argument("--batch_size", type=int, default=4, help="Inference batch size (GPU memory).")
    p.add_argument("--max_new_tokens", type=int, default=512, help="Generation cap.")
    p.add_argument("--context", type=str, default="", help="Optional context string for all utterances.")
    p.add_argument(
        "--output_predictions",
        type=str,
        default="",
        help="If set, write jsonl with ref/hyp/errors per line.",
    )
    p.add_argument(
        "--no_umsc_script_align",
        action="store_true",
        help="Do not convert hypothesis UAS/ULS via umsc to match reference script before scoring.",
    )
    p.add_argument("--device_map", type=str, default="cuda:0", help="Transformers device_map.")
    return p.parse_args()


def extract_reference_text(label: str) -> str:
    """Strip Qwen3 training prefix like 'language Uyghur<asr_text>...'."""
    s = (label or "").strip()
    if not s:
        return ""
    if _ASR_TEXT_TAG in s:
        return s.split(_ASR_TEXT_TAG, 1)[1].strip()
    return s


_LETTER_CATS = frozenset({"Lu", "Ll", "Lt", "Lm"})


def _is_arabic_script_char(ch: str) -> bool:
    """Heuristic: Arabic block / presentation forms used for Uyghur Arabic orthography."""
    o = ord(ch)
    if o == 0x0640:  # tatweel
        return False
    return (
        0x0600 <= o <= 0x06FF
        or 0x0750 <= o <= 0x077F
        or 0x08A0 <= o <= 0x08FF
        or 0xFB50 <= o <= 0xFDFF
        or 0xFE70 <= o <= 0xFEFF
    )


def _is_latin_letter_char(ch: str) -> bool:
    if _is_arabic_script_char(ch):
        return False
    return unicodedata.category(ch) in _LETTER_CATS


def count_arabic_latin_letters(s: str) -> Tuple[int, int]:
    """Count Arabic-script vs Latin letters (digits/punct ignored)."""
    a = l = 0
    for ch in s or "":
        if _is_arabic_script_char(ch):
            a += 1
        elif _is_latin_letter_char(ch):
            l += 1
    return a, l


def dominant_script_kind(arabic: int, latin: int) -> str:
    """Return 'arabic', 'latin', or 'unknown' for script-alignment decisions."""
    if arabic == 0 and latin == 0:
        return "unknown"
    if arabic > latin:
        return "arabic"
    if latin > arabic:
        return "latin"
    return "unknown"


def align_hypothesis_script_to_reference(
    hyp: str,
    ref: str,
    conv_uls_to_uas: Any,
    conv_uas_to_uls: Any,
) -> Tuple[str, str]:
    """
    If ref is Arabic and hyp is Latin -> convert hyp with ULS->UAS.
    If ref is Latin and hyp is Arabic -> convert hyp with UAS->ULS.
    Returns (possibly_converted_hyp, umsc_action) where umsc_action is
    'ULS_to_UAS', 'UAS_to_ULS', or 'none'.
    """
    ra, rl = count_arabic_latin_letters(ref)
    ha, hl = count_arabic_latin_letters(hyp)
    rk = dominant_script_kind(ra, rl)
    hk = dominant_script_kind(ha, hl)
    if rk == "arabic" and hk == "latin":
        return conv_uls_to_uas(hyp), "ULS_to_UAS"
    if rk == "latin" and hk == "arabic":
        return conv_uas_to_uls(hyp), "UAS_to_ULS"
    return hyp, "none"


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
        ap = ex["audio"]
        if not os.path.isfile(ap):
            print(f"Missing audio file: {ap}", file=sys.stderr)
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
        batch_paths = audios[start : start + args.batch_size]
        outs = model.transcribe(
            audio=batch_paths,
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

    umsc_align = not args.no_umsc_script_align
    conv_uls_to_uas: Any = None
    conv_uas_to_uls: Any = None
    if umsc_align:
        try:
            from umsc import UgMultiScriptConverter

            conv_uls_to_uas = UgMultiScriptConverter("ULS", "UAS")
            conv_uas_to_uls = UgMultiScriptConverter("UAS", "ULS")
        except ImportError:
            print("umsc not installed; scoring without UAS/ULS hypothesis alignment.", file=sys.stderr)
            print("  Install: pip install umsc   (or pass --no_umsc_script_align to silence)", file=sys.stderr)
            umsc_align = False

    hyps_raw_for_score: List[str] = []
    umsc_actions: List[str] = []
    umsc_counts = {"none": 0, "ULS_to_UAS": 0, "UAS_to_ULS": 0}
    if umsc_align and conv_uls_to_uas is not None and conv_uas_to_uls is not None:
        for pr, rr in zip(predictions, refs_raw):
            h2, action = align_hypothesis_script_to_reference(pr, rr, conv_uls_to_uas, conv_uas_to_uls)
            hyps_raw_for_score.append(h2)
            umsc_actions.append(action)
            umsc_counts[action] = umsc_counts.get(action, 0) + 1
    else:
        hyps_raw_for_score = list(predictions)
        umsc_actions = ["none"] * len(predictions)
        umsc_counts = {"none": len(predictions), "ULS_to_UAS": 0, "UAS_to_ULS": 0}

    normalizer = get_normalizer("ug")
    ref_wer = [normalizer.normalize_for_wer(r) for r in refs_raw]
    hyp_wer = [normalizer.normalize_for_wer(h) for h in hyps_raw_for_score]
    ref_cer = [normalizer.normalize_for_cer(r) for r in refs_raw]
    hyp_cer = [normalizer.normalize_for_cer(h) for h in hyps_raw_for_score]

    wer_result = compute_wer(ref_wer, hyp_wer, per_sample=True)
    cer_result = compute_cer(ref_cer, hyp_cer, per_sample=True)
    wer = wer_result["wer"]
    cer = cer_result["cer"]
    backend = "masr (UyghurNormalizer)"
    per_sample_wer = wer_result["per_sample"]
    per_sample_cer = cer_result["per_sample"]

    print("")
    print("=== Uyghur ASR metrics ===")
    print(f"Samples:     {len(rows)}")
    print(f"Scoring:     UyghurNormalizer (NFC, ZWNJ/ZWJ/BOM removed, punctuation removed, whitespace collapsed)")
    if umsc_align:
        print(
            "umsc align:  hypothesis converted to reference script when ref/hyp disagree "
            f"(ULS->UAS={umsc_counts.get('ULS_to_UAS', 0)}, UAS->ULS={umsc_counts.get('UAS_to_ULS', 0)}, "
            f"unchanged={umsc_counts.get('none', 0)})"
        )
    else:
        print("umsc align:  off")
    print(f"Backend:     {backend}")
    print(f"WER:         {wer * 100:.2f}%")
    print(f"CER:         {cer * 100:.2f}%")
    print("")
    print("Note: For Uyghur (space-separated orthography), WER is the usual primary metric;")
    print("      CER complements WER for Arabic-script spelling/detail.")

    if args.output_predictions:
        out_path = args.output_predictions
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as wf:
            for ex, rw, hw, pr, h_raw_sc, act, wr in zip(
                rows, ref_wer, hyp_wer, predictions, hyps_raw_for_score, umsc_actions, per_sample_wer
            ):
                rec = {
                    "audio": ex["audio"],
                    "reference_raw": extract_reference_text(ex["text"]),
                    "hypothesis_raw": pr,
                    "hypothesis_umsc_action": act,
                    "hypothesis_for_scoring_raw": h_raw_sc,
                    "reference_norm": rw,
                    "hypothesis_norm": hw,
                    "utterance_word_errors": wr["substitutions"] + wr["deletions"] + wr["insertions"],
                    "reference_word_count": max(wr["n_ref_tokens"], 1),
                }
                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote predictions to {out_path}")


if __name__ == "__main__":
    main()
