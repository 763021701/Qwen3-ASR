#!/usr/bin/env python3
# coding=utf-8
"""
Evaluate a Qwen3-ASR checkpoint on a Qwen3-style Cantonese jsonl manifest.

Why a separate script from Uyghur eval:
  - Cantonese transcripts here are Han-character strings, usually without word segmentation.
  - Therefore CER (character error rate) is the primary metric.
  - WER based on whitespace tokenization is usually not meaningful, so this script reports:
      * CER (primary)
      * Sentence accuracy / exact match rate

Normalization and metrics are delegated to MASR_Eval_Pkg (ChineseNormalizer + compute_cer).
Recommended: pip install opencc-python-reimplemented cn2an (for full ChineseNormalizer features).

Example:
  python evaluation/cantonese/eval_cantonese_asr_jsonl.py \
    --jsonl data/cantonese/common_voice_yue/cv_yue_test_qwen3.jsonl \
    --model /path/to/checkpoint \
    --batch_size 8 \
    --output_predictions outputs/cantonese_eval/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List

import torch

from masr_eval_pkg import compute_cer, compute_sentence_cer
from masr_eval_pkg.metrics.levenshtein import levenshtein_align
from masr_eval_pkg.normalizers import get_normalizer

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import parse_asr_output

_ASR_TEXT_TAG = "<asr_text>"

_ZH_CONVERT_MAP = {
    "off": "none",
    "to_traditional": "s2t",
    "to_simplified": "t2s",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cantonese ASR eval on Qwen3 jsonl: CER + sentence accuracy.")
    p.add_argument("--jsonl", type=str, required=True, help="Manifest: lines with audio + text (Qwen3 label).")
    p.add_argument("--model", type=str, required=True, help="Checkpoint or HF model id.")
    p.add_argument(
        "--language",
        type=str,
        default="Cantonese",
        help="Force this language at inference (default: Cantonese).",
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
        "--keep_whitespace",
        action="store_true",
        help="Keep whitespace for CER. Default is to remove all whitespace before scoring.",
    )
    p.add_argument(
        "--hanzi_script_norm",
        type=str,
        default="to_traditional",
        choices=("off", "to_traditional", "to_simplified"),
        help="Normalize Hanzi script before scoring. Default: convert both ref/hyp to Traditional Chinese.",
    )
    p.add_argument("--device_map", type=str, default="cuda:0", help="Transformers device_map.")
    return p.parse_args()


def extract_reference_text(label: str) -> str:
    """Strip Qwen3 training prefix like 'language Cantonese<asr_text>...'."""
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

    # --- Normalization via MASR_Eval_Pkg ChineseNormalizer ---
    zh_convert = _ZH_CONVERT_MAP[args.hanzi_script_norm]
    normalizer = get_normalizer(
        "zh",
        zh_convert=zh_convert,
        number_normalize="to_arabic",
    )

    if args.keep_whitespace:
        # Spaces count as characters: use normalize() + collapse whitespace, then manual CER
        def _norm_ws(s: str) -> str:
            n = normalizer.normalize(s)
            return re.sub(r"\s+", " ", n).strip()

        refs = [_norm_ws(r) for r in refs_raw]
        hyps = [_norm_ws(h) for h in predictions]

        total_cer_errors = 0
        total_cer_n = 0
        per_sample_errors: List[int] = []
        for rr, hh in zip(refs, hyps):
            rc, hc = list(rr), list(hh)
            if len(rc) == 0 and len(hc) == 0:
                per_sample_errors.append(0)
                continue
            if len(rc) == 0:
                total_cer_errors += len(hc)
                total_cer_n += max(len(hc), 1)
                per_sample_errors.append(len(hc))
            else:
                s, d, ins, _ = levenshtein_align(rc, hc)
                err = s + d + ins
                total_cer_errors += err
                total_cer_n += len(rc)
                per_sample_errors.append(err)
        cer = total_cer_errors / max(total_cer_n, 1)
        backend = "masr_levenshtein"
        ref_cer = refs
        hyp_cer = hyps
    else:
        # Standard CER: remove all whitespace
        ref_cer = [normalizer.normalize_for_cer(r) for r in refs_raw]
        hyp_cer = [normalizer.normalize_for_cer(h) for h in predictions]
        cer_result = compute_cer(ref_cer, hyp_cer, per_sample=True)
        cer = cer_result["cer"]
        backend = "masr"
        per_sample_cer = cer_result["per_sample"]

    exact_matches = sum(1 for r, h in zip(ref_cer, hyp_cer) if r == h)
    sentence_accuracy = exact_matches / max(len(ref_cer), 1)

    print("")
    print("=== Cantonese ASR metrics ===")
    print(f"Samples:            {len(rows)}")
    print(
        "Scoring:            "
        f"ChineseNormalizer (NFKC, OpenCC={zh_convert}, number ITN), "
        f"Hanzi={args.hanzi_script_norm}, whitespace="
        f"{'kept' if args.keep_whitespace else 'removed'}"
    )
    print(f"Backend:            {backend}")
    print(f"CER:                {cer * 100:.2f}%")
    print(f"Sentence Accuracy:  {sentence_accuracy * 100:.2f}%")
    print("")
    print("Note: For Cantonese Han-character transcripts, CER is the primary metric.")
    print("      WER based on whitespace tokenization is usually not meaningful and is omitted.")

    if args.output_predictions:
        out_path = args.output_predictions
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as wf:
            if args.keep_whitespace:
                for ex, pr, rr, hh, dist in zip(rows, predictions, ref_cer, hyp_cer, per_sample_errors):
                    rec = {
                        "audio": ex["audio"],
                        "reference_raw": extract_reference_text(ex["text"]),
                        "hypothesis_raw": pr,
                        "reference_norm": rr,
                        "hypothesis_norm": hh,
                        "utterance_char_errors": dist,
                        "reference_char_count": max(len(rr), 1),
                        "exact_match": rr == hh,
                    }
                    wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            else:
                for ex, pr, rc, hc, sc in zip(rows, predictions, ref_cer, hyp_cer, per_sample_cer):
                    rec = {
                        "audio": ex["audio"],
                        "reference_raw": extract_reference_text(ex["text"]),
                        "hypothesis_raw": pr,
                        "reference_norm": rc,
                        "hypothesis_norm": hc,
                        "utterance_char_errors": sc["substitutions"] + sc["deletions"] + sc["insertions"],
                        "reference_char_count": max(sc["n_ref_chars"], 1),
                        "exact_match": rc == hc,
                    }
                    wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote predictions to {out_path}")


if __name__ == "__main__":
    main()
