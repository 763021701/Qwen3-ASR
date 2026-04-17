#!/usr/bin/env python3
# coding=utf-8
"""
Evaluate a Qwen3-ASR checkpoint on a Qwen3-style jsonl manifest (e.g. ug_dev_qwen3.jsonl).

For Uyghur (Arabic script, space-separated orthography in Common Voice):
  - WER (word error rate) is the usual headline metric and matches most ASR papers.
  - CER (character / Unicode-scalar error rate) is a useful secondary metric for script detail.

Install (recommended): pip install jiwer
Without jiwer, a small Levenshtein fallback is used (same corpus-level definition).

Example:
  python finetuning/eval_uyghur_asr_jsonl.py \
    --jsonl data/ug_test_qwen3.jsonl \
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

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

    refs = [normalize_for_scoring(r) for r in refs_raw]
    hyps = [normalize_for_scoring(h) for h in predictions]

    try:
        wer, cer = corpus_wer_cer_jiwer(refs, hyps)
        backend = "jiwer"
    except ImportError:
        wer, cer = corpus_wer_cer_fallback(refs, hyps)
        backend = "builtin_levenshtein"

    print("")
    print("=== Uyghur ASR metrics ===")
    print(f"Samples:     {len(rows)}")
    print(f"Scoring:     NFC, ZWSP removed, whitespace collapsed")
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
