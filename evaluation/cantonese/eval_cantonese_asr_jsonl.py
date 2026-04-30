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

Install (recommended): pip install jiwer
Without jiwer, a small Levenshtein fallback is used for CER.

Recommended for Cantonese benchmark scoring:
  pip install jiwer opencc-python-reimplemented cn2an
This script defaults to converting both reference and hypothesis to Traditional Chinese
before CER scoring, so Simplified/Traditional spelling variants do not inflate CER.
It also removes Unicode punctuation by default, so punctuation differences do not inflate CER.
English is lowercased and Chinese numerals are inverse-text-normalized to Arabic digits.

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
import unicodedata
from typing import Any, Dict, List, Sequence, Tuple

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import parse_asr_output

_ASR_TEXT_TAG = "<asr_text>"
_ZW_RE = re.compile(r"[\u200c\u200d\ufeff]")
_CN_NUMERAL_RE = re.compile(r"[零〇一二两三四五六七八九十百千万億亿壹贰叁肆伍陆柒捌玖拾佰仟萬廿卅]+")


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


def build_hanzi_script_converter(mode: str):
    key = (mode or "off").strip().lower()
    if key == "off":
        return None

    config_map = {
        "to_traditional": "s2t",
        "to_simplified": "t2s",
    }
    config = config_map[key]
    try:
        from opencc import OpenCC  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise SystemExit(
            "Install OpenCC first for Hanzi script normalization: pip install opencc-python-reimplemented"
        ) from exc
    cc = OpenCC(config)
    return cc.convert


def remove_unicode_punctuation(s: str) -> str:
    return "".join(ch for ch in s if not unicodedata.category(ch).startswith("P"))


def apply_number_itn(s: str) -> str:
    try:
        import cn2an  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise SystemExit("Install cn2an first for number ITN: pip install cn2an") from exc

    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        try:
            return str(cn2an.cn2an(token, "smart"))
        except Exception:
            return token

    return _CN_NUMERAL_RE.sub(repl, s)


def normalize_for_scoring(s: str, *, keep_whitespace: bool, hanzi_script_converter) -> str:
    """
    Normalize text for Cantonese CER.

    Default behavior removes all whitespace because Han-character ASR references usually
    do not use reliable word boundaries and model outputs may insert spaces inconsistently.
    """
    s = unicodedata.normalize("NFC", (s or "").strip())
    s = _ZW_RE.sub("", s)
    if hanzi_script_converter is not None:
        s = hanzi_script_converter(s)
    s = s.lower()
    s = apply_number_itn(s)
    s = remove_unicode_punctuation(s)
    if keep_whitespace:
        s = " ".join(s.split())
    else:
        s = "".join(s.split())
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


def corpus_cer_fallback(refs: List[str], hyps: List[str]) -> float:
    c_err, c_den = 0, 0
    for r, h in zip(refs, hyps):
        rc = list(r)
        hc = list(h)
        if len(rc) == 0 and len(hc) == 0:
            continue
        if len(rc) == 0:
            c_err += len(hc)
            c_den += max(len(hc), 1)
        else:
            c_err += _levenshtein_1d(rc, hc)
            c_den += len(rc)
    return c_err / max(c_den, 1)


def corpus_cer_jiwer(refs: List[str], hyps: List[str]) -> float:
    import jiwer

    return float(jiwer.cer(refs, hyps))


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

    hanzi_script_converter = build_hanzi_script_converter(args.hanzi_script_norm)
    refs = [
        normalize_for_scoring(r, keep_whitespace=args.keep_whitespace, hanzi_script_converter=hanzi_script_converter)
        for r in refs_raw
    ]
    hyps = [
        normalize_for_scoring(h, keep_whitespace=args.keep_whitespace, hanzi_script_converter=hanzi_script_converter)
        for h in predictions
    ]

    try:
        cer = corpus_cer_jiwer(refs, hyps)
        backend = "jiwer"
    except ImportError:
        cer = corpus_cer_fallback(refs, hyps)
        backend = "builtin_levenshtein"

    exact_matches = sum(1 for r, h in zip(refs, hyps) if r == h)
    sentence_accuracy = exact_matches / max(len(refs), 1)

    print("")
    print("=== Cantonese ASR metrics ===")
    print(f"Samples:            {len(rows)}")
    print(
        "Scoring:            "
        f"NFC, ZWSP removed, English lowercased, number ITN, Unicode punctuation removed, "
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
            for ex, rr, hh, pr in zip(rows, refs, hyps, predictions):
                dist = _levenshtein_1d(list(rr), list(hh))
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
        print(f"Wrote predictions to {out_path}")


if __name__ == "__main__":
    main()
