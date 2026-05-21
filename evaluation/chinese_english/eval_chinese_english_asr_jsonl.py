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

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import parse_asr_output

_ASR_TEXT_TAG = "<asr_text>"
_ZW_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
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
_CN_NUMERAL_RE = re.compile(r"[零〇一二两兩三四五六七八九十百千万萬亿億壹贰貳叁參肆伍陆陸柒捌玖拾佰仟廿卅]+")
_CN_DIGIT_MAP = {
    "零": "0",
    "〇": "0",
    "一": "1",
    "壹": "1",
    "二": "2",
    "两": "2",
    "兩": "2",
    "贰": "2",
    "貳": "2",
    "三": "3",
    "叁": "3",
    "參": "3",
    "四": "4",
    "肆": "4",
    "五": "5",
    "伍": "5",
    "六": "6",
    "陆": "6",
    "陸": "6",
    "七": "7",
    "柒": "7",
    "八": "8",
    "捌": "8",
    "九": "9",
    "玖": "9",
}
_CN_NUMBER_UNITS = frozenset("十拾百佰千仟万萬亿億廿卅")


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


def _build_hanzi_converter(mode: str):
    if mode == "off":
        return None
    try:
        from opencc import OpenCC  # type: ignore
    except ImportError as exc:
        raise SystemExit("Install opencc-python-reimplemented for Han script normalization.") from exc
    return OpenCC("t2s" if mode == "to_simplified" else "s2t").convert


def _is_han(ch: str) -> bool:
    o = ord(ch)
    return (
        0x3400 <= o <= 0x4DBF
        or 0x4E00 <= o <= 0x9FFF
        or 0xF900 <= o <= 0xFAFF
        or 0x20000 <= o <= 0x2A6DF
        or 0x2A700 <= o <= 0x2B73F
        or 0x2B740 <= o <= 0x2B81F
        or 0x2B820 <= o <= 0x2CEAF
    )


def _is_ascii_letter(ch: str) -> bool:
    return ("a" <= ch <= "z") or ("A" <= ch <= "Z")


def _is_separator(ch: str) -> bool:
    cat = unicodedata.category(ch)
    return ch.isspace() or cat.startswith("P") or cat.startswith("S")


def _clean_global(text: str, hanzi_converter) -> str:
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


def _convert_cn_number_token(token: str) -> str:
    if not token:
        return token
    if all(ch in _CN_DIGIT_MAP for ch in token):
        return "".join(_CN_DIGIT_MAP[ch] for ch in token)
    try:
        import cn2an  # type: ignore
    except ImportError:
        return token
    try:
        value = cn2an.cn2an(token, "smart")
    except Exception:
        return token
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def normalize_and_tokenize(text: str, hanzi_converter=None) -> Tuple[str, List[str], str]:
    s = _clean_global(text, hanzi_converter)
    s = _CN_NUMERAL_RE.sub(lambda m: _convert_cn_number_token(m.group(0)), s)

    tokens: List[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if _is_ascii_letter(ch):
            j = i + 1
            while j < len(s):
                cj = s[j]
                if _is_ascii_letter(cj):
                    j += 1
                    continue
                if cj in ("'", "’", "`") and j + 1 < len(s) and _is_ascii_letter(s[j + 1]):
                    j += 1
                    continue
                break
            tokens.append(re.sub(r"['’`]", "", s[i:j]).lower())
            i = j
            continue

        if ch.isdigit():
            j = i + 1
            while j < len(s):
                cj = s[j]
                if cj.isdigit():
                    j += 1
                    continue
                if cj == "." and j + 1 < len(s) and s[j + 1].isdigit():
                    j += 2
                    continue
                break
            tokens.append(s[i:j])
            i = j
            continue

        if _is_han(ch):
            tokens.append(ch)
            i += 1
            continue

        if _is_separator(ch):
            i += 1
            continue

        i += 1

    tokens = [tok for tok in tokens if tok]
    norm_text = " ".join(tokens)
    norm_chars = "".join(tokens)
    return norm_text, tokens, norm_chars


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


def corpus_error_rates(ref_tokens: List[List[str]], hyp_tokens: List[List[str]], ref_chars: List[str], hyp_chars: List[str]):
    token_err = 0
    token_den = 0
    char_err = 0
    char_den = 0
    exact = 0

    for rt, ht, rc, hc in zip(ref_tokens, hyp_tokens, ref_chars, hyp_chars):
        token_err += _levenshtein_1d(rt, ht)
        token_den += max(len(rt), 1)
        char_err += _levenshtein_1d(list(rc), list(hc))
        char_den += max(len(rc), 1)
        exact += int(rt == ht)

    n = max(len(ref_tokens), 1)
    return {
        "mer": token_err / max(token_den, 1),
        "cer": char_err / max(char_den, 1),
        "sentence_accuracy": exact / n,
        "token_errors": token_err,
        "reference_tokens": token_den,
        "char_errors": char_err,
        "reference_chars": char_den,
    }


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

    hanzi_converter = _build_hanzi_converter(args.hanzi_script_norm)
    ref_norm_texts: List[str] = []
    hyp_norm_texts: List[str] = []
    ref_tokens: List[List[str]] = []
    hyp_tokens: List[List[str]] = []
    ref_chars: List[str] = []
    hyp_chars: List[str] = []

    for ref, hyp in zip(refs_raw, predictions):
        rt, r_tok, rc = normalize_and_tokenize(ref, hanzi_converter)
        ht, h_tok, hc = normalize_and_tokenize(hyp, hanzi_converter)
        ref_norm_texts.append(rt)
        hyp_norm_texts.append(ht)
        ref_tokens.append(r_tok)
        hyp_tokens.append(h_tok)
        ref_chars.append(rc)
        hyp_chars.append(hc)

    metrics = corpus_error_rates(ref_tokens, hyp_tokens, ref_chars, hyp_chars)

    print("")
    print("=== Chinese-English ASR metrics ===")
    print(f"Samples:             {len(rows)}")
    print(
        "Scoring:             NFKC, invisible/control chars removed, non-speech annotations removed, "
        "punctuation/symbols as token boundaries, Latin lowercased, Chinese numerals ITN, "
        "tokens = English words + numbers + individual Han chars"
    )
    print(f"Hanzi script norm:   {args.hanzi_script_norm}")
    print(f"MER:                 {metrics['mer'] * 100:.2f}%")
    print(f"CER:                 {metrics['cer'] * 100:.2f}%")
    print(f"Sentence Accuracy:   {metrics['sentence_accuracy'] * 100:.2f}%")
    print(f"Token errors:        {metrics['token_errors']} / {metrics['reference_tokens']}")
    print(f"Char errors:         {metrics['char_errors']} / {metrics['reference_chars']}")

    if args.output_predictions:
        parent = os.path.dirname(args.output_predictions)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.output_predictions, "w", encoding="utf-8") as wf:
            for ex, rr, hh, rnt, hnt, rt, ht, rc, hc in zip(
                rows,
                refs_raw,
                predictions,
                ref_norm_texts,
                hyp_norm_texts,
                ref_tokens,
                hyp_tokens,
                ref_chars,
                hyp_chars,
            ):
                token_dist = _levenshtein_1d(rt, ht)
                char_dist = _levenshtein_1d(list(rc), list(hc))
                rec = {
                    "audio": ex["audio"],
                    "reference_raw": rr,
                    "hypothesis_raw": hh,
                    "reference_norm": rnt,
                    "hypothesis_norm": hnt,
                    "reference_tokens": rt,
                    "hypothesis_tokens": ht,
                    "utterance_token_errors": token_dist,
                    "reference_token_count": max(len(rt), 1),
                    "utterance_char_errors": char_dist,
                    "reference_char_count": max(len(rc), 1),
                    "exact_match": rt == ht,
                }
                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote predictions to {args.output_predictions}")


if __name__ == "__main__":
    main()
