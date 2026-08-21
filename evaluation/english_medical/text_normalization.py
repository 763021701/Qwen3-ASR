# coding=utf-8
"""Shared medical text normalization for Qwen3-ASR English medical experiments.

Lightweight module (no torch) used by:
- data preparation / sampling (tools/prepare_real_raw_denoised_sft.py)
- in-training CER/WER checkpoint selection (finetuning/qwen3_asr_sft.py)
- final evaluation (evaluation/english_medical/eval_english_medical_asr_jsonl.py)
"""

from __future__ import annotations

import re
import string
import unicodedata

from masr_eval_pkg.normalizers import get_normalizer

_DASH_RE = re.compile(r"[-\u2010\u2011\u2012\u2013\u2014\u2212]")
_SINGLE_LETTER_SEQUENCE_RE = re.compile(r"(?<![A-Za-z])(?:[A-Za-z][ \t]+){1,}[A-Za-z](?![A-Za-z])")
_ALPHANUMERIC_BOUNDARY_RE = re.compile(r"(?<=\d)(?=[A-Za-z])|(?<=[A-Za-z])(?=\d)")
_CANTONESE_CLASSIFIER_RE = re.compile(r"(?<=\d)\s*個\s*(?=[A-Za-z])")
_GRAM_VARIANT_RE = re.compile(r"\b(?:gramme|grams|gm)\b", re.IGNORECASE)
_PARAFFIN_VARIANT_RE = re.compile(r"\bparaffine\b", re.IGNORECASE)
_NON_DECIMAL_PERIOD_RE = re.compile(r"(?<!\d)\.|\.(?!\d)")
_WS_RE = re.compile(r"\s+")
_MULTIPLICATION_SENTINEL = "qwenmultoken"
_PUNCTUATION_WITHOUT_PERIOD = str.maketrans("", "", string.punctuation.replace(".", ""))
_CLOSED_HYPHENATED_COMPOUNDS = {
    "antero-inferior": "anteroinferior",
    "antero-posteriorly": "anteroposteriorly",
    "antero-superior": "anterosuperior",
    "medio-laterally": "mediolaterally",
    "micro-nodules": "micronodules",
    "supero-inferiorly": "superoinferiorly",
}
_ZH_NUMBER_NORMALIZER = get_normalizer(
    "zh",
    zh_convert="none",
    number_normalize="to_arabic",
    remove_punctuation=False,
)
_EN_NUMBER_NORMALIZER = get_normalizer(
    "en",
    number_normalize="to_arabic",
    remove_punctuation=False,
)


def normalize_english(text: str) -> str:
    """Normalize mixed English/Chinese medical transcripts for WER."""
    s = unicodedata.normalize("NFKC", text or "")
    s = _DASH_RE.sub("-", s)
    for hyphenated, closed in _CLOSED_HYPHENATED_COMPOUNDS.items():
        s = re.sub(
            rf"(?<![A-Za-z]){re.escape(hyphenated)}(?![A-Za-z])",
            closed,
            s,
            flags=re.IGNORECASE,
        )
    s = s.replace("-", " ")
    s = re.sub(r"(?<![A-Za-z])x(?![A-Za-z])", _MULTIPLICATION_SENTINEL, s, flags=re.IGNORECASE)
    s = s.replace("×", f" {_MULTIPLICATION_SENTINEL} ").replace("乘", f" {_MULTIPLICATION_SENTINEL} ")
    s = s.replace("兩", "二").replace("两", "二")
    s = _ZH_NUMBER_NORMALIZER.normalize(s)
    s = s.replace(chr(40), " ").replace(chr(41), " ")
    s = _EN_NUMBER_NORMALIZER.normalize(s)
    s = _SINGLE_LETTER_SEQUENCE_RE.sub(lambda match: "".join(match.group(0).split()), s)
    s = _ALPHANUMERIC_BOUNDARY_RE.sub(" ", s)
    s = _CANTONESE_CLASSIFIER_RE.sub(" ", s)
    s = _GRAM_VARIANT_RE.sub("gram", s)
    s = _PARAFFIN_VARIANT_RE.sub("paraffin", s)
    s = s.replace(_MULTIPLICATION_SENTINEL, " x ")
    s = s.translate(_PUNCTUATION_WITHOUT_PERIOD)
    s = _NON_DECIMAL_PERIOD_RE.sub("", s)
    s = s.lower()
    s = _WS_RE.sub(" ", s).strip()
    return s
