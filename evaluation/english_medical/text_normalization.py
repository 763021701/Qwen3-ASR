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
_WHISPER_ONES_RE = re.compile(r"\bones\b")
_WHISPER_ONE_RE = re.compile(r"\bone\b")
_OCLOCK_RE = re.compile(r"\bo\s*['']?\s*clock\b", re.IGNORECASE)
_REPEATED_TWO_RE = re.compile(r"\btwo\s*,\s*two\b", re.IGNORECASE)
_LEVEL_ROMAN_RE = re.compile(r"\blevel\s+(i{1,3})\b")
_WS_RE = re.compile(r"\s+")
_MULTIPLICATION_SENTINEL = "qwenmultoken"
_OCLOCK_SENTINEL = "qwenoclock"
_REPEATED_TWO_SENTINEL = "qwentwotwo"
_MAX_SPELLING_LETTERS = 8
_PUNCTUATION_WITHOUT_PERIOD = str.maketrans("", "", string.punctuation.replace(".", ""))
_CLOSED_HYPHENATED_COMPOUNDS = {
    "antero-inferior": "anteroinferior",
    "antero-posteriorly": "anteroposteriorly",
    "antero-superior": "anterosuperior",
    "medio-laterally": "mediolaterally",
    "micro-nodules": "micronodules",
    "supero-inferiorly": "superoinferiorly",
}
_ROMAN_LEVEL_MAP = {"i": "1", "ii": "2", "iii": "3"}
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


def _merge_spaced_letters(text: str) -> str:
    def _should_skip_merge(match: re.Match[str]) -> bool:
        start = match.start()
        prefix = text[:start].lower()
        if prefix.endswith(" to "):
            return True
        if prefix.endswith("block ") or prefix.endswith("blocks "):
            return True
        return False

    def _merge_match(match: re.Match[str]) -> str:
        if _should_skip_merge(match):
            return match.group(0)
        letters = match.group(0).split()
        if len(letters) <= _MAX_SPELLING_LETTERS:
            return "".join(letters)
        chunks: list[str] = []
        remaining = letters
        while len(remaining) > _MAX_SPELLING_LETTERS:
            chunks.insert(0, "".join(remaining[-_MAX_SPELLING_LETTERS:]))
            remaining = remaining[:-_MAX_SPELLING_LETTERS]
        if remaining:
            chunks.insert(0, "".join(remaining))
        return " ".join(chunks)

    return _SINGLE_LETTER_SEQUENCE_RE.sub(_merge_match, text)


def _restore_arabic_one(text: str) -> str:
    """Undo Whisper's isolated 1 -> one / 1s -> ones rewrite."""
    text = _WHISPER_ONES_RE.sub("1s", text)
    return _WHISPER_ONE_RE.sub("1", text)


def _close_spaced_compounds(text: str) -> str:
    for hyphenated, closed in _CLOSED_HYPHENATED_COMPOUNDS.items():
        spaced = hyphenated.replace("-", " ")
        text = re.sub(
            rf"(?<![A-Za-z]){re.escape(spaced)}(?![A-Za-z])",
            closed,
            text,
            flags=re.IGNORECASE,
        )
    return text


def _normalize_level_roman(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        roman = match.group(1).lower()
        return f"level {_ROMAN_LEVEL_MAP.get(roman, roman)}"

    return _LEVEL_ROMAN_RE.sub(_replace, text)


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
    s = s.replace("兩", "二").replace("两", "二").replace("倆", "二")
    s = s.replace("廿", "二十")
    s = _OCLOCK_RE.sub(_OCLOCK_SENTINEL, s)
    s = _REPEATED_TWO_RE.sub(_REPEATED_TWO_SENTINEL, s)
    s = _ZH_NUMBER_NORMALIZER.normalize(s)
    s = s.replace(_OCLOCK_SENTINEL, " oclock ")
    s = s.replace(_REPEATED_TWO_SENTINEL, " 2 2 ")
    s = s.replace(chr(40), " ").replace(chr(41), " ")
    # Merge "S N" before the English number normalizer so "SN 1, SN 2" is not
    # read as "1 s" (plural ones) and "S N. S N" is not later glued into "snsn".
    s = _merge_spaced_letters(s)
    s = _EN_NUMBER_NORMALIZER.normalize(s)
    s = _restore_arabic_one(s)
    s = _merge_spaced_letters(s)
    s = _close_spaced_compounds(s)
    s = _normalize_level_roman(s)
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
