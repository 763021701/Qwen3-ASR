# coding=utf-8
"""Text normalization (TN) for the Qwen3-ASR CTC pipeline.

CTC targets must be the spoken form. Route by language:
  - Chinese -> cn_tn.TextNorm (verbalize digits to anyi, e.g. ``1.5`` -> ``一点五``;
    fullwidth->halfwidth, lowercase, remove punctuation/space).
  - English -> a digit-preserving normalizer (lowercase, drop punctuation/symbols but
    keep ``.``/``,`` inside numbers, collapse whitespace). e.g. ``1.5`` -> ``1.5``,
    ``1,234`` -> ``1,234``, ``Hello, World!`` -> ``hello world``.

Zero external dependencies: ``cn_tn`` is stdlib-only and the English path is a small
custom normalizer (whisper's EnglishTextNormalizer was rejected because it verbalizes
``1.5`` -> ``one.5``, and its scoreformat step splits ``15`` -> ``1 5``).

The LLM branch keeps the original written ``target`` untouched; only CTC labels and
eval reference/hypothesis go through ``normalize_ctc_text``.
"""
import re

from .cn_tn import TextNorm
from qwen_asr.inference.utils import _ASR_TEXT_TAG

_LANG_RE = re.compile(r"language\s+(\w+)", re.IGNORECASE)
_CJK_RE = re.compile(r"[一-鿿]")  # CJK Unified Ideographs (U+4E00 - U+9FFF)
_INTRA_NUM_PUNCT = {".", ","}  # keep these only when flanked by digits
_SPACE_RE = re.compile(r"\s+")

_cn_norm = None


def _get_cn() -> TextNorm:
    global _cn_norm
    if _cn_norm is None:
        _cn_norm = TextNorm(to_banjiao=True, to_lower=True, remove_space=True)
    return _cn_norm


def _normalize_en(text: str) -> str:
    """Lowercase, drop punctuation/symbols, keep ``.``/``,`` only inside numbers.

    ``1.5`` -> ``1.5``, ``1,234`` -> ``1,234``, ``Hello, World!`` -> ``hello world``,
    ``iPhone 15`` -> ``iphone 15``.
    """
    s = text.lower()
    out = []
    n = len(s)
    for i, ch in enumerate(s):
        if ch.isalnum() or ch.isspace():
            out.append(ch)
        elif ch in _INTRA_NUM_PUNCT:
            prev = s[i - 1] if i > 0 else ""
            nxt = s[i + 1] if i + 1 < n else ""
            if prev.isdigit() and nxt.isdigit():
                out.append(ch)
    return _SPACE_RE.sub(" ", "".join(out)).strip()


def normalize_ctc_text(text: str) -> str:
    """Normalize a transcript to the spoken form used by CTC labels.

    Handles the ``language X<asr_text>Y`` prefix convention (see
    ``finetuning/README.md``) by stripping it and routing ``Y`` by language.
    For raw text without the prefix, routes by a CJK heuristic.

    Returns the normalized transcript only (no language/<asr_text> markers).
    """
    if not text:
        return ""
    s = str(text)
    if _ASR_TEXT_TAG in s:
        meta, transcript = s.split(_ASR_TEXT_TAG, 1)
        m = _LANG_RE.search(meta)
        lang = m.group(1) if m else ""
    else:
        lang, transcript = "", s
    transcript = transcript.strip()
    if not transcript:
        return ""

    lang_l = lang.lower()
    if lang_l == "chinese" or (lang_l != "english" and _CJK_RE.search(transcript)):
        return _get_cn()(transcript)
    return _normalize_en(transcript)
