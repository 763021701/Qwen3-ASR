"""English TN for woitn CTC targets: digits -> words with num2words (DESIGN.md §7.2 rule 5).

Handles currency ($ € £, with cents or a scale word), percentages, ordinals (21st), decades (90s), decimals (digits
after the point read one by one), comma-grouped numbers, years (bare 4-digit numbers in 1100-2099), leading-zero
digit strings (read digit by digit) and digits inside words (mp3 -> mp three). Slashes between digits (fractions,
dates) are not verbalized reliably: english_tn returns None and the sample is dropped. num2words writes the British
"and" ("one hundred and five"). Not handled: readings that depend on context ("911" becomes "nine hundred and
eleven") and unit abbreviations ("kg" stays).
"""
from __future__ import annotations

import re

from num2words import num2words

_DIGITS = "zero one two three four five six seven eight nine".split()
_NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_CURRENCY = {"$": ("dollar", "dollars", "cent", "cents"), "€": ("euro", "euros", "cent", "cents"),
             "£": ("pound", "pounds", "penny", "pence")}
_CURRENCY_RE = re.compile(rf"([$€£])\s?({_NUM})(\s+(?:thousand|million|billion|trillion)\b)?", re.I)
_PERCENT_RE = re.compile(rf"({_NUM})\s?%")
_ORDINAL_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.I)
_DECADE_RE = re.compile(r"\b(\d{2,4})'?s\b")
_NUMBER_RE = re.compile(_NUM)


def _cardinal(digits: str, year: bool = False) -> str:
    digits = digits.replace(",", "")
    if (len(digits) > 1 and digits[0] == "0") or len(digits) > 15:  # codes, phone numbers
        return " ".join(_DIGITS[int(d)] for d in digits)
    n = int(digits)
    return num2words(n, to="year") if year and len(digits) == 4 and 1100 <= n <= 2099 else num2words(n)


def _number(s: str, year: bool = False) -> str:
    whole, _, frac = s.replace(",", "").partition(".")
    words = _cardinal(whole, year and not frac and "," not in s)  # "1,234" is a count, not a year
    return words + (" point " + " ".join(_DIGITS[int(d)] for d in frac) if frac else "")


def _currency(m) -> str:
    one, many, cent, cents = _CURRENCY[m.group(1)]
    amount, scale = m.group(2), m.group(3)
    if scale:
        return f" {_number(amount)} {scale.strip().lower()} {many} "
    whole, _, frac = amount.replace(",", "").partition(".")
    words = f"{_cardinal(whole)} {one if whole == '1' else many}"
    c = int(frac.ljust(2, "0")[:2]) if frac else 0
    return f" {words} and {num2words(c)} {cent if c == 1 else cents} " if c else f" {words} "


def _decade(m) -> str:
    words = _number(m.group(1), year=True)
    return f" {words[:-1]}ies " if words.endswith("y") else f" {words}s "


def english_tn(text: str):
    """Verbalize the digits of an English transcript; None if some cannot be verbalized reliably."""
    if re.search(r"\d\s*/\s*\d", text):
        return None
    text = _CURRENCY_RE.sub(_currency, text)
    text = _PERCENT_RE.sub(lambda m: f" {_number(m.group(1))} percent ", text)
    text = _ORDINAL_RE.sub(lambda m: f" {num2words(int(m.group(1)), to='ordinal')} ", text)
    text = _DECADE_RE.sub(_decade, text)
    text = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", text)
    return _NUMBER_RE.sub(lambda m: f" {_number(m.group(0), year=True)} ", text)
