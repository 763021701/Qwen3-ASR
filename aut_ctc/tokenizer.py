"""SenseVoiceSmall SentencePiece vocab -> compact zh/en/yue/dialect CTC subset, text normalization, MER tokens.

New id convention: 0 = CTC blank, 1 = <unk>, 2.. = kept original pieces in ascending original id, then the extra
characters (one id each, in file order).
- Source: chn_jpn_yue_eng_ko_spectok.bpe.model from FunAudioLLM/SenseVoiceSmall (named bpe, the English
  part is actually unigram). FunASR Model License v1.1: attribution required, keep the model name.
- Original id 0 is <unk>, which SenseVoice also uses as CTC blank; here the two are split and any
  out-of-vocabulary piece maps to new id 1.
- Extra characters (extra_chars.txt): Han characters of Wu / Cantonese / Min / Hakka writing that the SenseVoice vocab
  lacks. Fun-ASR-Nano's CTC vocab has the same 10,124 Han characters and falls back to UTF-8 bytes for them.
- Isolated "▁": text is encoded with a leading ▁ so the first word gets the same ▁-initial pieces as any later word
  (the model has add_dummy_prefix=False and strips leading spaces). An isolated ▁ is kept only between two
  Latin/digit pieces: words such as "qwen" or "russian" have no ▁-initial piece, and dropping it would glue them to
  the previous word. Next to Han text it carries no information and is dropped.
- ``sp`` can be a sentencepiece.SentencePieceProcessor or anything implementing
  get_piece_size / id_to_piece / piece_to_id / encode / decode.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

LANG_TAGS = ("<|zh|>", "<|en|>", "<|zh/en|>", "<|en/zh|>", "<|nospeech|>")
DIALECT_TAGS = ("<|yue|>", "<|wuyu|>", "<|minnan|>", "<|dialect|>")
STYLE_TAGS = ("<|withitn|>", "<|woitn|>")
EMO_TAGS = ("<|HAPPY|>", "<|SAD|>", "<|ANGRY|>", "<|NEUTRAL|>", "<|FEARFUL|>", "<|DISGUSTED|>",
            "<|SURPRISED|>", "<|OTHER|>", "<|EMO_UNKNOWN|>")
EVENT_TAGS = ("<|Speech|>", "<|/Speech|>", "<|BGM|>", "<|/BGM|>", "<|Laughter|>", "<|/Laughter|>",
              "<|Applause|>", "<|/Applause|>", "<|Cry|>", "<|Sneeze|>", "<|Breath|>", "<|Cough|>",
              "<|Sing|>", "<|Speech_Noise|>", "<|GBG|>", "<|Event_UNK|>")
# Phase 1 only uses language/style tags; the others are kept now so the vocab (and the output layer shape) does
# not change in later phases.
DEFAULT_SPECIALS = LANG_TAGS + DIALECT_TAGS + STYLE_TAGS + EMO_TAGS + EVENT_TAGS
KEEP_CLASSES = ("han", "latin", "digit", "punct", "space")
DEFAULT_EXTRA_CHARS = Path(__file__).with_name("extra_chars.txt")

_TAG_RE = re.compile(r"<\|[^|]+\|>")
CJK = "㐀-䶿一-鿿豈-﫿\U00020000-\U0003134F"  # Ext-A, URO, compatibility, Ext-B..H


def char_class(c: str) -> str:
    o = ord(c)
    if 0x3400 <= o <= 0x4DBF or 0x4E00 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF or o >= 0x20000:
        return "han"
    if 0xAC00 <= o <= 0xD7AF or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F:
        return "hangul"
    if 0x3040 <= o <= 0x30FF or 0x31F0 <= o <= 0x31FF:
        return "kana"
    if c.isdigit():
        return "digit"
    if c.isalpha() or c == "'":
        return "latin"
    return "other"


def piece_class(piece: str) -> str:
    if piece in ("<unk>", "<s>", "</s>"):
        return "control"
    if _TAG_RE.fullmatch(piece):
        return "special"
    if re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", piece):
        return "byte"
    body = piece[1:] if piece.startswith("▁") else piece
    if not body:
        return "space"
    kinds = {char_class(c) for c in body}
    if kinds == {"han"}:
        return "han"
    if "hangul" in kinds:
        return "hangul"
    if "kana" in kinds:
        return "kana"
    if kinds <= {"latin"} and any(c.isalpha() for c in body):
        return "latin"
    if kinds == {"digit"}:
        return "digit"
    if kinds <= {"other", "latin"}:  # pure punctuation / symbols (incl. a lone apostrophe)
        return "punct"
    return "mixed"


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_extra_chars(path) -> list[str]:
    """One character per line; text after # is a comment."""
    with open(path, encoding="utf-8") as f:
        return [c for c in (line.split("#", 1)[0].strip() for line in f) if c]


class SenseVoiceCTCTokenizer:
    BLANK, UNK = 0, 1

    def __init__(self, sp, keep_specials=DEFAULT_SPECIALS, keep_classes=KEEP_CLASSES, extra_chars=()):
        self.sp = sp
        n = sp.get_piece_size()
        self.pieces = [sp.id_to_piece(i) for i in range(n)]
        classes = [piece_class(p) for p in self.pieces]
        specials = set(keep_specials)
        keep = [i for i, (p, c) in enumerate(zip(self.pieces, classes))
                if c in keep_classes or (c == "special" and p in specials)]
        missing = specials - {self.pieces[i] for i in keep}
        if missing:
            raise ValueError(f"source vocab lacks these special tags: {sorted(missing)}")
        self.new2old = [-1, -1] + keep
        self.old2new = {o: n_ + 2 for n_, o in enumerate(keep)}
        self.space_id = sp.piece_to_id("▁")
        if self.pieces[self.space_id] != "▁":  # no isolated ▁ piece: piece_to_id fell back to <unk>
            self.space_id = -1
        self._wordish = [c in ("latin", "digit") for c in classes]
        self.extra_chars = list(extra_chars)
        bad = [c for c in self.extra_chars if len(c) != 1 or self.pieces[sp.piece_to_id(c)] == c]
        if bad or len(set(self.extra_chars)) != len(self.extra_chars):
            raise ValueError(f"extra chars must be distinct single characters missing from the vocab: {bad}")
        self.extra = {c: len(self.new2old) + k for k, c in enumerate(self.extra_chars)}
        self._extra_re = re.compile("([" + "".join(map(re.escape, self.extra_chars)) + "])") if self.extra_chars else None
        self.vocab_size = len(self.new2old) + len(self.extra_chars)

    # ------------------------------------------------------------------ encode / decode
    def encode(self, text: str) -> list[int]:
        """``text`` should already be normalized. Returns new ids: never 0 (blank), OOV -> 1."""
        seq = []  # SentencePiece ids, extra characters as negated new ids
        for k, part in enumerate(self._extra_re.split(text) if self._extra_re else [text]):
            if k % 2:
                seq.append(-self.extra[part])
            elif part.strip():
                seq += self.sp.encode("▁" + part.strip())
        ids = []
        for k, o in enumerate(seq):
            if o == self.space_id and not (0 < k < len(seq) - 1 and seq[k - 1] >= 0 and seq[k + 1] >= 0
                                          and self._wordish[seq[k - 1]] and self._wordish[seq[k + 1]]):
                continue
            ids.append(-o if o < 0 else self.old2new.get(o, self.UNK))
        return ids

    def tag_id(self, tag: str) -> int:
        return self.old2new[self.sp.piece_to_id(tag)]

    def decode(self, ids, keep_tags: bool = False) -> str:
        parts, run, base = [], [], len(self.new2old)
        for i in list(ids) + [None]:
            if i is not None and 2 <= i < base:
                run.append(self.new2old[i])
                continue
            if run:
                parts.append(self.sp.decode(run))
                run = []
            if i is not None and i >= base:
                parts.append(self.extra_chars[i - base])
        text = " ".join(parts)
        if not keep_tags:
            text = _TAG_RE.sub("", text)
        return fix_spacing(text)

    def id_to_piece(self, i: int) -> str:
        if i < 2:
            return ("<blank>", "<unk>")[i]
        return self.pieces[self.new2old[i]] if i < len(self.new2old) else self.extra_chars[i - len(self.new2old)]

    # ------------------------------------------------------------------ persistence
    def save_vocab(self, path, source_model: str = "") -> None:
        """Save the mapping; training, inference and evaluation must share the same file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"source_model": str(source_model),
                       "source_sha256": file_sha256(source_model) if source_model else "",
                       "vocab_size": self.vocab_size,
                       "new2old": self.new2old,
                       "extra_chars": self.extra_chars,
                       "pieces": [self.id_to_piece(i) for i in range(self.vocab_size)]},
                      f, ensure_ascii=False, indent=0)


def load_tokenizer(sp_model, vocab_json=None, extra_chars=None) -> SenseVoiceCTCTokenizer:
    """Build from the SentencePiece model. ``extra_chars``: path of an extra-character list. With ``vocab_json`` the
    saved mapping (extra characters included) is used and must match the model and, if given, ``extra_chars``."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(sp_model))
    chars = read_extra_chars(extra_chars) if extra_chars else []
    if vocab_json is None:
        return SenseVoiceCTCTokenizer(sp, extra_chars=chars)
    saved = json.loads(Path(vocab_json).read_text(encoding="utf-8"))
    if extra_chars and chars != saved.get("extra_chars", []):
        raise ValueError(f"{extra_chars} differs from the extra characters recorded in {vocab_json}")
    tok = SenseVoiceCTCTokenizer(sp, extra_chars=saved.get("extra_chars", []))
    if saved["new2old"] != tok.new2old or saved["source_sha256"] != file_sha256(sp_model):
        raise ValueError(f"{vocab_json} was built from a different SentencePiece model than {sp_model}")
    return tok


# ---------------------------------------------------------------------- text normalization
_PUNCT_RE = re.compile(r"[^\w\s'" + CJK + r"]|_")
_LATIN_FOLD = {"ß": "ss", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ø": "o", "Ø": "O"}


def to_halfwidth_alnum(text: str) -> str:
    """Full-width ASCII letters/digits/space -> half-width; Chinese punctuation untouched."""
    out = []
    for c in text:
        o = ord(c)
        if c == "　":
            out.append(" ")
        elif 0xFF10 <= o <= 0xFF19 or 0xFF21 <= o <= 0xFF3A or 0xFF41 <= o <= 0xFF5A:
            out.append(chr(o - 0xFEE0))
        else:
            out.append(c)
    return "".join(out)


def fold_latin(text: str) -> str:
    """é -> e, ß -> ss, ...: the vocab only has unaccented Latin letters (accented ones would be <unk>)."""
    out = []
    for c in text:
        if ord(c) < 0x80 or char_class(c) != "latin":
            out.append(c)
        else:
            out.append(_LATIN_FOLD.get(c) or unicodedata.normalize("NFKD", c).encode("ascii", "ignore").decode() or c)
    return "".join(out)


def fix_spacing(text: str) -> str:
    """No space between Han chars; one space between Han and Latin (so English words get ▁ pieces)."""
    text = re.sub(rf"(?<=[{CJK}])\s+(?=[{CJK}])", "", text)
    text = re.sub(rf"(?<=[{CJK}])(?=[A-Za-z0-9])", " ", text)
    text = re.sub(rf"(?<=[A-Za-z0-9'])(?=[{CJK}])", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str, style: str = "woitn") -> tuple[str, bool]:
    """Return (normalized text, contains Arabic digits).

    woitn (phase 1): drop punctuation (keep word-internal apostrophes), lowercase English; digits should be
        spoken form -- when some remain, the caller decides between TN and dropping the sample.
    withitn (phase 2): keep punctuation, case and digits.
    Accented Latin letters are folded for both styles. Traditional->simplified (OpenCC t2s) and TN are done by the
    data pipeline, not here.
    """
    text = fold_latin(unicodedata.normalize("NFC", to_halfwidth_alnum(text)))
    text = "".join(c for c in text if unicodedata.category(c)[0] != "C" or c in "\t\n ")
    if style == "woitn":
        text = _PUNCT_RE.sub(" ", text)
        text = re.sub(r"(?<![A-Za-z])'|'(?![A-Za-z])", " ", text)  # keep only word-internal apostrophes
        text = text.lower()
    elif style != "withitn":
        raise ValueError(style)
    text = fix_spacing(text)
    return text, bool(re.search(r"\d", text))


def mer_tokens(text: str) -> list[str]:
    """Shared CER/WER/MER tokens: one per Han char, one per (lowercase) English word or digit run."""
    text = _TAG_RE.sub(" ", to_halfwidth_alnum(text)).lower()
    return re.findall(rf"[{CJK}]|[a-z]+(?:'[a-z]+)*|\d+", text)
