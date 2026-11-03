"""SenseVoiceSmall SentencePiece vocab -> compact zh/en CTC subset, text normalization, MER tokens.

New id convention: 0 = CTC blank, 1 = <unk>, 2.. = kept original pieces in ascending original id.
- Source: chn_jpn_yue_eng_ko_spectok.bpe.model from FunAudioLLM/SenseVoiceSmall (named bpe, the English
  part is actually unigram). FunASR Model License v1.1: attribution required, keep the model name.
- Original id 0 is <unk>, which SenseVoice also uses as CTC blank; here the two are split and any
  out-of-vocabulary piece maps to new id 1.
- The isolated "▁" piece (dummy prefix, space before a Han char) has no acoustic counterpart: dropped.
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
STYLE_TAGS = ("<|withitn|>", "<|woitn|>")
EMO_TAGS = ("<|HAPPY|>", "<|SAD|>", "<|ANGRY|>", "<|NEUTRAL|>", "<|FEARFUL|>", "<|DISGUSTED|>",
            "<|SURPRISED|>", "<|OTHER|>", "<|EMO_UNKNOWN|>")
EVENT_TAGS = ("<|Speech|>", "<|/Speech|>", "<|BGM|>", "<|/BGM|>", "<|Laughter|>", "<|/Laughter|>",
              "<|Applause|>", "<|/Applause|>", "<|Cry|>", "<|Sneeze|>", "<|Breath|>", "<|Cough|>",
              "<|Sing|>", "<|Speech_Noise|>", "<|GBG|>", "<|Event_UNK|>")
# Phase 1 only uses language/style tags; emotion/event tags are kept now so the vocab (and the output
# layer shape) does not change in later phases.
DEFAULT_SPECIALS = LANG_TAGS + STYLE_TAGS + EMO_TAGS + EVENT_TAGS
KEEP_CLASSES = ("han", "latin", "digit", "punct")

_TAG_RE = re.compile(r"<\|[^|]+\|>")
_CJK = r"㐀-䶿一-鿿豈-﫿"


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


class SenseVoiceCTCTokenizer:
    BLANK, UNK = 0, 1

    def __init__(self, sp, keep_specials=DEFAULT_SPECIALS, keep_classes=KEEP_CLASSES):
        self.sp = sp
        n = sp.get_piece_size()
        self.pieces = [sp.id_to_piece(i) for i in range(n)]
        self.space_id = sp.piece_to_id("▁")
        if self.pieces[self.space_id] != "▁":  # no isolated ▁ piece: piece_to_id fell back to <unk>
            self.space_id = -1
        specials = set(keep_specials)
        keep = [i for i, p in enumerate(self.pieces)
                if piece_class(p) in keep_classes or (piece_class(p) == "special" and p in specials)]
        missing = specials - {self.pieces[i] for i in keep}
        if missing:
            raise ValueError(f"source vocab lacks these special tags: {sorted(missing)}")
        self.new2old = [-1, -1] + keep
        self.old2new = {o: n_ + 2 for n_, o in enumerate(keep)}
        self.vocab_size = len(self.new2old)

    # ------------------------------------------------------------------ encode / decode
    def encode(self, text: str) -> list[int]:
        """``text`` should already be normalized. Returns new ids: never 0 (blank), OOV -> 1."""
        return [self.old2new.get(i, self.UNK) for i in self.sp.encode(text) if i != self.space_id]

    def tag_id(self, tag: str) -> int:
        return self.old2new[self.sp.piece_to_id(tag)]

    def decode(self, ids, keep_tags: bool = False) -> str:
        old = [self.new2old[i] for i in ids if i >= 2]
        text = self.sp.decode(old)
        if not keep_tags:
            text = _TAG_RE.sub("", text)
        return re.sub(rf"(?<=[{_CJK}])\s+(?=[{_CJK}])", "", text).strip()

    def id_to_piece(self, i: int) -> str:
        return {0: "<blank>", 1: "<unk>"}.get(i) or self.pieces[self.new2old[i]]

    def dropped_boundary_count(self, text: str) -> int:
        """Non-initial isolated ▁ followed by a non-Han piece: dropping those merges two words on decode
        (DESIGN §6.6 check)."""
        old = self.sp.encode(text)
        return sum(old[k] == self.space_id and piece_class(self.pieces[old[k + 1]]) != "han"
                   for k in range(1, len(old) - 1))

    # ------------------------------------------------------------------ persistence
    def save_vocab(self, path, source_model: str = "") -> None:
        """Save the mapping; training, inference and evaluation must share the same file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"source_model": str(source_model),
                       "source_sha256": file_sha256(source_model) if source_model else "",
                       "vocab_size": self.vocab_size,
                       "new2old": self.new2old,
                       "pieces": [self.id_to_piece(i) for i in range(self.vocab_size)]},
                      f, ensure_ascii=False, indent=0)


def load_tokenizer(sp_model, vocab_json=None) -> SenseVoiceCTCTokenizer:
    """Build from the SentencePiece model; if ``vocab_json`` is given it must describe the same mapping."""
    import sentencepiece as spm

    tok = SenseVoiceCTCTokenizer(spm.SentencePieceProcessor(model_file=str(sp_model)))
    if vocab_json is not None:
        saved = json.loads(Path(vocab_json).read_text(encoding="utf-8"))
        if saved["new2old"] != tok.new2old or saved["source_sha256"] != file_sha256(sp_model):
            raise ValueError(f"{vocab_json} was built from a different SentencePiece model than {sp_model}")
    return tok


# ---------------------------------------------------------------------- text normalization
_PUNCT_RE = re.compile(r"[^\w\s'" + _CJK + r"]|_")


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


def fix_spacing(text: str) -> str:
    """No space between Han chars; one space between Han and Latin (so English words get ▁ pieces)."""
    text = re.sub(rf"(?<=[{_CJK}])\s+(?=[{_CJK}])", "", text)
    text = re.sub(rf"(?<=[{_CJK}])(?=[A-Za-z0-9])", " ", text)
    text = re.sub(rf"(?<=[A-Za-z0-9'])(?=[{_CJK}])", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str, style: str = "woitn") -> tuple[str, bool]:
    """Return (normalized text, contains Arabic digits).

    woitn (phase 1): drop punctuation (keep word-internal apostrophes), lowercase English; digits should be
        spoken form -- when some remain, the caller decides between TN and dropping the sample.
    withitn (phase 2): keep punctuation, case and digits.
    Traditional->simplified (OpenCC t2s) and TN are done by the data pipeline, not here.
    """
    text = unicodedata.normalize("NFC", to_halfwidth_alnum(text))
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
    return re.findall(rf"[{_CJK}]|[a-z]+(?:'[a-z]+)*|\d+", text)
