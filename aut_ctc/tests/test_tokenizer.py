"""Tokenizer / transcript cleaning tests (DESIGN.md §15 T7, T8, T11; T9/T10 need the real vocab).

  python -m aut_ctc.tests.test_tokenizer                                   # synthetic SentencePiece vocab
  python -m aut_ctc.tests.test_tokenizer --sp_model /path/to/chn_jpn_yue_eng_ko_spectok.bpe.model \
      [--texts train_texts.txt]                                            # real vocab: T7-T10
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from ..data import clean_transcript
from ..tokenizer import (DEFAULT_EXTRA_CHARS, DEFAULT_SPECIALS, DIALECT_TAGS, load_tokenizer, mer_tokens,
                         normalize_text, piece_class, read_extra_chars)
from .common import synthetic_sp_model

GOLDEN = ["我们用 python 写一个 demo", "这个 feature 下周上线", "今天天气不错", "hello world"]


def round_trip(tok, text):
    return tok.decode(tok.encode(text))


def check_tokenizer(tok, real: bool):
    # T7: vocab size and tags
    if real:
        expected = 19911 + len(DIALECT_TAGS) + 1 + len(tok.extra_chars)  # DESIGN §6.3 + dialect tags + ▁ + extras
        assert tok.vocab_size == expected, (tok.vocab_size, expected)
    assert all(tok.id_to_piece(tok.tag_id(t)) == t for t in DEFAULT_SPECIALS)
    # T8: no blank, no isolated ▁ next to Han text, OOV -> 1, round trip
    for s in GOLDEN:
        ids = tok.encode(s)
        assert ids and 0 not in ids and max(ids) < tok.vocab_size, (s, ids)
        assert all(tok.id_to_piece(i) != "▁" for i in ids)
        assert round_trip(tok, s) == s, (s, round_trip(tok, s))
        print(f"     {s!r:28s} -> {[tok.id_to_piece(i) for i in ids]}")
    assert tok.UNK in tok.encode("肜") and set(tok.encode("안녕")) == {tok.UNK}
    print(f"[ok] T7 V={tok.vocab_size}, all {len(DEFAULT_SPECIALS)} tags have ids; T8 no 0 / no isolated ▁ next to Han"
          " / OOV and Korean -> <unk> / exact round trip")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sp_model", default=None)
    p.add_argument("--texts", default=None, help="T10: one raw transcript per line (>= 1000 lines)")
    args = p.parse_args()

    # T11: normalization
    t, has_digit = normalize_text("Hello， World! 这是 一个 测试。我用Python写Demo")
    assert t == "hello world 这是一个测试我用 python 写 demo" and not has_digit, t
    t2, d2 = normalize_text("会议定在3点", "woitn")
    assert d2 and t2 == "会议定在 3 点", t2
    assert normalize_text("Café naïve Straße, Ærø") == ("cafe naive strasse aero", False)
    assert mer_tokens("我们用python写demo！") == ["我", "们", "用", "python", "写", "demo"]
    assert mer_tokens("𠮶个人") == ["𠮶", "个", "人"]
    assert clean_transcript("会议定在3点，价格1.5元", "zh") == ("会议定在三点价格一点五元", "ok")
    assert clean_transcript("I have 3 apples.", "en") == (None, "digits_without_tn")
    assert clean_transcript("<SIL> Hello, [noise] world <COMMA>", "en") == ("hello world", "ok")
    assert clean_transcript("iPhone 15 发布了", None) == ("iphone 十五发布了", "ok")
    assert clean_transcript("", "zh") == ("", "ok")
    print("[ok] T11 normalize_text (accents folded) / mer_tokens (Ext-B chars) / clean_transcript (Chinese TN, "
          "English digits dropped, markup removed)")

    tmp = Path(tempfile.mkdtemp())
    synthetic = load_tokenizer(synthetic_sp_model(tmp / "synthetic.model"))
    print(f"synthetic vocab ({synthetic.sp.get_piece_size()} pieces):")
    check_tokenizer(synthetic, real=False)
    synthetic.save_vocab(tmp / "vocab.json", tmp / "synthetic.model")
    load_tokenizer(tmp / "synthetic.model", tmp / "vocab.json")  # same model: accepted
    other = synthetic_sp_model(tmp / "other.model", extra_lines=["另一个不同的语料"] * 50, vocab_size=380)
    try:
        load_tokenizer(other, tmp / "vocab.json")
        raise AssertionError("vocab.json from another model must be rejected")
    except ValueError:
        pass
    print("[ok] vocab.json records the source model hash; a different model is rejected")

    if args.sp_model:
        from sentencepiece import sentencepiece_model_pb2

        extra = read_extra_chars(DEFAULT_EXTRA_CHARS)
        real = load_tokenizer(args.sp_model, extra_chars=DEFAULT_EXTRA_CHARS)
        proto = sentencepiece_model_pb2.ModelProto()
        proto.ParseFromString(real.sp.serialized_model_proto())
        ns = proto.normalizer_spec
        print(f"real vocab: {real.sp.get_piece_size()} pieces + {len(extra)} extra chars; normalizer={ns.name} "
              f"add_dummy_prefix={ns.add_dummy_prefix} remove_extra_whitespaces={ns.remove_extra_whitespaces}")
        print(f"     full-width punctuation '，！（）' -> {real.sp.encode('，！（）', out_type=str)}")
        print("     code-switch golden cases (T9):")
        check_tokenizer(real, real=True)
        # Words without a ▁-initial piece keep their boundary; the first word is segmented like any other.
        for s in ["i use pytorch and qwen daily", "the russian llm", "speech recognition"]:
            assert round_trip(real, s) == s, (s, round_trip(real, s))
        assert [real.id_to_piece(i) for i in real.encode("speech recognition")] == ["▁speech", "▁recognition"]
        # Extra characters: own ids, exact round trip, never <unk>
        for s in ["侬搿个物事汏一汏", "我揾咗佢好耐", "𠊎係客家人", "伊真媠", "𠮶个人"]:
            ids = real.encode(s)
            assert real.UNK not in ids and round_trip(real, s) == s, (s, ids, round_trip(real, s))
        tags = [t for t in DIALECT_TAGS if real.id_to_piece(real.tag_id(t)) == t]
        print(f"[ok] English: 'and qwen' / 'russian' keep their word boundary, first word -> ▁speech; "
              f"{len(extra)} extra chars round-trip; dialect tags {tags}")
        real.save_vocab(tmp / "real_vocab.json", args.sp_model)
        (tmp / "fewer.txt").write_text("\n".join(extra[:-1]), encoding="utf-8")
        try:
            load_tokenizer(args.sp_model, tmp / "real_vocab.json", tmp / "fewer.txt")
            raise AssertionError("a different extra-character list must be rejected")
        except ValueError:
            pass
        assert load_tokenizer(args.sp_model, tmp / "real_vocab.json").vocab_size == real.vocab_size
        print("[ok] vocab.json records the extra characters; a different list is rejected")
        if args.texts:  # T10, with the same cleaning (TN included) as training
            lines = [line.strip() for line in open(args.texts, encoding="utf-8") if line.strip()][:1000]
            bad, unk, n = 0, 0, 0
            for line in lines:
                text, _ = clean_transcript(line)
                if text is None:
                    continue
                n += 1
                ids = real.encode(text)
                unk += real.UNK in ids
                if real.UNK not in ids and real.decode(ids) != text:
                    bad += 1
            print(f"[{'ok' if bad == 0 else 'FAIL'}] T10 round trip on {n} texts: {bad} mismatches, {unk} with <unk>")
            assert bad == 0
        kinds = {piece_class(real.pieces[o]) for o in real.new2old[2:]}
        print(f"[ok] T9 real vocab checks passed; kept piece classes {sorted(kinds)}")
    print("ALL TOKENIZER TESTS PASSED")


if __name__ == "__main__":
    main()
