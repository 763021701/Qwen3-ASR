import os

import pytest
import torch

from finetuning.ctc_auxiliary import (
    CTCHead,
    CTCVocabulary,
    SenseVoiceCTCTokenizer,
    compute_ctc_loss,
    ctc_greedy_decode,
    normalize_ctc_text,
    transcript_from_target,
)

SENSEVOICE_TOKENIZER = os.environ.get(
    "SENSEVOICE_TOKENIZER",
    "/root/.cache/huggingface/hub/models--FunAudioLLM--SenseVoiceSmall/snapshots/"
    "3847d57b6bdf2dd8875cb1508d2af43d80a16bf7/chn_jpn_yue_eng_ko_spectok.bpe.model",
)
requires_tokenizer = pytest.mark.skipif(
    not os.path.isfile(SENSEVOICE_TOKENIZER),
    reason="SenseVoice tokenizer not available",
)


def test_transcript_from_target_strips_language_prefix():
    assert (
        transcript_from_target("language None<asr_text>hello world")
        == "hello world"
    )
    assert transcript_from_target("plain text") == "plain text"


def test_normalize_ctc_text_removes_case_and_punctuation():
    assert (
        normalize_ctc_text("language None<asr_text>SN1 and non-SN, 2.5 cm.")
        == "sn1 and non sn 2.5 cm"
    )
    assert (
        normalize_ctc_text("language English<asr_text>Cranial base, normal.")
        == "cranial base normal"
    )
    assert (
        normalize_ctc_text("language None<asr_text>盧嘅啦, 零二三九三二七.")
        == "盧嘅啦 零二三九三二七"
    )


@requires_tokenizer
def test_sensevoice_tokenizer_matches_sensevoice_vocab():
    tokenizer = SenseVoiceCTCTokenizer(SENSEVOICE_TOKENIZER)
    assert tokenizer.vocab_size == 25055
    assert tokenizer.blank_id == 0
    ids = tokenizer.encode("coronal full slabs")
    assert ids and all(idx != 0 for idx in ids)
    assert tokenizer.decode(ids) == "coronal full slabs"


@requires_tokenizer
def test_ctc_vocabulary_roundtrip(tmp_path):
    tokenizer = SenseVoiceCTCTokenizer(SENSEVOICE_TOKENIZER)
    texts = [
        "language None<asr_text>Cranial base, normal.",
        "language None<asr_text>The the the SN1 and non-SN.",
    ]
    vocab = CTCVocabulary.from_texts(tokenizer, texts)
    assert vocab.blank_id == 0
    assert vocab.size == len(vocab.source_ids) + 1
    assert vocab.size > 1

    compact = vocab.encode(texts[0])
    assert compact
    assert all(1 <= token < vocab.size for token in compact)
    assert vocab.decode(compact) == "cranial base normal"

    path = tmp_path / "ctc_vocab.json"
    vocab.save(str(path))
    loaded = CTCVocabulary.load(str(path), tokenizer)
    assert loaded.source_ids == vocab.source_ids
    assert loaded.decode(compact) == "cranial base normal"


@requires_tokenizer
def test_ctc_vocabulary_drops_unseen_tokens():
    tokenizer = SenseVoiceCTCTokenizer(SENSEVOICE_TOKENIZER)
    vocab = CTCVocabulary.from_texts(tokenizer, ["language None<asr_text>hello"])
    assert vocab.encode("language None<asr_text>hello 阿奇霉素") == vocab.encode(
        "language None<asr_text>hello"
    )


def test_ctc_head_loss_and_gradients():
    torch.manual_seed(0)
    head = CTCHead(hidden_size=16, vocab_size=32)
    hidden_states = [torch.randn(20, 16), torch.randn(30, 16)]
    targets = torch.tensor([[3, 4, 5, 0, 0], [6, 7, 8, 9, 10]])
    target_lengths = torch.tensor([3, 5])

    loss = compute_ctc_loss(hidden_states, head, targets, target_lengths)
    assert torch.isfinite(loss)
    loss.backward()
    assert head.proj[-1].weight.grad is not None
    assert torch.isfinite(head.proj[-1].weight.grad).all()


def test_ctc_coverage_detach_shape():
    torch.manual_seed(0)
    head = CTCHead(hidden_size=16, vocab_size=32)
    hidden_states = [torch.randn(20, 16), torch.randn(30, 16)]
    targets = torch.tensor([[3, 4, 5, 0, 0], [6, 7, 8, 9, 10]])
    target_lengths = torch.tensor([3, 5])
    loss = compute_ctc_loss(
        hidden_states,
        head,
        targets,
        target_lengths,
        coverage_weight=0.01,
        coverage_detach=True,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert head.proj[-1].weight.grad is not None


def test_ctc_loss_rejects_target_longer_than_audio():
    head = CTCHead(hidden_size=8, vocab_size=16)
    hidden_states = [torch.randn(3, 8)]
    targets = torch.tensor([[1, 2, 3, 4]])
    with pytest.raises(ValueError, match="CTC target longer"):
        compute_ctc_loss(hidden_states, head, targets, torch.tensor([4]))


def test_ctc_greedy_decode_collapses_repeats():
    logits = torch.full((1, 8, 4), -10.0)
    sequence = [1, 1, 0, 2, 2, 3, 0, 3]
    for frame, token in enumerate(sequence):
        logits[0, frame, token] = 10.0
    decoded = ctc_greedy_decode(logits, torch.tensor([8]))
    assert decoded == [[1, 0, 2, 3, 0, 3]]