import random

from finetuning.qwen3_asr_raft import char_cer, classify_clip, select_raft_rows


def test_char_cer_uses_evaluation_normalization():
    assert char_cer("Hello, World.", "hello world") == 0.0
    assert char_cer("hello world", "hello brave world") > 0.0
    assert char_cer("", "hello") == 1.0


def test_classify_clip_buckets():
    assert classify_clip(best_cer=0.05, worst_cer=0.6, max_cer=0.15) == "reachable_hard"
    assert classify_clip(best_cer=0.0, worst_cer=0.0, max_cer=0.15) == "easy"
    assert classify_clip(best_cer=0.9, worst_cer=0.9, max_cer=0.15) == "unreachable"


def _clip(audio, reference, completions, aug=1, noise_aug=1):
    return {
        "audio": audio,
        "text": f"language None<asr_text>{reference}",
        "reference": reference,
        "completions": completions,
        "aug": aug,
        "noise_aug": noise_aug,
    }


def test_select_raft_rows_keeps_only_reachable_hard_with_flags():
    clips = [
        _clip("a.wav", "the liver shows steatosis", [
            "language None<asr_text>the liver shows steatosis",
            "language None<asr_text>the liver show steatosis",
        ]),
        _clip("b.wav", "uniform tan cut surface", [
            "language None<asr_text>uniform pink cut surface",
            "language None<asr_text>uniform brown cut surface",
        ], aug=1, noise_aug=0),
        _clip("c.wav", "received in formalin", [
            "language None<asr_text>received in formalin",
        ]),
        _clip("d.wav", "extensive cauterisation was performed", [
            "language None<asr_text>completely unrelated gibberish output",
            "language None<asr_text>more unrelated gibberish here",
        ]),
    ]
    rows, report = select_raft_rows(
        clips, max_cer=0.15, raft_weight=3, stabilizer_rows=0, rng=random.Random(0)
    )
    # a and b are solved sometimes (reachable-hard); c is all-perfect (easy);
    # d never gets close (unreachable)
    assert report["reachable_hard"] == 2 and report["kept_clips"] == 2
    assert report["easy"] == 1 and report["unreachable"] == 1
    assert report["mean_best_cer"] is not None and report["mean_best_cer"] < 0.15
    assert len(rows) == 6
    assert rows[0]["audio"] == "a.wav"
    assert rows[0]["aug"] == 1 and rows[0]["noise_aug"] == 1
    # target is the best sample's raw completion (prefix preserved)
    assert rows[0]["text"].startswith("language None<asr_text>")
    assert "steatosis" in rows[0]["text"]
    for row in rows:
        assert isinstance(row["aug"], int) and isinstance(row["noise_aug"], int)


def test_select_raft_rows_stabilizers_inherit_source_flags():
    clips = [
        _clip("c.wav", "received in formalin", [
            "language None<asr_text>received in formalin",
        ], aug=1, noise_aug=0),
    ]
    rows, report = select_raft_rows(
        clips, max_cer=0.15, raft_weight=3, stabilizer_rows=2, rng=random.Random(0)
    )
    assert report["easy"] == 1 and report["kept_clips"] == 0
    # no RAFT rows; stabilizers are reference-target copies of source rows
    # (capped at the number of available source clips)
    assert len(rows) == 1
    for row in rows:
        assert row["text"] == "language None<asr_text>received in formalin"
        assert row["noise_aug"] == 0
