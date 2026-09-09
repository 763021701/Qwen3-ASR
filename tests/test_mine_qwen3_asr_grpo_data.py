from argparse import Namespace

from tools.mine_qwen3_asr_grpo_data import (
    MiningThresholds,
    classify_error_rates,
    oversample_records,
)


THRESHOLDS = MiningThresholds(
    easy_mean_max=0.15,
    easy_std_max=0.05,
    capable_best_max=0.30,
    hard_std_min=0.05,
    suspect_worst_min=1.0,
    suspect_best_min=0.60,
)


def test_classify_four_categories():
    assert classify_error_rates([0.02, 0.06], THRESHOLDS)[0] == "easy"
    # best at the capable boundary with real variance -> hard
    assert classify_error_rates([0.30, 0.60], THRESHOLDS)[0] == "hard"
    # deterministically wrong: even the best rollout is >= 60% wrong
    assert classify_error_rates([0.70, 0.76], THRESHOLDS)[0] == "suspect"
    # mediocre without variance and without a hallucination event
    assert classify_error_rates([0.50, 0.55], THRESHOLDS)[0] == "weak"


def test_occasional_full_hallucination_with_solvable_best_is_hard():
    """7/8 rollouts perfect, one full hallucination: the model can solve the
    clip and the group carries variance — GRPO material, not a bad-data
    suspect (2026-09-09 taxonomy redesign; the old catastrophic-first rule
    misfiled half of these as catastrophic)."""
    assert classify_error_rates([0.0] * 7 + [3.0], THRESHOLDS)[0] == "hard"


def test_always_hallucinating_is_suspect():
    """Even the best rollout fully hallucinates -> suspect (bad label/audio)."""
    assert classify_error_rates([1.2, 2.0], THRESHOLDS)[0] == "suspect"


def test_cantonese_word_boundaries_are_not_suspects():
    """Regression: a Cantonese digit string with one wrong character reads as
    WER >= 100% under word segmentation but is a small CER — the round-1
    mining labelled 79% of its catastrophic clips this way (2026-09-08 user
    audio verification). CER-based classification keeps such clips easy and
    out of the suspect list."""
    cers = [0.02, 0.06]  # one character wrong out of dozens
    assert classify_error_rates(cers, THRESHOLDS)[0] == "easy"


def test_easy_requires_low_variance_not_just_low_mean():
    # mean <= 0.15 but occasional moderate errors -> enough variance for hard
    assert classify_error_rates([0.0, 0.0, 0.4], THRESHOLDS)[0] == "hard"


def test_oversample_records_preserves_source_and_metadata():
    args = Namespace(
        weight_easy=0,
        weight_hard=3,
        weight_suspect=0,
        weight_weak=0,
    )
    source = {"audio": "/tmp/a.wav", "text": "language None<asr_text>hello"}
    sampled = oversample_records(
        [(source, "hard", {"mean_cer": 0.5, "best_cer": 0.1, "std_cer": 0.3})],
        args,
    )
    assert len(sampled) == 3
    assert all(row["grpo_category"] == "hard" for row in sampled)
    assert all(row["audio"] == source["audio"] for row in sampled)
    assert all(row["grpo_mining_mean_cer"] == 0.5 for row in sampled)


def test_oversample_records_skips_zero_weight_categories():
    args = Namespace(
        weight_easy=0,
        weight_hard=1,
        weight_suspect=0,
        weight_weak=0,
        dedupe_by_audio=1,
    )
    stats = {"mean_cer": 0.5, "best_cer": 0.1, "std_cer": 0.3}
    source = {"audio": "/tmp/a.wav", "text": "language None<asr_text>hello"}
    sampled = oversample_records(
        [(source, "suspect", stats), (source, "hard", stats)],
        args,
    )
    assert len(sampled) == 1
    assert sampled[0]["grpo_category"] == "hard"
