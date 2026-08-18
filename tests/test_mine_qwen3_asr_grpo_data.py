from argparse import Namespace

from tools.mine_qwen3_asr_grpo_data import (
    MiningThresholds,
    classify_wers,
    oversample_records,
)


THRESHOLDS = MiningThresholds(
    easy_mean_wer_max=0.15,
    easy_std_wer_max=0.05,
    recoverable_mean_wer_min=0.35,
    recoverable_best_wer_max=0.20,
    recoverable_std_wer_min=0.15,
    unstable_mean_wer_min=0.15,
    unstable_mean_wer_max=0.50,
    unstable_std_wer_min=0.12,
    wrong_best_wer_min=0.60,
    wrong_std_wer_max=0.10,
    catastrophic_wer_min=1.0,
)


def test_classify_requested_grpo_mining_categories():
    assert classify_wers([0.02, 0.06], THRESHOLDS)[0] == "easy"
    assert classify_wers([0.10, 0.80], THRESHOLDS)[0] == "recoverable"
    assert classify_wers([0.30, 0.60], THRESHOLDS)[0] == "unstable"
    assert classify_wers([0.70, 0.76], THRESHOLDS)[0] == "consistently_wrong"
    assert classify_wers([0.05, 1.20], THRESHOLDS)[0] == "catastrophic_hallucination"


def test_catastrophic_takes_priority_over_recoverable():
    assert classify_wers([0.0, 1.2], THRESHOLDS)[0] == "catastrophic_hallucination"


def test_oversample_records_preserves_source_and_metadata():
    args = Namespace(
        weight_easy=1,
        weight_recoverable=3,
        weight_unstable=1,
        weight_consistently_wrong=1,
        weight_catastrophic_hallucination=1,
        weight_mixed=1,
    )
    source = {"audio": "/tmp/a.wav", "text": "language None<asr_text>hello"}
    sampled = oversample_records(
        [(source, "recoverable", {"mean_wer": 0.5, "best_wer": 0.1, "std_wer": 0.3})],
        args,
    )
    assert len(sampled) == 3
    assert all(row["grpo_category"] == "recoverable" for row in sampled)
    assert all(row["audio"] == source["audio"] for row in sampled)


def test_oversample_records_can_dedupe_exact_audio_paths():
    args = Namespace(
        weight_easy=0,
        weight_recoverable=1,
        weight_unstable=1,
        weight_consistently_wrong=0,
        weight_catastrophic_hallucination=1,
        weight_mixed=1,
        dedupe_by_audio=1,
    )
    stats = {"mean_wer": 0.5, "best_wer": 0.1, "std_wer": 0.3}
    source = {"audio": "/tmp/a.wav", "text": "language None<asr_text>hello"}
    sampled = oversample_records(
        [(source, "recoverable", stats), (source, "catastrophic_hallucination", stats)],
        args,
    )
    assert len(sampled) == 1
    assert sampled[0]["grpo_category"] == "recoverable"
