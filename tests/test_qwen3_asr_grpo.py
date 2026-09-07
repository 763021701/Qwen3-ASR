import torch

from finetuning.qwen3_asr_grpo import (
    asr_rewards,
    character_error_rate,
    completion_mask,
    extract_reference_text,
    group_advantages,
    word_error_rate,
)
from finetuning.qwen3_asr_sft import strip_target_brackets


def test_extract_reference_text_strips_asr_label():
    assert extract_reference_text("language None<asr_text>Hello world") == "Hello world"
    assert extract_reference_text("Hello world") == "Hello world"


def test_word_error_rate_uses_english_normalization():
    assert word_error_rate("Hello, World!", "hello world") == 0.0
    assert word_error_rate("one two", "one") == 0.5


def test_cer_is_punctuation_insensitive_and_smoother_than_wer_for_word_forms():
    assert character_error_rate("Block (N)", "Block N") == 0.0
    assert character_error_rate("hemithyroidectomy", "hysterectomy") < 1.0
    assert character_error_rate("hemithyroidectomy", "hysterectomy") < word_error_rate(
        "hemithyroidectomy", "hysterectomy"
    )


def test_mixed_asr_reward_preserves_wer_default_and_can_use_cer():
    completions = [
        "language None<asr_text>hysterectomy",
        "language None<asr_text>unrelated",
    ]
    references = ["hemithyroidectomy", "hemithyroidectomy"]
    wer_reward = asr_rewards(references, completions)
    cer_reward = asr_rewards(references, completions, cer_weight=1.0)
    assert wer_reward.tolist() == [-1.0, -1.0]
    assert cer_reward[0] < wer_reward[0]
    assert cer_reward[0] > cer_reward[1]


def test_strip_target_brackets_preserves_contents_and_asr_tag():
    assert strip_target_brackets("language English<asr_text>Block (N) [frozen] {A}") == (
        "language English<asr_text>Block N frozen A"
    )


def test_asr_rewards_prefers_lower_wer_completion():
    rewards = asr_rewards(
        ["scraped cytology smear", "scraped cytology smear"],
        [
            "language None<asr_text>Scraped cytology smear",
            "language None<asr_text>Specimen received in formalin",
        ],
    )
    assert rewards[0] == 0.0
    assert rewards[0] > rewards[1]


def test_group_advantages_are_centered_per_prompt():
    advantages = group_advantages(torch.tensor([0.0, -1.0, -2.0, -2.0]), 2)
    assert torch.allclose(advantages[:2].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(advantages[2:], torch.zeros(2), atol=1e-6)


def test_completion_mask_keeps_first_eos_and_removes_following_tokens():
    ids = torch.tensor([[7, 8, 151645, 9], [7, 8, 9, 10]])
    mask = completion_mask(ids, [151645, 151643])
    assert mask.tolist() == [[True, True, True, False], [True, True, True, True]]


def test_cer_decomposition_counts_sub_del_ins():
    from finetuning.qwen3_asr_grpo import cer_decomposition

    # 5 inserted chars ("brave"), punctuation/case normalized away
    assert cer_decomposition("Hello, world.", "hello brave world") == (0, 0, 5, 10)
    # 1 deletion
    assert cer_decomposition("hello world", "hello worl") == (0, 1, 0, 10)
    # 1 substitution
    assert cer_decomposition("hello world", "hello wovld") == (1, 0, 0, 10)
    # empty hypothesis: all deletions
    assert cer_decomposition("hello", "") == (0, 5, 0, 5)
    # empty reference: insertions with ref_len 0
    assert cer_decomposition("", "hello") == (0, 0, 5, 0)


def test_loop_penalty_ignores_repeats_present_in_reference():
    from finetuning.qwen3_asr_grpo import loop_penalty_ratio

    # exact copy: every repeat is licensed by the reference
    assert loop_penalty_ratio("aa bb aa bb aa bb", "aa bb aa bb aa bb") == 0.0
    # no repetition
    assert loop_penalty_ratio("abcdefgh", "abcdefgh") == 0.0
    # novel repeated tail absent from the reference
    penalized = loop_penalty_ratio(
        "cervix end of dictation blah blah blah blah", "cervix end of dictation"
    )
    assert penalized > 0.0
    # short n-grams below min_ngram are not flagged ("cm cm" spacing artifacts)
    assert loop_penalty_ratio("5 cm 3 cm", "5 cm 3 cm", min_ngram=4) == 0.0


def test_weighted_cer_loop_penalizes_insertion_more_than_deletion():
    reference = "the specimen weighs thirty five grams"
    complete = f"language None<asr_text>{reference}"
    insertion = f"language None<asr_text>{reference} period period period"
    deletion = f"language None<asr_text>{reference[:-7]}"
    rewards = asr_rewards(
        [reference] * 3,
        [complete, insertion, deletion],
        reward_mode="weighted_cer_loop",
    )
    assert rewards[0] == 0.0
    # insertion-weighted objective must rank insertion worse than deletion
    assert rewards[1] < rewards[2]


def test_asr_rewards_weighted_cer_loop_returns_stats():
    reference = "received in formalin"
    completion = f"language None<asr_text>{reference} blah blah blah blah"
    rewards, stats = asr_rewards(
        [reference, reference],
        [completion, f"language None<asr_text>{reference}"],
        reward_mode="weighted_cer_loop",
        return_stats=True,
    )
    assert len(rewards) == 2
    for values in stats.values():
        assert len(values) == 2
    assert stats["loop_pen"][0] > 0.0
    assert stats["loop_pen"][1] == 0.0
    assert stats["ins_ratio"][1] == 0.0


def test_asr_rewards_legacy_modes_unchanged_by_new_kwargs():
    completions = [
        "language None<asr_text>hysterectomy",
        "language None<asr_text>unrelated",
    ]
    references = ["hemithyroidectomy", "hemithyroidectomy"]
    legacy = asr_rewards(references, completions)
    with_kwargs = asr_rewards(
        references,
        completions,
        insertion_weight=2.0,
        deletion_weight=0.5,
        loop_weight=1.0,
    )
    assert torch.equal(legacy, with_kwargs)
