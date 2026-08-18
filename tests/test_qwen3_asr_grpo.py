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
