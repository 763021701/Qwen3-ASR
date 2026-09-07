#!/usr/bin/env python3
# coding=utf-8
"""GRPO post-training for Qwen3-ASR.

By default only the text decoder and LM head are optimized (audio tower and
aligner frozen); ``--freeze_modules`` controls freezing per part
(encoder / aligner / llm). When an audio part is trainable, the training-time
forward recomputes audio features through the live audio tower so those
parameters receive gradients; rollout sampling always reuses one shared
encoding per group. Optimized against a group-relative, reference WER/CER
reward. The implementation follows the rollout -> group advantage -> clipped
policy loss plus reference-KL structure used by minimind's GRPO trainer.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import GenerationConfig

from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import parse_asr_output

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evaluation.english_medical.text_normalization import normalize_english

if __package__:
    from .module_freeze import (PARTS, count_part_parameters, parse_parts,
                                set_part_freeze)
else:
    from module_freeze import (PARTS, count_part_parameters, parse_parts,
                               set_part_freeze)


_ASR_TEXT_TAG = "<asr_text>"
_WS_RE = re.compile(r"\s+")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_reference_text(label: str) -> str:
    """Return the transcript portion of a Qwen3-ASR target label."""
    value = (label or "").strip()
    if _ASR_TEXT_TAG in value:
        return value.split(_ASR_TEXT_TAG, 1)[1].strip()
    return value


def normalize_english_asr_text(text: str) -> str:
    """Match the existing English ASR training/evaluation normalization style."""
    value = (text or "").lower()
    value = value.translate(str.maketrans("", "", string.punctuation))
    return _WS_RE.sub(" ", value).strip()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute word error rate without an extra runtime dependency."""
    ref_words = normalize_english_asr_text(reference).split()
    hyp_words = normalize_english_asr_text(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    previous = list(range(len(hyp_words) + 1))
    for ref_index, ref_word in enumerate(ref_words, start=1):
        current = [ref_index]
        for hyp_index, hyp_word in enumerate(hyp_words, start=1):
            current.append(
                min(
                    previous[hyp_index] + 1,
                    current[hyp_index - 1] + 1,
                    previous[hyp_index - 1] + (ref_word != hyp_word),
                )
            )
        previous = current
    return previous[-1] / len(ref_words)


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Compute punctuation-insensitive character error rate for ASR text."""
    ref_chars = normalize_english_asr_text(reference).replace(" ", "")
    hyp_chars = normalize_english_asr_text(hypothesis).replace(" ", "")
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0

    previous = list(range(len(hyp_chars) + 1))
    for ref_index, ref_char in enumerate(ref_chars, start=1):
        current = [ref_index]
        for hyp_index, hyp_char in enumerate(hyp_chars, start=1):
            current.append(
                min(
                    previous[hyp_index] + 1,
                    current[hyp_index - 1] + 1,
                    previous[hyp_index - 1] + (ref_char != hyp_char),
                )
            )
        previous = current
    return previous[-1] / len(ref_chars)


def asr_rewards(
    references: Sequence[str],
    raw_completions: Sequence[str],
    reward_mode: str = "wer_cer",
    cer_weight: float = 0.0,
    *,
    insertion_weight: float = 1.5,
    deletion_weight: float = 1.0,
    loop_weight: float = 0.5,
    loop_min_ngram: int = 4,
    loop_max_ngram: int = 48,
    return_stats: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, List[float]]]:
    """Return negative WER/CER rewards for generated ASR completions.

    ``weighted_cer_loop`` is the hallucination-guarded objective:
    ``-( (S + deletion_weight*D + insertion_weight*I)/N + loop_weight*loop_pen )``
    where ``loop_pen`` is the over-repeated-span coverage of the hypothesis.
    With ``return_stats=True`` it also returns per-sample component values
    (weighted CER, insertion/deletion ratios, loop penalty, length ratio) for
    reward monitoring.
    """
    if len(references) != len(raw_completions):
        raise ValueError("references and raw_completions must have the same length.")
    if not 0.0 <= cer_weight <= 1.0:
        raise ValueError("cer_weight must be in [0, 1].")
    if reward_mode not in {"wer", "wer_cer", "cer", "weighted_cer_loop"}:
        raise ValueError(
            "reward_mode must be one of: wer, wer_cer, cer, weighted_cer_loop."
        )
    if insertion_weight < 0 or deletion_weight < 0 or loop_weight < 0:
        raise ValueError("insertion/deletion/loop weights must be non-negative.")
    if loop_min_ngram < 2 or loop_max_ngram < loop_min_ngram:
        raise ValueError("loop n-gram range must satisfy 2 <= min <= max.")

    rewards: List[float] = []
    stats: Dict[str, List[float]] = {
        "weighted_cer": [],
        "ins_ratio": [],
        "del_ratio": [],
        "loop_pen": [],
        "len_ratio": [],
    }
    for reference, raw_completion in zip(references, raw_completions):
        try:
            _, hypothesis = parse_asr_output(raw_completion, user_language=None)
        except Exception:
            hypothesis = ""
        wer = word_error_rate(reference, hypothesis)
        cer = character_error_rate(reference, hypothesis)
        if reward_mode == "weighted_cer_loop":
            subs, dels, ins, n = cer_decomposition(reference, hypothesis)
            if n:
                weighted_cer = (
                    subs + deletion_weight * dels + insertion_weight * ins
                ) / n
            else:
                weighted_cer = 0.0 if not hypothesis else 1.0
            loop_pen = loop_penalty_ratio(
                hypothesis, reference, loop_min_ngram, loop_max_ngram
            )
            rewards.append(-(weighted_cer + loop_weight * loop_pen))
            ref_chars = len(normalize_english(reference).replace(" ", ""))
            hyp_chars = len(normalize_english(hypothesis).replace(" ", ""))
            stats["weighted_cer"].append(weighted_cer)
            stats["ins_ratio"].append(ins / n if n else 0.0)
            stats["del_ratio"].append(dels / n if n else 0.0)
            stats["loop_pen"].append(loop_pen)
            stats["len_ratio"].append(
                hyp_chars / ref_chars if ref_chars else float(hyp_chars > 0)
            )
        elif reward_mode == "cer":
            rewards.append(-cer)
        elif reward_mode == "wer":
            rewards.append(-wer)
        else:
            rewards.append(-(wer + cer_weight * cer))
    tensor = torch.tensor(rewards, dtype=torch.float32)
    if return_stats:
        return tensor, stats
    return tensor


def cer_decomposition(reference: str, hypothesis: str) -> Tuple[int, int, int, int]:
    """(substitutions, deletions, insertions, ref_len) on eval-normalized chars.

    Uses the shared evaluation normalization (numerals, hyphens, dictionary
    compounds, ...) so the reward optimizes the same surface as the selection
    metric, then drops spaces for a character-level alignment.
    """
    ref = normalize_english(reference).replace(" ", "")
    hyp = normalize_english(hypothesis).replace(" ", "")
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0, 0, m, 0
    if m == 0:
        return 0, n, 0, n

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        row = dp[i]
        prev = dp[i - 1]
        row[0] = i
        rc = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if rc == hyp[j - 1] else 1
            row[j] = min(prev[j - 1] + cost, prev[j] + 1, row[j - 1] + 1)

    subs = dels = inss = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i and j and dp[i][j] == dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            if ref[i - 1] != hyp[j - 1]:
                subs += 1
            i -= 1
            j -= 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            inss += 1
            j -= 1
    return subs, dels, inss, n


def _over_repeat_coverage(hyp: str, ref: str, min_ngram: int, max_ngram: int) -> float:
    """Fraction of hypothesis chars inside over-repeated n-grams.

    An n-gram is over-repeated when it occurs more often in the hypothesis
    than in the reference, so legitimate repeated report vocabulary that the
    reference also repeats is never penalized. Spans are counted once
    (greedy longest first) so a long repeated tail is not double-counted
    through its substrings.
    """
    hyp_len = len(hyp)
    if hyp_len < min_ngram:
        return 0.0

    ref_counts: Dict[str, int] = {}
    for n in range(min_ngram, min(max_ngram, len(ref)) + 1):
        for i in range(len(ref) - n + 1):
            gram = ref[i:i + n]
            ref_counts[gram] = ref_counts.get(gram, 0) + 1
    hyp_counts: Dict[str, int] = {}
    for n in range(min_ngram, min(max_ngram, hyp_len) + 1):
        for i in range(hyp_len - n + 1):
            gram = hyp[i:i + n]
            hyp_counts[gram] = hyp_counts.get(gram, 0) + 1

    covered = 0
    i = 0
    while i + min_ngram <= hyp_len:
        best = 0
        for n in range(min_ngram, min(max_ngram, hyp_len - i) + 1):
            gram = hyp[i:i + n]
            if hyp_counts[gram] > ref_counts.get(gram, 0):
                best = n
        if best:
            covered += best
            i += best
        else:
            i += 1
    return covered / hyp_len


def loop_penalty_ratio(
    hypothesis: str, reference: str, min_ngram: int = 4, max_ngram: int = 48
) -> float:
    """Cyclic-hallucination penalty: over-repeated span coverage of the hypothesis."""
    return _over_repeat_coverage(
        normalize_english(hypothesis).replace(" ", ""),
        normalize_english(reference).replace(" ", ""),
        min_ngram,
        max_ngram,
    )


def group_advantages(rewards: torch.Tensor, num_generations: int) -> torch.Tensor:
    """Standardize each prompt's G rewards independently, as in GRPO."""
    if rewards.ndim != 1 or rewards.numel() % num_generations:
        raise ValueError("rewards must be a flat tensor divisible by num_generations.")
    grouped = rewards.view(-1, num_generations)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True, unbiased=False)
    return ((grouped - mean) / (std + 1e-4)).reshape(-1)


def completion_mask(completion_ids: torch.Tensor, eos_token_ids: Sequence[int]) -> torch.Tensor:
    """Mask tokens through the first EOS; use all tokens when generation hits its cap."""
    if completion_ids.ndim != 2:
        raise ValueError("completion_ids must have shape [batch, completion_length].")
    eos = torch.zeros_like(completion_ids, dtype=torch.bool)
    for token_id in eos_token_ids:
        eos |= completion_ids.eq(token_id)
    positions = torch.arange(completion_ids.shape[1], device=completion_ids.device).unsqueeze(0)
    end = torch.full(
        (completion_ids.shape[0],),
        completion_ids.shape[1] - 1,
        device=completion_ids.device,
        dtype=torch.long,
    )
    has_eos = eos.any(dim=1)
    end[has_eos] = eos[has_eos].int().argmax(dim=1)
    return positions <= end.unsqueeze(1)


def _load_audio(path: str, sample_rate: int) -> np.ndarray:
    waveform, _ = librosa.load(path, sr=sample_rate, mono=True)
    return waveform.astype(np.float32, copy=False)


def _prefix_text(processor: Any, prompt: str) -> str:
    messages = [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": None}]},
    ]
    return processor.apply_chat_template(
        [messages], add_generation_prompt=True, tokenize=False
    )[0]


def _merge_audio_embeddings(
    thinker: Any,
    input_ids: torch.Tensor,
    audio_features: torch.Tensor,
) -> torch.Tensor:
    inputs_embeds = thinker.get_input_embeddings()(input_ids)
    audio_mask = input_ids.eq(thinker.config.audio_token_id)
    placeholder_count = int(audio_mask.sum().item())
    if placeholder_count != audio_features.shape[0]:
        raise RuntimeError(
            "Audio placeholder count does not match encoded audio features: "
            f"{placeholder_count} != {audio_features.shape[0]}."
        )
    return inputs_embeds.masked_scatter(
        audio_mask.unsqueeze(-1).expand_as(inputs_embeds),
        audio_features.to(dtype=inputs_embeds.dtype),
    )


def _repeat_audio_features(
    audio_features: torch.Tensor,
    prefix_ids: torch.Tensor,
    audio_token_id: int,
    num_generations: int,
) -> torch.Tensor:
    counts = prefix_ids.eq(audio_token_id).sum(dim=1).tolist()
    if sum(counts) != audio_features.shape[0]:
        raise RuntimeError("Cannot split audio features by audio placeholder counts.")
    chunks = torch.split(audio_features, counts, dim=0)
    return torch.cat(
        [chunk for chunk in chunks for _ in range(num_generations)], dim=0
    )


@dataclass
class RolloutBatch:
    full_ids: torch.Tensor
    full_attention_mask: torch.Tensor
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    repeated_audio_features: torch.Tensor
    repeated_feature_attention_mask: torch.Tensor
    raw_completions: List[str]
    num_generations: int


@torch.no_grad()
def rollout_groups(
    thinker: Any,
    processor: Any,
    prefix_inputs: Dict[str, torch.Tensor],
    num_generations: int,
    generation_config: GenerationConfig,
) -> RolloutBatch:
    """Encode each audio once, then sample G completions from the same prefix."""
    input_ids = prefix_inputs["input_ids"]
    attention_mask = prefix_inputs["attention_mask"]
    feature_attention_mask = prefix_inputs["feature_attention_mask"]
    audio_features = thinker.get_audio_features(
        prefix_inputs["input_features"],
        feature_attention_mask=feature_attention_mask,
    )
    prefix_embeds = _merge_audio_embeddings(thinker, input_ids, audio_features)

    repeated_ids = input_ids.repeat_interleave(num_generations, dim=0)
    repeated_attention = attention_mask.repeat_interleave(num_generations, dim=0)
    repeated_embeds = prefix_embeds.repeat_interleave(num_generations, dim=0)
    repeated_feature_attention = feature_attention_mask.repeat_interleave(
        num_generations, dim=0
    )
    repeated_audio_features = _repeat_audio_features(
        audio_features,
        input_ids,
        thinker.config.audio_token_id,
        num_generations,
    ).detach()

    was_training = thinker.training
    thinker.eval()
    generated = thinker.generate(
        input_ids=repeated_ids,
        inputs_embeds=repeated_embeds,
        attention_mask=repeated_attention,
        input_features=None,
        feature_attention_mask=repeated_feature_attention,
        generation_config=generation_config,
    )
    if was_training:
        thinker.train()

    prompt_width = repeated_ids.shape[1]
    completion_ids = generated[:, prompt_width:]
    response_mask = completion_mask(
        completion_ids, generation_config.eos_token_id
    )
    full_attention = torch.cat(
        [repeated_attention, response_mask.to(dtype=repeated_attention.dtype)], dim=1
    )
    raw_completions = processor.batch_decode(
        completion_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return RolloutBatch(
        full_ids=generated.detach(),
        full_attention_mask=full_attention,
        completion_ids=completion_ids.detach(),
        completion_mask=response_mask,
        repeated_audio_features=repeated_audio_features,
        repeated_feature_attention_mask=repeated_feature_attention,
        raw_completions=list(raw_completions),
        num_generations=num_generations,
    )


def completion_logps(
    thinker: Any,
    rollout: RolloutBatch,
    live_audio: Tuple[torch.Tensor, torch.Tensor] = None,
) -> torch.Tensor:
    """Compute raw-model log p(completion token | prompt, audio evidence).

    With ``live_audio=(input_features, feature_attention_mask)`` the audio
    features are recomputed through the (possibly trainable) audio tower so
    encoder/aligner parameters receive gradients; otherwise the detached
    rollout-time features are reused.
    """
    if live_audio is None:
        audio_features = rollout.repeated_audio_features
    else:
        input_features, feature_attention_mask = live_audio
        audio_features = thinker.get_audio_features(
            input_features, feature_attention_mask=feature_attention_mask
        )
        prompt_width = rollout.full_ids.shape[1] - rollout.completion_ids.shape[1]
        audio_features = _repeat_audio_features(
            audio_features,
            rollout.full_ids[:, :prompt_width],
            thinker.config.audio_token_id,
            rollout.num_generations,
        )
    inputs_embeds = _merge_audio_embeddings(
        thinker, rollout.full_ids, audio_features
    )
    outputs = thinker(
        input_ids=rollout.full_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=rollout.full_attention_mask,
        input_features=None,
        feature_attention_mask=rollout.repeated_feature_attention_mask,
    )
    next_token_logps = F.log_softmax(outputs.logits[:, :-1, :].float(), dim=-1)
    selected = next_token_logps.gather(
        2, rollout.full_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    prompt_width = rollout.full_ids.shape[1] - rollout.completion_ids.shape[1]
    return selected[:, prompt_width - 1 : prompt_width - 1 + rollout.completion_ids.shape[1]]


class JsonlDataset(Dataset):
    def __init__(self, path: str, max_samples: int = 0) -> None:
        self.rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not row.get("audio") or "text" not in row:
                    raise ValueError(f"{path}:{line_number} requires audio and text fields.")
                if not os.path.isfile(row["audio"]):
                    raise FileNotFoundError(f"{path}:{line_number} missing audio: {row['audio']}")
                self.rows.append(row)
                if max_samples > 0 and len(self.rows) >= max_samples:
                    break
        if not self.rows:
            raise ValueError(f"No usable rows found in {path}.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.rows[index]


def _collate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return rows


def _prepare_prefix_inputs(
    rows: Sequence[Dict[str, Any]],
    processor: Any,
    device: torch.device,
    dtype: torch.dtype,
    sample_rate: int,
) -> Dict[str, torch.Tensor]:
    prompts = [_prefix_text(processor, str(row.get("prompt") or "")) for row in rows]
    audios = [_load_audio(str(row["audio"]), sample_rate) for row in rows]
    inputs = processor(text=prompts, audio=audios, return_tensors="pt", padding=True)
    return inputs.to(device).to(dtype)


def _resolve_frozen_parts(args: argparse.Namespace) -> List[str]:
    """Parts frozen for this run; unset --freeze_modules keeps the legacy LLM-only default."""
    if args.freeze_modules is None:
        return list(parse_parts("encoder,aligner"))
    return list(parse_parts(args.freeze_modules))


def _save_checkpoint(
    model: Any,
    processor: Any,
    optimizer: AdamW,
    output_dir: str,
    global_step: int,
    epoch: int,
) -> str:
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    generation_config = model.generation_config
    if not getattr(generation_config, "do_sample", True):
        generation_config.do_sample = True
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    processor.save_pretrained(checkpoint_dir)
    torch.save(
        {"optimizer": optimizer.state_dict(), "global_step": global_step, "epoch": epoch},
        os.path.join(checkpoint_dir, "grpo_trainer_state.pt"),
    )
    return checkpoint_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO post-training for Qwen3-ASR (default: LLM-only).")
    parser.add_argument("--model_path", required=True, help="SFT checkpoint or base Qwen3-ASR model path.")
    parser.add_argument("--train_file", required=True, help="JSONL with audio, text, and optional prompt.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--reference_model_path", default="", help="Defaults to model_path.")
    parser.add_argument("--batch_size", type=int, default=1, help="Prompts per rollout batch.")
    parser.add_argument("--num_generations", type=int, default=4, help="Samples per audio prompt.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_acc", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--beta", type=float, default=0.01, help="Reference-KL coefficient.")
    parser.add_argument(
        "--reward_mode",
        choices=("wer", "wer_cer", "cer", "weighted_cer_loop"),
        default="wer_cer",
        help="Reward objective: -WER, -(WER + cer_weight*CER), -CER, or the "
        "hallucination-guarded weighted CER with loop penalty.",
    )
    parser.add_argument(
        "--cer_weight",
        type=float,
        default=0.0,
        help="Additional CER penalty weight; 0 uses WER only.",
    )
    parser.add_argument(
        "--insertion_weight",
        type=float,
        default=1.5,
        help="weighted_cer_loop: weight for inserted characters (hallucinations).",
    )
    parser.add_argument(
        "--deletion_weight",
        type=float,
        default=1.0,
        help="weighted_cer_loop: weight for deleted characters.",
    )
    parser.add_argument(
        "--loop_weight",
        type=float,
        default=0.5,
        help="weighted_cer_loop: weight for the over-repeated-span (loop) penalty.",
    )
    parser.add_argument(
        "--loop_min_ngram",
        type=int,
        default=4,
        help="weighted_cer_loop: minimum character n-gram length for loop detection.",
    )
    parser.add_argument(
        "--loop_max_ngram",
        type=int,
        default=48,
        help="weighted_cer_loop: maximum character n-gram length for loop detection.",
    )
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO ratio clip range.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--log_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0, help="0 runs all epochs.")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--freeze_modules",
        default=None,
        help="Comma-separated parts to freeze: encoder,aligner,llm. Unset keeps the "
             "legacy default (encoder,aligner frozen; LLM trained). Empty string: no freeze. "
             "Note: trainable audio parts recompute audio features in the training forward.",
    )
    parser.add_argument("--dry_run", type=int, default=0, choices=(0, 1))
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch_size must be positive.")
    if args.num_generations < 2:
        raise ValueError("--num_generations must be at least 2 for group-relative rewards.")
    if args.grad_acc < 1:
        raise ValueError("--grad_acc must be positive.")
    if args.max_new_tokens < 1:
        raise ValueError("--max_new_tokens must be positive.")
    if args.temperature <= 0:
        raise ValueError("--temperature must be greater than zero when sampling.")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top_p must be in (0, 1].")
    if args.top_k < 0:
        raise ValueError("--top_k must be non-negative.")
    if not 0.0 <= args.cer_weight <= 1.0:
        raise ValueError("--cer_weight must be in [0, 1].")
    if args.insertion_weight < 0 or args.deletion_weight < 0 or args.loop_weight < 0:
        raise ValueError("--insertion_weight/--deletion_weight/--loop_weight must be non-negative.")
    if args.loop_min_ngram < 2 or args.loop_max_ngram < args.loop_min_ngram:
        raise ValueError("--loop_min_ngram/--loop_max_ngram must satisfy 2 <= min <= max.")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    seed_everything(args.seed)
    dataset = JsonlDataset(args.train_file, max_samples=args.max_samples)
    print(f"[data] rows={len(dataset)} train_file={args.train_file}")
    if args.dry_run:
        sample = dataset[0]
        print(
            "[dry-run] model=%s batch_size=%d num_generations=%d max_new_tokens=%d"
            % (args.model_path, args.batch_size, args.num_generations, args.max_new_tokens)
        )
        print("[dry-run] first_audio=%s" % sample["audio"])
        frozen_parts = _resolve_frozen_parts(args)
        trainable_parts = [p for p in PARTS if p not in frozen_parts]
        print(
            "[dry-run] reward=%s; frozen=%s; trainable=%s"
            % (
                "-CER"
                if args.reward_mode == "cer"
                else "-WER"
                if args.reward_mode == "wer"
                else "-(wCER + %g*loop), wCER=(S + %g*D + %g*I)/N"
                % (args.loop_weight, args.deletion_weight, args.insertion_weight)
                if args.reward_mode == "weighted_cer_loop"
                else "-(WER + %g*CER)" % args.cer_weight,
                ",".join(frozen_parts) or "none",
                ",".join(trainable_parts) or "none",
            )
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-ASR GRPO requires CUDA.")
    device = torch.device("cuda:0")
    use_bf16 = torch.cuda.get_device_capability(device)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    reference_path = args.reference_model_path or args.model_path

    print(f"[load] policy={args.model_path}")
    policy_wrapper = Qwen3ASRModel.from_pretrained(args.model_path, dtype=dtype, device_map=None)
    policy_model = policy_wrapper.model.to(device)
    processor = policy_wrapper.processor
    frozen_parts = _resolve_frozen_parts(args)
    set_part_freeze(policy_model, frozen_parts)
    part_counts = count_part_parameters(policy_model)
    for part in PARTS:
        total, trainable = part_counts[part]
        print(f"[freeze] {part}: total={total:,} trainable={trainable:,}")
    audio_trainable = "encoder" not in frozen_parts or "aligner" not in frozen_parts

    print(f"[load] reference={reference_path}")
    reference_wrapper = Qwen3ASRModel.from_pretrained(reference_path, dtype=dtype, device_map=None)
    reference_model = reference_wrapper.model.to(device).eval()
    reference_model.requires_grad_(False)
    reference_model.thinker.audio_tower.eval()

    optimizer = AdamW(
        [param for param in policy_model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate_rows,
    )
    eos_token_ids = policy_model.generation_config.eos_token_id or [151645, 151643]
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=eos_token_ids,
        pad_token_id=policy_model.generation_config.pad_token_id or eos_token_ids[-1],
        return_dict_in_generate=False,
    )

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    should_stop = False
    for epoch in range(args.epochs):
        for rows in loader:
            policy_model.eval()
            prefix_inputs = _prepare_prefix_inputs(rows, processor, device, dtype, args.sr)
            rollout = rollout_groups(
                policy_model.thinker,
                processor,
                prefix_inputs,
                args.num_generations,
                generation_config,
            )
            references = [
                extract_reference_text(str(row["text"]))
                for row in rows
                for _ in range(args.num_generations)
            ]
            rewards, reward_stats = asr_rewards(
                references,
                rollout.raw_completions,
                reward_mode=args.reward_mode,
                cer_weight=args.cer_weight,
                insertion_weight=args.insertion_weight,
                deletion_weight=args.deletion_weight,
                loop_weight=args.loop_weight,
                loop_min_ngram=args.loop_min_ngram,
                loop_max_ngram=args.loop_max_ngram,
                return_stats=args.reward_mode == "weighted_cer_loop",
            )
            rewards = rewards.to(device)
            advantages = group_advantages(rewards, args.num_generations)

            live_audio = (
                (prefix_inputs["input_features"], prefix_inputs["feature_attention_mask"])
                if audio_trainable else None
            )
            with torch.inference_mode():
                old_logps = completion_logps(policy_model.thinker, rollout, live_audio).detach()
                reference_logps = completion_logps(reference_model.thinker, rollout).detach()

            policy_model.train()
            if not audio_trainable:
                policy_model.thinker.audio_tower.eval()
            new_logps = completion_logps(policy_model.thinker, rollout, live_audio)
            mask = rollout.completion_mask.to(dtype=new_logps.dtype)
            ratio = torch.exp(new_logps - old_logps)
            unclipped = ratio * advantages.unsqueeze(1)
            clipped = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon) * advantages.unsqueeze(1)
            policy_term = -torch.minimum(unclipped, clipped)
            kl_delta = reference_logps - new_logps
            kl_term = torch.exp(kl_delta) - kl_delta - 1
            per_token_loss = policy_term + args.beta * kl_term
            sequence_loss = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            loss = sequence_loss.mean()
            (loss / args.grad_acc).backward()

            global_step += 1
            if global_step % args.grad_acc == 0:
                torch.nn.utils.clip_grad_norm_(
                    [param for param in policy_model.parameters() if param.requires_grad],
                    args.max_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if global_step % args.log_steps == 0:
                average_length = rollout.completion_mask.sum(dim=1).float().mean().item()
                average_kl = ((reference_logps - new_logps).detach() * mask).sum().item() / mask.sum().item()
                grouped = rewards.view(-1, args.num_generations)
                nondegenerate = (grouped.std(dim=1, unbiased=False) > 1e-6).float().mean().item()
                message = (
                    "[step %d] loss=%.6f reward=%.4f kl=%.5f len=%.1f nondegenerate=%.2f"
                    % (global_step, loss.item(), rewards.mean().item(), average_kl, average_length, nondegenerate)
                )
                if reward_stats is not None:
                    loop_pens = reward_stats["loop_pen"]
                    message += (
                        " wcer=%.4f insR=%.4f delR=%.4f loopR=%.4f loop%%=%.2f lenR=%.2f"
                        % (
                            sum(reward_stats["weighted_cer"]) / len(reward_stats["weighted_cer"]),
                            sum(reward_stats["ins_ratio"]) / len(reward_stats["ins_ratio"]),
                            sum(reward_stats["del_ratio"]) / len(reward_stats["del_ratio"]),
                            sum(loop_pens) / len(loop_pens),
                            sum(p > 0 for p in loop_pens) / len(loop_pens),
                            sum(reward_stats["len_ratio"]) / len(reward_stats["len_ratio"]),
                        )
                    )
                print(message)

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                policy_model.eval()
                path = _save_checkpoint(policy_model, processor, optimizer, args.output_dir, global_step, epoch)
                print(f"[save] {path}")

            del prefix_inputs, rollout, old_logps, reference_logps, new_logps, loss
            if args.max_steps > 0 and global_step >= args.max_steps:
                should_stop = True
                break
        if should_stop:
            break

    if global_step % args.grad_acc:
        torch.nn.utils.clip_grad_norm_(
            [param for param in policy_model.parameters() if param.requires_grad], args.max_grad_norm
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    policy_model.eval()
    final_path = _save_checkpoint(policy_model, processor, optimizer, args.output_dir, global_step, epoch)
    print(f"[done] final_checkpoint={final_path}")


if __name__ == "__main__":
    main()
