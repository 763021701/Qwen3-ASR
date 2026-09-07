# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Full-parameter fine-tuning for Qwen3-ASR.
# For LoRA fine-tuning, see finetuning/qwen3_asr_sft_lora.py.
import argparse
import json
import math
import os
import random
import re
import shutil
import string
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
from datasets import load_dataset
from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import parse_asr_output
from transformers import (GenerationConfig, Trainer, TrainerCallback,
                          TrainingArguments)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

if __package__:
    from .module_freeze import (PARTS, count_part_parameters, parse_parts,
                                set_part_freeze)
    from .noise_augmentation import NoiseLibrary, maybe_add_noise
    from .nospeech_augmentation import NoSpeechAugmentConfig, apply_nospeech_augment
    from .processor_collate import build_processor_batch_inputs
else:
    from module_freeze import (PARTS, count_part_parameters, parse_parts,
                               set_part_freeze)
    from noise_augmentation import NoiseLibrary, maybe_add_noise
    from nospeech_augmentation import NoSpeechAugmentConfig, apply_nospeech_augment
    from processor_collate import build_processor_batch_inputs

_ASR_TEXT_TAG = "<asr_text>"
_WER_WS_RE = re.compile(r"\s+")


def _normalize_english(text: str) -> str:
    value = (text or "").lower()
    value = value.translate(str.maketrans("", "", string.punctuation))
    return _WER_WS_RE.sub(" ", value).strip()


def _edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_char in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_char in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[hyp_index] + 1,
                    current[hyp_index - 1] + 1,
                    previous[hyp_index - 1] + (ref_char != hyp_char),
                )
            )
        previous = current
    return previous[-1]


def _corpus_character_error_rate(
    references: List[str], hypotheses: List[str]
) -> float:
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    total_edits = 0
    total_reference_chars = 0
    for reference, hypothesis in zip(references, hypotheses):
        ref_chars = reference.replace(" ", "")
        hyp_chars = hypothesis.replace(" ", "")
        total_edits += _edit_distance(ref_chars, hyp_chars)
        total_reference_chars += len(ref_chars)
    if total_reference_chars == 0:
        return 0.0 if total_edits == 0 else 1.0
    return total_edits / total_reference_chars


def patch_outer_forward(model):
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return
    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError(
            "Cannot patch forward: model has no model.thinker.forward. "
            "Your qwen3_asr model may be incompatible."
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True

_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    best_step = None
    best_path = None
    for name in os.listdir(output_dir):
        m = _CKPT_RE.match(name)
        if not m:
            continue
        step = int(m.group(1))
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and (best_step is None or step > best_step):
            best_step = step
            best_path = path
    return best_path


def load_audio(path: str, sr: int = 16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def build_prefix_messages(prompt: str, audio_array):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


_LANG_PREFIX_RE = re.compile(r"language [^<]*<asr_text>")
_TARGET_BRACKETS_RE = re.compile(r"[()\[\]{}]")


def _swap_language_to_none(text: str) -> str:
    """Derive the 'language None<asr_text>...' variant from an explicit label."""
    return _LANG_PREFIX_RE.sub("language None<asr_text>", text or "", count=1)


def strip_target_brackets(text: str) -> str:
    """Remove ASCII bracket glyphs while preserving their contents and ASR tag."""
    return _TARGET_BRACKETS_RE.sub("", text or "")


def make_preprocess_fn_prefix_only(processor, curriculum: bool = False):
    def _preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ex.get("prompt", "")
        dummy_audio = None
        prefix_msgs = build_prefix_messages(prompt, dummy_audio)
        prefix_text = processor.apply_chat_template(
            [prefix_msgs], add_generation_prompt=True, tokenize=False
        )[0]
        out = {
            "prompt": prompt,
            "audio": ex["audio"],
            "target": ex["text"],
            "prefix_text": prefix_text,
            "aug": ex.get("aug", 1),
            "noise_aug": ex.get("noise_aug", 1),
        }
        if curriculum:
            out["target_none"] = _swap_language_to_none(ex["text"])
        return out

    return _preprocess


def _parse_float_list(value: str, default: Tuple[float, ...]) -> Tuple[float, ...]:
    raw = (value or "").strip()
    if not raw:
        return default
    return tuple(float(x.strip()) for x in raw.split(",") if x.strip())


@dataclass
class AudioAugmentConfig:
    """Optional online augmentation for training only."""

    enabled: bool = False
    augment_prob: float = 1.0
    speed_prob: float = 0.5
    speed_factors: Tuple[float, ...] = (0.9, 1.0, 1.1)
    speed_factor_min: Optional[float] = None
    speed_factor_max: Optional[float] = None
    noise_prob: float = 0.5
    noise_snr_min: float = 5.0
    noise_snr_max: float = 20.0
    noise_dir: str = ""
    specaug_prob: float = 0.5
    specaug_time_mask_param: int = 50
    specaug_freq_mask_param: int = 27
    specaug_num_time_masks: int = 2
    specaug_num_freq_masks: int = 2
    nospeech: NoSpeechAugmentConfig = field(default_factory=NoSpeechAugmentConfig)

    _specaug_masks: Any = field(default=None, repr=False, compare=False)
    noise_library: Optional[NoiseLibrary] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if not self.enabled:
            return
        try:
            import torchaudio
        except ImportError as e:
            raise ImportError(
                "Audio augmentation requires torchaudio. Install a build matching your PyTorch, e.g.: "
                "pip install --no-deps 'torchaudio==2.8.0+cu128' "
                "--index-url https://download.pytorch.org/whl/cu128"
            ) from e
        self.noise_library = NoiseLibrary.from_dir(self.noise_dir)
        self._specaug_masks = _build_specaug_transforms(self)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "AudioAugmentConfig":
        return cls(
            enabled=bool(getattr(args, "augment", 0)),
            augment_prob=float(getattr(args, "augment_prob", 1.0)),
            speed_prob=float(getattr(args, "speed_prob", 0.5)),
            speed_factors=_parse_float_list(
                getattr(args, "speed_factors", ""), (0.9, 1.0, 1.1)
            ),
            speed_factor_min=getattr(args, "speed_factor_min", None),
            speed_factor_max=getattr(args, "speed_factor_max", None),
            noise_prob=float(getattr(args, "noise_prob", 0.5)),
            noise_snr_min=float(getattr(args, "noise_snr_min", 5.0)),
            noise_snr_max=float(getattr(args, "noise_snr_max", 20.0)),
            noise_dir=str(getattr(args, "noise_dir", "") or ""),
            specaug_prob=float(getattr(args, "specaug_prob", 0.5)),
            specaug_time_mask_param=int(getattr(args, "specaug_time_mask_param", 50)),
            specaug_freq_mask_param=int(getattr(args, "specaug_freq_mask_param", 27)),
            specaug_num_time_masks=int(getattr(args, "specaug_num_time_masks", 2)),
            specaug_num_freq_masks=int(getattr(args, "specaug_num_freq_masks", 2)),
            nospeech=NoSpeechAugmentConfig.from_args(args),
        )


def _build_specaug_transforms(cfg: AudioAugmentConfig):
    import torchaudio

    time_masks = [
        torchaudio.transforms.TimeMasking(cfg.specaug_time_mask_param)
        for _ in range(cfg.specaug_num_time_masks)
    ]
    freq_masks = [
        torchaudio.transforms.FrequencyMasking(cfg.specaug_freq_mask_param)
        for _ in range(cfg.specaug_num_freq_masks)
    ]
    return time_masks, freq_masks


def apply_speed_perturbation(wav: np.ndarray, sr: int, factor: float) -> np.ndarray:
    """Change playback speed via resample (factor>1 faster/shorter, factor<1 slower/longer).

    Uses librosa's soxr backend: torchaudio.functional.resample's default sinc filter is
    ~1.2s for a 6s clip when sr/new_sr are coprime (huge kernel), which starves the GPU
    during training. soxr_hq is ~1000x faster and still anti-aliased.
    """
    if factor == 1.0:
        return wav
    new_sr = max(1, int(round(sr / factor)))
    out = librosa.resample(wav, orig_sr=sr, target_sr=new_sr, res_type="soxr_hq")
    return out.astype(np.float32, copy=False)


def augment_waveform(
    wav: np.ndarray,
    sr: int,
    cfg: AudioAugmentConfig,
    rng: random.Random,
    *,
    noise_enabled: bool = True,
) -> np.ndarray:
    if not cfg.enabled or rng.random() > cfg.augment_prob:
        return wav
    out = wav.astype(np.float32, copy=False)
    if rng.random() < cfg.speed_prob:
        if cfg.speed_factor_min is not None and cfg.speed_factor_max is not None:
            factor = rng.uniform(cfg.speed_factor_min, cfg.speed_factor_max)
        else:
            factor = rng.choice(cfg.speed_factors)
        out = apply_speed_perturbation(out, sr, factor)
    if noise_enabled:
        out = maybe_add_noise(
            out,
            sr,
            rng,
            prob=cfg.noise_prob,
            snr_min=cfg.noise_snr_min,
            snr_max=cfg.noise_snr_max,
            library=cfg.noise_library,
        )
    return out


def apply_specaugment(features: torch.Tensor, cfg: AudioAugmentConfig, rng: random.Random) -> torch.Tensor:
    """Mask mel features; expects shape (batch, n_mels, time) or (n_mels, time)."""
    if not cfg.enabled or cfg._specaug_masks is None or rng.random() > cfg.specaug_prob:
        return features
    time_masks, freq_masks = cfg._specaug_masks
    out = features.clone()
    if out.dim() == 2:
        for fm in freq_masks:
            out = fm(out)
        for tm in time_masks:
            out = tm(out)
        return out
    for i in range(out.size(0)):
        sample = out[i]
        for fm in freq_masks:
            sample = fm(sample)
        for tm in time_masks:
            sample = tm(sample)
        out[i] = sample
    return out


@dataclass
class DataCollatorForQwen3ASRFinetuning:
    processor: Any
    sampling_rate: int = 16000
    augment: Optional[AudioAugmentConfig] = None
    curriculum: bool = False
    curriculum_switch_epoch: float = 1.0
    current_epoch: float = 0.0
    strip_target_brackets: bool = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_paths = [f["audio"] for f in features]
        prefix_texts = [f["prefix_text"] for f in features]
        use_none = self.curriculum and self.current_epoch >= self.curriculum_switch_epoch
        if use_none:
            targets = [f.get("target_none", f["target"]) for f in features]
        else:
            targets = [f["target"] for f in features]
        noise_flags = [int(f.get("noise_aug", 1)) for f in features]

        cfg = self.augment
        use_aug = cfg is not None and cfg.enabled
        rng = random.Random() if use_aug else None

        audios = [load_audio(p, sr=self.sampling_rate) for p in audio_paths]
        aug_flags = [int(f.get("aug", 1)) for f in features]

        if use_aug and cfg.nospeech.enabled:
            audios, targets = apply_nospeech_augment(
                audios,
                targets,
                aug_flags,
                rng,
                cfg.nospeech,
                self.sampling_rate,
            )
        if self.strip_target_brackets:
            targets = [strip_target_brackets(target) for target in targets]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]

        if use_aug:
            audios = [
                augment_waveform(
                    wav,
                    self.sampling_rate,
                    cfg,
                    rng,
                    noise_enabled=bool(noise_aug),
                )
                if aug == 1
                else wav
                for wav, aug, noise_aug in zip(audios, aug_flags, noise_flags)
            ]

        full_inputs, prefix_inputs = build_processor_batch_inputs(
            self.processor, full_texts, prefix_texts, audios
        )

        if use_aug and "input_features" in full_inputs:
            feats = full_inputs["input_features"]
            for i, feat in enumerate(features):
                if int(feat.get("aug", 1)) == 1:
                    feats[i] = apply_specaugment(feats[i], cfg, rng)

        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        # The processor may left-pad (real tokens right-aligned), so the prefix
        # does NOT start at position 0. Mask the prefix at the first real token,
        # and mask all padding via attention_mask (the pad token may be
        # <|audio_pad|>, not pad_token_id, so an id-based mask would miss it).
        attn = full_inputs["attention_mask"]
        starts = (attn == 1).long().argmax(dim=1).tolist()
        for i, (pl, st) in enumerate(zip(prefix_lens, starts)):
            labels[i, st : st + pl] = -100
        labels[attn == 0] = -100

        full_inputs["labels"] = labels
        return full_inputs


class CastFloatInputsTrainer(Trainer):
    """Trainer with a separate eval collator and optional module learning rates."""

    def __init__(
        self,
        *args,
        eval_data_collator=None,
        lr_encoder=None,
        lr_aligner=None,
        lr_llm=None,
        **kwargs,
    ):
        self.eval_data_collator = eval_data_collator
        self.lr_encoder = lr_encoder
        self.lr_aligner = lr_aligner
        self.lr_llm = lr_llm
        super().__init__(*args, **kwargs)

    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=model_dtype)
        return inputs

    def get_eval_dataloader(self, eval_dataset=None):
        if self.eval_data_collator is None:
            return super().get_eval_dataloader(eval_dataset)
        original = self.data_collator
        self.data_collator = self.eval_data_collator
        try:
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.data_collator = original

    @staticmethod
    def _lr_group(name: str) -> str:
        if any(
            marker in name
            for marker in (
                "audio_tower.conv_out",
                "audio_tower.proj1",
                "audio_tower.proj2",
            )
        ):
            return "aligner"
        if "audio_tower" in name:
            return "encoder"
        if "thinker.model." in name or "thinker.lm_head" in name:
            return "llm"
        return "other"

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        if self.lr_encoder is None and self.lr_aligner is None and self.lr_llm is None:
            return super().create_optimizer()

        groups: Dict[str, list] = {"encoder": [], "aligner": [], "llm": [], "other": []}
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                groups[self._lr_group(name)].append(parameter)

        base_lr = self.args.learning_rate
        lrs = {
            "encoder": self.lr_encoder if self.lr_encoder is not None else base_lr,
            "aligner": self.lr_aligner if self.lr_aligner is not None else base_lr,
            "llm": self.lr_llm if self.lr_llm is not None else base_lr,
            "other": base_lr,
        }
        optimizer_groups = [
            {"params": parameters, "lr": lrs[group], "weight_decay": self.args.weight_decay}
            for group, parameters in groups.items()
            if parameters
        ]
        if self.args.process_index == 0:
            for group, parameters in groups.items():
                if parameters:
                    print(
                        "[optimizer] %-7s: %d params  lr=%.1e"
                        % (group, sum(p.numel() for p in parameters), lrs[group])
                    )
        self.optimizer = torch.optim.AdamW(
            optimizer_groups,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )
        return self.optimizer


def copy_required_hf_files_for_qwen_asr(src_dir: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    required = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "chat_template.json",
        "merges.txt",
        "vocab.json",
    ]
    for fn in required:
        src = os.path.join(src_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, fn))


class MakeEveryCheckpointInferableCallback(TrainerCallback):
    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path

    def on_save(self, args: TrainingArguments, state, control, **kwargs):
        if args.process_index != 0:
            return control

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            ckpt_dir = kwargs.get("checkpoint", ckpt_dir)

        copy_required_hf_files_for_qwen_asr(self.base_model_path, ckpt_dir)
        return control


class KeepBestCheckpointsCallback(TrainerCallback):
    """Keep only the checkpoints with the lowest validation losses."""

    ranking_filename = "best_checkpoints.json"

    def __init__(self, limit: int):
        if limit < 1:
            raise ValueError("Best-checkpoint limit must be positive.")
        self.limit = limit
        self.eval_losses: Dict[int, float] = {}

    def _ranking_path(self, output_dir: str) -> str:
        return os.path.join(output_dir, self.ranking_filename)

    def _load_ranking(self, output_dir: str) -> None:
        path = self._ranking_path(output_dir)
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data.get("checkpoints", []):
                self.eval_losses[int(item["step"])] = float(item["eval_loss"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            print(f"[best-checkpoints] ignoring unreadable ranking file {path}: {e}")

    def _existing_ranked(self, output_dir: str) -> List[Tuple[float, int, str]]:
        ranked = []
        for step, loss in self.eval_losses.items():
            ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
            if os.path.isdir(ckpt_dir):
                ranked.append((loss, step, ckpt_dir))
        return sorted(ranked)

    def _write_ranking(self, output_dir: str, ranked: List[Tuple[float, int, str]]) -> None:
        path = self._ranking_path(output_dir)
        tmp_path = path + ".tmp"
        payload = {
            "metric": "eval_loss",
            "greater_is_better": False,
            "limit": self.limit,
            "checkpoints": [
                {
                    "rank": rank,
                    "step": step,
                    "eval_loss": loss,
                    "path": os.path.abspath(ckpt_dir),
                }
                for rank, (loss, step, ckpt_dir) in enumerate(ranked, start=1)
            ],
        }
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp_path, path)

    def on_train_begin(self, args, state, control, **kwargs):
        if args.process_index == 0:
            self._load_ranking(args.output_dir)
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if args.process_index != 0 or not metrics:
            return control
        loss = metrics.get("eval_loss")
        if loss is None:
            return control
        loss = float(loss)
        if math.isfinite(loss):
            self.eval_losses[int(state.global_step)] = loss
        return control

    def on_save(self, args, state, control, **kwargs):
        if args.process_index != 0:
            return control

        ranked = self._existing_ranked(args.output_dir)
        keep_steps = {step for _, step, _ in ranked[: self.limit]}
        for _, step, ckpt_dir in ranked[self.limit :]:
            shutil.rmtree(ckpt_dir)
            print(
                f"[best-checkpoints] removed checkpoint-{step}; "
                f"outside best {self.limit} by eval_loss"
            )

        kept = [item for item in ranked if item[1] in keep_steps]
        self._write_ranking(args.output_dir, kept)
        summary = ", ".join(
            f"checkpoint-{step}={loss:.6f}" for loss, step, _ in kept
        )
        print(f"[best-checkpoints] retained: {summary}")
        return control


def _wer_extract_ref(label: str) -> str:
    """Strip the Qwen3 'language X<asr_text>' prefix; return the reference transcript."""
    s = (label or "").strip()
    if _ASR_TEXT_TAG in s:
        return s.split(_ASR_TEXT_TAG, 1)[1].strip()
    return s


def _wer_lang_from_label(label: str) -> str:
    """Map the language atom in a data label to the scoring language."""
    head = label.split(_ASR_TEXT_TAG, 1)[0].lower() if _ASR_TEXT_TAG in (label or "") else ""
    if "english" in head:
        return "en"
    if "chinese" in head:
        return "zh"
    return "ug"


class KeepBestWerCheckpointsCallback(KeepBestCheckpointsCallback):
    """Keep checkpoints ranked by generation WER on a fixed validation subset."""

    ranking_filename = "best_wer_checkpoints.json"

    def __init__(
        self,
        limit,
        asr_wrapper,
        eval_dataset,
        sr=16000,
        wer_eval_samples=400,
        wer_batch_size=4,
        eval_speed_aug=False,
        speed_min=0.8,
        speed_max=1.6,
        seed=1234,
        metric="wer",
        output_dir="",
    ):
        super().__init__(limit)
        self.asr_wrapper = asr_wrapper
        self.eval_dataset = eval_dataset
        self.sr = sr
        self.wer_eval_samples = wer_eval_samples
        self.wer_batch_size = max(1, wer_batch_size)
        self.eval_speed_aug = bool(eval_speed_aug)
        self.speed_min = speed_min
        self.speed_max = speed_max
        if metric not in {"wer", "cer"}:
            raise ValueError("Generation checkpoint metric must be wer or cer")
        self.seed = seed
        self.metric = metric
        self.ranking_filename = f"best_{metric}_checkpoints.json"
        self.predictions_dir = os.path.join(output_dir, "generation_eval") if output_dir else ""
        self._subset = None

    def _load_ranking(self, output_dir: str) -> None:
        path = self._ranking_path(output_dir)
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            for item in data.get("checkpoints", []):
                value = item.get(self.metric, item.get("eval_loss"))
                if value is not None:
                    self.eval_losses[int(item["step"])] = float(value)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            print(f"[best-wer] ignoring unreadable ranking file {path}: {exc}")

    def _write_ranking(self, output_dir: str, ranked: List[Tuple[float, int, str]]) -> None:
        path = self._ranking_path(output_dir)
        tmp_path = path + ".tmp"
        payload = {
            "metric": self.metric,
            "greater_is_better": False,
            "limit": self.limit,
            "checkpoints": [
                {
                    "rank": rank,
                    "step": step,
                    self.metric: value,
                    "path": os.path.abspath(ckpt_dir),
                }
                for rank, (value, step, ckpt_dir) in enumerate(ranked, start=1)
            ],
        }
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp_path, path)

    def _build_subset(self):
        if self._subset is not None:
            return self._subset
        dataset = self.eval_dataset
        indices = list(range(len(dataset)))
        if self.wer_eval_samples and 0 < self.wer_eval_samples < len(indices):
            random.Random(self.seed).shuffle(indices)
            indices = indices[: self.wer_eval_samples]
        rows = dataset.select(indices)
        self._subset = [
            (row["audio"], _wer_extract_ref(row["target"]), _wer_lang_from_label(row["target"]))
            for row in rows
        ]
        return self._subset

    def _load_and_augment(self, path):
        import zlib

        wav = load_audio(path, sr=self.sr)
        if self.eval_speed_aug:
            seed = zlib.crc32(path.encode("utf-8")) & 0xFFFFFFFF
            factor = random.Random(seed).uniform(self.speed_min, self.speed_max)
            wav = apply_speed_perturbation(wav, self.sr, factor)
        return wav

    def _compute_error_rate(self) -> tuple[float, List[Dict[str, str]]]:
        items = self._build_subset()
        hypotheses = [""] * len(items)
        model = self.asr_wrapper.model
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for start in range(0, len(items), self.wer_batch_size):
                    batch = items[start : start + self.wer_batch_size]
                    inputs = [(self._load_and_augment(audio), self.sr) for audio, _, _ in batch]
                    outputs = self.asr_wrapper.transcribe(
                        audio=inputs, language=None, return_time_stamps=False
                    )
                    for offset, output in enumerate(outputs):
                        _, text = parse_asr_output(output.text, user_language=None)
                        hypotheses[start + offset] = text or ""
        finally:
            model.train(was_training)

        if self.metric == "cer":
            from evaluation.english_medical.text_normalization import normalize_english

            records = [
                {
                    "audio": audio,
                    "reference": reference,
                    "reference_normalized": normalize_english(reference),
                    "hypothesis": hypothesis,
                    "hypothesis_normalized": normalize_english(hypothesis),
                }
                for (audio, reference, _), hypothesis in zip(items, hypotheses)
            ]
            references = [record["reference_normalized"] for record in records]
            normalized_hypotheses = [record["hypothesis_normalized"] for record in records]
            return _corpus_character_error_rate(references, normalized_hypotheses), records

        from masr_eval_pkg import compute_wer

        records = [
            {
                "audio": audio,
                "reference": reference,
                "reference_normalized": _normalize_english(reference),
                "hypothesis": hypothesis,
                "hypothesis_normalized": _normalize_english(hypothesis),
            }
            for (audio, reference, _), hypothesis in zip(items, hypotheses)
        ]
        references = [record["reference_normalized"] for record in records]
        normalized_hypotheses = [record["hypothesis_normalized"] for record in records]
        return float(compute_wer(references, normalized_hypotheses)["wer"]), records

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if args.process_index != 0 or self.eval_dataset is None:
            return control
        value, records = self._compute_error_rate()
        if self.predictions_dir and records:
            os.makedirs(self.predictions_dir, exist_ok=True)
            predictions_path = os.path.join(
                self.predictions_dir, f"step-{int(state.global_step)}_predictions.jsonl"
            )
            with open(predictions_path, "w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if metrics is not None and math.isfinite(value):
            metrics[self.metric] = value
        if math.isfinite(value):
            self.eval_losses[int(state.global_step)] = value
        print(
            "[best-%s] step=%d %s=%.4f"
            % (self.metric, state.global_step, self.metric, value)
        )
        return control


class StopOnMetricPlateauCallback(TrainerCallback):
    """Stop after repeated evaluations without a meaningful metric improvement."""

    def __init__(self, metric: str, patience: int, threshold: float, ranking_filename: str):
        if patience < 1:
            raise ValueError("Early-stopping patience must be positive.")
        self.metric = metric
        self.patience = patience
        self.threshold = threshold
        self.ranking_filename = ranking_filename
        self.best_value: Optional[float] = None
        self.bad_evals = 0

    def on_train_begin(self, args, state, control, **kwargs):
        path = os.path.join(args.output_dir, self.ranking_filename)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            values = [
                float(item.get(self.metric, item.get("eval_loss")))
                for item in data.get("checkpoints", [])
                if item.get(self.metric, item.get("eval_loss")) is not None
            ]
            if values:
                self.best_value = min(values)
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or metrics.get(self.metric) is None:
            return control
        value = float(metrics[self.metric])
        improved = self.best_value is None or value < self.best_value - self.threshold
        if improved:
            self.best_value = value
            self.bad_evals = 0
        else:
            self.bad_evals += 1
        print(
            "[early-stop] step=%d %s=%.6f best=%.6f bad_evals=%d/%d"
            % (state.global_step, self.metric, value, self.best_value, self.bad_evals, self.patience)
        )
        if self.bad_evals >= self.patience:
            control.should_training_stop = True
            print("[early-stop] convergence criterion reached; stopping training.")
        return control


class KeepFrozenPartsCallback(TrainerCallback):
    """Re-apply per-part freezing after Trainer flips module train/eval modes."""

    def __init__(self, frozen_parts):
        self.frozen_parts = tuple(frozen_parts)

    def _refreeze(self, model) -> None:
        if model is not None:
            set_part_freeze(model, self.frozen_parts)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self._refreeze(model)
        return control

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        self._refreeze(model)
        return control

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        self._refreeze(model)
        return control


class CurriculumEpochCallback(TrainerCallback):
    """Switch the collator target to the 'language None' variant from the switch epoch on."""

    def __init__(self, collator, switch_epoch: float = 1.0):
        self.collator = collator
        self.switch_epoch = float(switch_epoch)

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.collator.current_epoch = float(state.epoch or 0.0)
        if args.process_index == 0:
            phase = "None" if self.collator.current_epoch >= self.switch_epoch else "explicit"
            print("[curriculum] epoch=%.4f target_phase=%s" % (
                self.collator.current_epoch, phase))
        return control


def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR Finetuning (full-parameter)")

    # Paths
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--train_file", type=str, default="train.jsonl")
    p.add_argument("--eval_file", type=str, default="")
    p.add_argument("--output_dir", type=str, default="./qwen3-asr-finetuning-out")

    # Audio
    p.add_argument("--sr", type=int, default=16000)

    # Train hyper-params
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_acc", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--lr_scheduler_type", type=str, default="linear")
    p.add_argument("--lr_encoder", type=float, default=None)
    p.add_argument("--lr_aligner", type=float, default=None)
    p.add_argument("--lr_llm", type=float, default=None)
    p.add_argument("--warmup_ratio", type=float, default=0.02)
    p.add_argument(
        "--freeze_modules",
        type=str,
        default="",
        help="Comma-separated parts to freeze: encoder,aligner,llm. "
             "encoder=audio_tower minus the aligner projections; "
             "aligner=audio_tower conv_out/proj1/proj2; llm=thinker.model+lm_head. "
             "Empty (default) trains all parts.",
    )
    p.add_argument(
        "--freeze_audio_tower",
        type=int,
        default=0,
        choices=(0, 1),
        help="Deprecated alias for --freeze_modules encoder,aligner.",
    )
    p.add_argument(
        "--curriculum",
        type=int,
        default=0,
        choices=(0, 1),
        help="Epoch-aware language curriculum: epochs < --curriculum_switch_epoch use the "
             "explicit language label in the data; from that epoch on the collator derives "
             "'language None<asr_text>' so the model learns to judge language itself.",
    )
    p.add_argument(
        "--curriculum_switch_epoch",
        type=float,
        default=1.0,
        help="Epoch (0-indexed) at which the collator switches targets to the None variant.",
    )
    p.add_argument(
        "--strip_target_brackets",
        type=int,
        default=0,
        choices=(0, 1),
        help="Remove ()[]{} from SFT targets while preserving their contents.",
    )

    # DataLoader
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--persistent_workers", type=int, default=1)
    p.add_argument("--prefetch_factor", type=int, default=2)

    # Save
    p.add_argument("--save_strategy", type=str, default="steps")
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=5)
    p.add_argument(
        "--save_best_total_limit",
        type=int,
        default=0,
        help="When >0 with --eval_file, retain only this many checkpoints with the lowest eval_loss.",
    )
    p.add_argument(
        "--save_best_metric",
        type=str,
        default="eval_loss",
        choices=("eval_loss", "wer", "cer"),
        help="Metric for --save_best_total_limit: eval_loss, WER, or normalized CER.",
    )
    p.add_argument("--wer_eval_samples", type=int, default=400,
                   help="Capped dev subset size for the WER selection metric (balanced by random subset).")
    p.add_argument("--wer_batch_size", type=int, default=4,
                   help="Inference batch size for the WER selection generation.")
    p.add_argument("--wer_max_new_tokens", type=int, default=512,
                   help="Max new tokens for the in-training WER/CER selection generation "
                        "(passed to the ASR wrapper).")
    p.add_argument("--early_stopping_metric", type=str, default="eval_loss",
                   choices=("eval_loss", "wer", "cer"))
    p.add_argument("--early_stopping_patience", type=int, default=0)
    p.add_argument("--early_stopping_threshold", type=float, default=0.0)
    p.add_argument("--eval_speed_aug", type=int, default=0, choices=(0, 1),
                   help="Apply a deterministic per-utterance speed perturbation to dev audio "
                        "used for the WER selection metric (robust dev).")
    p.add_argument("--eval_speed_min", type=float, default=0.8)
    p.add_argument("--eval_speed_max", type=float, default=1.6)

    # Resume
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume", type=int, default=0)

    # Online audio augmentation (train only; default off)
    p.add_argument("--augment", type=int, default=0, choices=(0, 1))
    p.add_argument("--augment_prob", type=float, default=1.0)
    p.add_argument("--speed_prob", type=float, default=0.5)
    p.add_argument("--speed_factors", type=str, default="0.9,1.0,1.1")
    p.add_argument("--speed_factor_min", type=float, default=None,
                   help="If set with --speed_factor_max, sample the speed factor "
                        "uniformly in [min,max] instead of the discrete --speed_factors list.")
    p.add_argument("--speed_factor_max", type=float, default=None)
    p.add_argument("--noise_prob", type=float, default=0.5)
    p.add_argument("--noise_snr_min", type=float, default=5.0)
    p.add_argument("--noise_snr_max", type=float, default=20.0)
    p.add_argument(
        "--noise_dir",
        type=str,
        default="",
        help="Directory of real noise wavs (e.g. DNS-Challenge noise_fullband). "
        "When set, AddNoise samples real noise files instead of synthetic white noise.",
    )
    p.add_argument("--specaug_prob", type=float, default=0.5)
    p.add_argument("--specaug_time_mask_param", type=int, default=50)
    p.add_argument("--specaug_freq_mask_param", type=int, default=27)
    p.add_argument("--specaug_num_time_masks", type=int, default=2)
    p.add_argument("--specaug_num_freq_masks", type=int, default=2)

    # No-speech segment augmentation (train only; default off)
    p.add_argument("--nospeech_augment", type=int, default=0, choices=(0, 1))
    p.add_argument("--nospeech_prob", type=float, default=0.3)
    p.add_argument("--nospeech_pad_min_sec", type=float, default=0.5)
    p.add_argument("--nospeech_pad_max_sec", type=float, default=3.0)

    return p.parse_args()


def main():
    args_cli = parse_args()

    if not args_cli.train_file:
        raise ValueError("TRAIN_FILE is required (json/jsonl). Needs fields: audio, text, optional prompt")

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args_cli.model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
        max_new_tokens=args_cli.wer_max_new_tokens,
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor
    patch_outer_forward(model)

    model.generation_config = GenerationConfig.from_model_config(model.config)
    frozen_parts = list(parse_parts(args_cli.freeze_modules))
    if args_cli.freeze_audio_tower == 1:
        frozen_parts += [p for p in ("encoder", "aligner") if p not in frozen_parts]
    if frozen_parts:
        set_part_freeze(model, frozen_parts)
        part_counts = count_part_parameters(model)
        for part in PARTS:
            total, trainable = part_counts[part]
            print(
                "[freeze] %-7s total_params=%d trainable_params=%d"
                % (part, total, trainable)
            )

    raw_ds = load_dataset(
        "json",
        data_files={
            "train": args_cli.train_file,
            **({"validation": args_cli.eval_file} if args_cli.eval_file else {}),
        },
    )
    ds = raw_ds.map(
        make_preprocess_fn_prefix_only(processor, curriculum=bool(args_cli.curriculum)),
        num_proc=1,
    )

    keep = {"prompt", "audio", "target", "prefix_text", "aug", "noise_aug"}
    if args_cli.curriculum:
        keep.add("target_none")
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    augment_cfg = AudioAugmentConfig.from_args(args_cli)
    train_collator = DataCollatorForQwen3ASRFinetuning(
        processor=processor,
        sampling_rate=args_cli.sr,
        augment=augment_cfg if augment_cfg.enabled else None,
        curriculum=bool(args_cli.curriculum),
        curriculum_switch_epoch=args_cli.curriculum_switch_epoch,
        strip_target_brackets=bool(args_cli.strip_target_brackets),
    )
    eval_collator = DataCollatorForQwen3ASRFinetuning(
        processor=processor,
        sampling_rate=args_cli.sr,
        augment=None,
        strip_target_brackets=bool(args_cli.strip_target_brackets),
    )

    keep_best_limit = args_cli.save_best_total_limit if args_cli.eval_file else 0
    if args_cli.save_best_total_limit > 0 and not args_cli.eval_file:
        raise ValueError("--save_best_total_limit requires --eval_file")
    if args_cli.save_best_metric in {"wer", "cer"} and not args_cli.eval_file:
        raise ValueError("--save_best_metric wer/cer requires --eval_file")

    training_args = TrainingArguments(
        output_dir=args_cli.output_dir,
        per_device_train_batch_size=args_cli.batch_size,
        gradient_accumulation_steps=args_cli.grad_acc,
        learning_rate=args_cli.lr,
        num_train_epochs=args_cli.epochs,
        logging_steps=args_cli.log_steps,
        lr_scheduler_type=args_cli.lr_scheduler_type,
        warmup_ratio=args_cli.warmup_ratio,
        dataloader_num_workers=args_cli.num_workers,
        dataloader_pin_memory=(args_cli.pin_memory == 1),
        dataloader_persistent_workers=(args_cli.persistent_workers == 1),
        dataloader_prefetch_factor=args_cli.prefetch_factor if args_cli.num_workers > 0 else None,
        save_strategy=args_cli.save_strategy,
        save_steps=args_cli.save_steps,
        save_total_limit=None if keep_best_limit > 0 else args_cli.save_total_limit,
        save_safetensors=True,
        eval_strategy="steps" if args_cli.eval_file else "no",
        eval_steps=args_cli.save_steps if args_cli.eval_file else None,
        do_eval=bool(args_cli.eval_file),
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="none",
    )

    callbacks = [MakeEveryCheckpointInferableCallback(base_model_path=args_cli.model_path)]
    if keep_best_limit > 0:
        if args_cli.save_best_metric in {"wer", "cer"}:
            callbacks.append(KeepBestWerCheckpointsCallback(
                limit=keep_best_limit,
                asr_wrapper=asr_wrapper,
                eval_dataset=ds.get("validation"),
                sr=args_cli.sr,
                wer_eval_samples=args_cli.wer_eval_samples,
                wer_batch_size=args_cli.wer_batch_size,
                eval_speed_aug=bool(args_cli.eval_speed_aug),
                speed_min=args_cli.eval_speed_min,
                speed_max=args_cli.eval_speed_max,
                metric=args_cli.save_best_metric,
                output_dir=args_cli.output_dir,
            ))
        else:
            callbacks.append(KeepBestCheckpointsCallback(limit=keep_best_limit))
    if args_cli.early_stopping_patience > 0:
        if not args_cli.eval_file:
            raise ValueError("--early_stopping_patience requires --eval_file")
        ranking_filename = (
            "best_checkpoints.json"
            if args_cli.early_stopping_metric == "eval_loss"
            else f"best_{args_cli.early_stopping_metric}_checkpoints.json"
        )
        callbacks.append(
            StopOnMetricPlateauCallback(
                metric=args_cli.early_stopping_metric,
                patience=args_cli.early_stopping_patience,
                threshold=args_cli.early_stopping_threshold,
                ranking_filename=ranking_filename,
            )
        )
    if frozen_parts:
        callbacks.append(KeepFrozenPartsCallback(frozen_parts))
    if args_cli.curriculum:
        callbacks.append(CurriculumEpochCallback(train_collator, args_cli.curriculum_switch_epoch))

    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", None),
        data_collator=train_collator,
        eval_data_collator=eval_collator if args_cli.eval_file else None,
        tokenizer=processor.tokenizer,
        callbacks=callbacks,
        lr_encoder=args_cli.lr_encoder,
        lr_aligner=args_cli.lr_aligner,
        lr_llm=args_cli.lr_llm,
    )

    if trainer.args.process_index == 0 and augment_cfg.enabled:
        print(
            "[augment] enabled: speed_prob=%s noise_prob=%s specaug_prob=%s"
            % (augment_cfg.speed_prob, augment_cfg.noise_prob, augment_cfg.specaug_prob)
        )
        if augment_cfg.noise_library is not None and augment_cfg.noise_library.enabled:
            print(
                "[augment] noise_dir=%s files=%d"
                % (augment_cfg.noise_dir, augment_cfg.noise_library.num_files)
            )
        if augment_cfg.nospeech.enabled:
            ns = augment_cfg.nospeech
            print(
                "[augment] nospeech_prob=%s pad=%.1f-%.1fs"
                % (ns.prob, ns.pad_min_sec, ns.pad_max_sec)
            )

    resume_from = (args_cli.resume_from or "").strip()
    if not resume_from and args_cli.resume == 1:
        resume_from = find_latest_checkpoint(training_args.output_dir) or ""

    if resume_from:
        if trainer.args.process_index == 0:
            print(f"[resume] resume_from_checkpoint = {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
