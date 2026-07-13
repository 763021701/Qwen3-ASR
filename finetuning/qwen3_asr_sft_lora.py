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
# LoRA fine-tuning for Qwen3-ASR.
# For full-parameter fine-tuning, see finetuning/qwen3_asr_sft.py.
import argparse
import os
import random
import re
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
from datasets import load_dataset
from qwen_asr import Qwen3ASRModel
from transformers import (GenerationConfig, Trainer, TrainerCallback,
                          TrainingArguments)

LORA_TARGETS = {
    "encoder": r"^audio_tower\.layers\.\d+\..*\.(q_proj|k_proj|v_proj|out_proj|fc1|fc2)$",
    "aligner": r"^audio_tower\.(conv_out|proj1|proj2)$",
    "encoder_aligner": (
        r"^(audio_tower\.(conv_out|proj1|proj2)$"
        r"|audio_tower\.layers\.\d+\..*\.(q_proj|k_proj|v_proj|out_proj|fc1|fc2)$)"
    ),
    "encoder_b4_aligner": (
        r"^(audio_tower\.(conv_out|proj1|proj2)$"
        r"|audio_tower\.layers\.(20|21|22|23)\..*\.(q_proj|k_proj|v_proj|out_proj|fc1|fc2)$)"
    ),
    "llm": r"^model\.layers\.\d+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
    "all": (
        r"^(audio_tower\.(conv_out|proj1|proj2)$"
        r"|audio_tower\.layers\.\d+\..*\.(q_proj|k_proj|v_proj|out_proj|fc1|fc2)$"
        r"|model\.layers\.\d+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$)"
    ),
}

def patch_outer_forward(model):
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return

    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError(
            "Cannot patch forward: model has no `.thinker.forward`. "
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

def apply_lora(model, args) -> bool:
    """Wrap model.thinker with PEFT LoRA. Returns True if LoRA was applied."""
    if not getattr(args, "use_lora", 0):
        return False

    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    old_lora = str(getattr(args, "merge_lora_into_base_from", "") or "").strip()
    if old_lora:
        if getattr(args, "resume", 0) or str(getattr(args, "resume_from", "") or "").strip():
            raise ValueError("Do not use --merge_lora_into_base_from with --resume or --resume_from.")
        print(f"[merge_lora] merging previous LoRA from: {old_lora}")
        model.thinker = PeftModel.from_pretrained(
            model.thinker, old_lora, is_trainable=False
        ).merge_and_unload()

    for param in model.parameters():
        param.requires_grad = False

    scope = getattr(args, "lora_scope", "encoder_aligner")
    if scope not in LORA_TARGETS:
        raise ValueError(f"Unknown lora_scope {scope!r}. Choices: {list(LORA_TARGETS)}")

    lora_config = LoraConfig(
        r=int(getattr(args, "lora_r", 8)),
        lora_alpha=int(getattr(args, "lora_alpha", 16)),
        lora_dropout=float(getattr(args, "lora_dropout", 0.05)),
        bias=str(getattr(args, "lora_bias", "none")),
        task_type=TaskType.CAUSAL_LM,
        target_modules=LORA_TARGETS[scope],
    )
    model.thinker = get_peft_model(model.thinker, lora_config)
    model.thinker.print_trainable_parameters()
    return True

def count_parameters(module) -> Tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable

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

def make_preprocess_fn_prefix_only(processor):
    def _preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ex.get("prompt", "")
        dummy_audio = None
        prefix_msgs = build_prefix_messages(prompt, dummy_audio)
        prefix_text = processor.apply_chat_template(
            [prefix_msgs], add_generation_prompt=True, tokenize=False
        )[0]
        return {
            "prompt": prompt,
            "audio": ex["audio"],
            "target": ex["text"],
            "prefix_text": prefix_text,
        }

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
    noise_prob: float = 0.5
    noise_snr_min: float = 5.0
    noise_snr_max: float = 20.0
    specaug_prob: float = 0.5
    specaug_time_mask_param: int = 50
    specaug_freq_mask_param: int = 27
    specaug_num_time_masks: int = 2
    specaug_num_freq_masks: int = 2

    _specaug_masks: Any = field(default=None, repr=False, compare=False)

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
            noise_prob=float(getattr(args, "noise_prob", 0.5)),
            noise_snr_min=float(getattr(args, "noise_snr_min", 5.0)),
            noise_snr_max=float(getattr(args, "noise_snr_max", 20.0)),
            specaug_prob=float(getattr(args, "specaug_prob", 0.5)),
            specaug_time_mask_param=int(getattr(args, "specaug_time_mask_param", 50)),
            specaug_freq_mask_param=int(getattr(args, "specaug_freq_mask_param", 27)),
            specaug_num_time_masks=int(getattr(args, "specaug_num_time_masks", 2)),
            specaug_num_freq_masks=int(getattr(args, "specaug_num_freq_masks", 2)),
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
    """Change playback speed via resample (factor>1 faster/shorter, factor<1 slower/longer)."""
    if factor == 1.0:
        return wav
    import torchaudio

    w = torch.from_numpy(wav).float().unsqueeze(0)
    # factor>1 => faster => fewer samples when interpreted at ``sr``
    new_sr = max(1, int(round(sr / factor)))
    out = torchaudio.functional.resample(w, sr, new_sr)
    return out.squeeze(0).numpy().astype(np.float32)

def apply_add_noise(wav: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix synthetic white noise at the given SNR (dB)."""
    signal = torch.from_numpy(wav).float()
    noise = torch.randn_like(signal)
    signal_power = signal.pow(2).mean().clamp(min=1e-10)
    noise_power = noise.pow(2).mean().clamp(min=1e-10)
    snr_linear = 10 ** (snr_db / 10.0)
    scale = torch.sqrt(signal_power / (noise_power * snr_linear))
    mixed = signal + noise * scale
    peak = mixed.abs().max()
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.numpy().astype(np.float32)

def augment_waveform(
    wav: np.ndarray, sr: int, cfg: AudioAugmentConfig, rng: random.Random
) -> np.ndarray:
    if not cfg.enabled or rng.random() > cfg.augment_prob:
        return wav
    out = wav.astype(np.float32, copy=False)
    if rng.random() < cfg.speed_prob:
        factor = rng.choice(cfg.speed_factors)
        out = apply_speed_perturbation(out, sr, factor)
    if rng.random() < cfg.noise_prob:
        lo, hi = cfg.noise_snr_min, cfg.noise_snr_max
        snr = rng.uniform(min(lo, hi), max(lo, hi))
        out = apply_add_noise(out, snr)
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

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_paths = [f["audio"] for f in features]
        prefix_texts = [f["prefix_text"] for f in features]
        targets = [f["target"] for f in features]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]

        cfg = self.augment
        use_aug = cfg is not None and cfg.enabled
        rng = random.Random() if use_aug else None

        audios = []
        for p in audio_paths:
            wav = load_audio(p, sr=self.sampling_rate)
            if use_aug:
                wav = augment_waveform(wav, self.sampling_rate, cfg, rng)
            audios.append(wav)

        full_inputs = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_inputs = self.processor(
            text=prefix_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        if use_aug and "input_features" in full_inputs:
            full_inputs["input_features"] = apply_specaugment(
                full_inputs["input_features"], cfg, rng
            )

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
    def __init__(self, *args, eval_data_collator=None, lr_encoder=2e-5, lr_aligner=2e-5,
                 lr_llm=2e-5, **kwargs):
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

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        thinker = self.model.thinker
        if hasattr(thinker, "save_pretrained"):
            thinker.save_pretrained(output_dir, safe_serialization=True)
        else:
            super().save_model(output_dir, _internal_call)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        model = model or self.model
        adapter_path = os.path.join(resume_from_checkpoint, "adapter_model.safetensors")
        if os.path.isfile(adapter_path):
            from safetensors.torch import load_file as safe_load_file
            model.thinker.load_state_dict(safe_load_file(adapter_path), strict=False)
            return
        return super()._load_from_checkpoint(resume_from_checkpoint, model=model)

    @staticmethod
    def _lora_group(name: str) -> str:
        """Group a parameter name into encoder / aligner / llm / other for per-module LR."""
        if "lora_" not in name:
            return "other"
        if any(x in name for x in ["audio_tower.conv_out", "audio_tower.proj1", "audio_tower.proj2"]):
            return "aligner"
        if "audio_tower.layers." in name:
            return "encoder"
        if "model.layers." in name and "audio_tower.layers." not in name:
            return "llm"
        return "other"

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        groups: Dict[str, list] = {"encoder": [], "aligner": [], "llm": [], "other": []}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                groups[self._lora_group(n)].append(p)

        lrs = {"encoder": self.lr_encoder, "aligner": self.lr_aligner,
               "llm": self.lr_llm, "other": self.args.learning_rate}
        optim_groups = [
            {"params": params, "lr": lrs[name], "weight_decay": self.args.weight_decay}
            for name, params in groups.items() if params
        ]

        if self.args.process_index == 0:
            for name, params in groups.items():
                if params:
                    print("[optimizer] %-7s: %d params  lr=%.1e"
                          % (name, sum(p.numel() for p in params), lrs[name]))

        self.optimizer = torch.optim.AdamW(
            optim_groups,
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

def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR LoRA Finetuning")

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
    p.add_argument("--warmup_ratio", type=float, default=0.02)

    # DataLoader
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--persistent_workers", type=int, default=1)
    p.add_argument("--prefetch_factor", type=int, default=2)

    # Save
    p.add_argument("--save_strategy", type=str, default="steps")
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=5)

    # Resume
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume", type=int, default=0)

    # Online audio augmentation (train only; default off)
    p.add_argument("--augment", type=int, default=0, choices=(0, 1))
    p.add_argument("--augment_prob", type=float, default=1.0)
    p.add_argument("--speed_prob", type=float, default=0.5)
    p.add_argument("--speed_factors", type=str, default="0.9,1.0,1.1")
    p.add_argument("--noise_prob", type=float, default=0.5)
    p.add_argument("--noise_snr_min", type=float, default=5.0)
    p.add_argument("--noise_snr_max", type=float, default=20.0)
    p.add_argument("--specaug_prob", type=float, default=0.5)
    p.add_argument("--specaug_time_mask_param", type=int, default=50)
    p.add_argument("--specaug_freq_mask_param", type=int, default=27)
    p.add_argument("--specaug_num_time_masks", type=int, default=2)
    p.add_argument("--specaug_num_freq_masks", type=int, default=2)

    # LoRA (default off)
    p.add_argument("--use_lora", type=int, default=0, choices=(0, 1))
    p.add_argument("--lora_scope", type=str, default="encoder_aligner",
                   choices=["encoder", "aligner", "encoder_aligner",
                            "encoder_b4_aligner", "llm", "all"])
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_bias", type=str, default="none")
    p.add_argument("--merge_lora_into_base_from", type=str, default="",
                   help="Path to a previous LoRA adapter to merge into base model before training a new LoRA stage.")
    p.add_argument("--lr_encoder", type=float, default=2e-5,
                   help="Learning rate for speech encoder LoRA params (per-module LR).")
    p.add_argument("--lr_aligner", type=float, default=2e-5,
                   help="Learning rate for audio-text aligner LoRA params (per-module LR).")
    p.add_argument("--lr_llm", type=float, default=2e-5,
                   help="Learning rate for LLM LoRA params (per-module LR).")

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
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    use_lora = apply_lora(model, args_cli)
    if use_lora:
        print("[lora] scope=%s r=%d alpha=%d dropout=%s"
              % (args_cli.lora_scope, args_cli.lora_r, args_cli.lora_alpha, args_cli.lora_dropout))

    raw_ds = load_dataset(
        "json",
        data_files={
            "train": args_cli.train_file,
            **({"validation": args_cli.eval_file} if args_cli.eval_file else {}),
        },
    )
    ds = raw_ds.map(make_preprocess_fn_prefix_only(processor), num_proc=1)

    keep = {"prompt", "audio", "target", "prefix_text"}
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    augment_cfg = AudioAugmentConfig.from_args(args_cli)
    train_collator = DataCollatorForQwen3ASRFinetuning(
        processor=processor,
        sampling_rate=args_cli.sr,
        augment=augment_cfg if augment_cfg.enabled else None,
    )
    eval_collator = DataCollatorForQwen3ASRFinetuning(
        processor=processor,
        sampling_rate=args_cli.sr,
        augment=None,
    )

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
        save_total_limit=args_cli.save_total_limit,
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
