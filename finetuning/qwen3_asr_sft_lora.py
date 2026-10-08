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
import json
import math
import re
import shutil
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from qwen_asr import Qwen3ASRModel
from transformers import (GenerationConfig, Trainer, TrainerCallback,
                          TrainingArguments)

# Share the SFT data recipe and generation metric for controlled comparisons.
if __package__:
    from .qwen3_asr_sft import (
        AudioAugmentConfig, DataCollatorForQwen3ASRFinetuning,
        KeepBestWerCheckpointsCallback, _load_extra_eval_items,
        make_preprocess_fn_prefix_only,
    )
else:
    from qwen3_asr_sft import (
        AudioAugmentConfig, DataCollatorForQwen3ASRFinetuning,
        KeepBestWerCheckpointsCallback, _load_extra_eval_items,
        make_preprocess_fn_prefix_only,
    )

# Audio attention projections live in self_attn; fc1/fc2 are direct layer children.
LORA_TARGETS = {
    "encoder": r"^audio_tower\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|out_proj)|fc1|fc2)$",
    "aligner": r"^audio_tower\.(conv_out|proj1|proj2)$",
    "encoder_aligner": (
        r"^(audio_tower\.(conv_out|proj1|proj2)$"
        r"|audio_tower\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|out_proj)|fc1|fc2)$)"
    ),
    "encoder_b4_aligner": (
        r"^(audio_tower\.(conv_out|proj1|proj2)$"
        r"|audio_tower\.layers\.(20|21|22|23)\.(?:self_attn\.(?:q_proj|k_proj|v_proj|out_proj)|fc1|fc2)$)"
    ),
    "llm": r"^model\.layers\.\d+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
    "all": "all-linear",
}

def _resolve_lora_scope(scope: str) -> str:
    """Resolve --lora_scope to a target regex or PEFT's ``all-linear``.

    Accepts a scope name or a comma-separated combination of scope names
    (e.g. ``"aligner,llm"``); named parts are combined as a regex union.
    ``all`` takes precedence and delegates linear-layer selection and output-head
    exclusions to PEFT. Parts without LoRA stay fully frozen.
    """
    tokens = [t.strip() for t in scope.split(",") if t.strip()]
    if not tokens:
        raise ValueError("--lora_scope must name at least one part.")
    regexes = []
    for token in tokens:
        if token not in LORA_TARGETS:
            raise ValueError(f"Unknown lora_scope part {token!r}. Choices: {', '.join(LORA_TARGETS)}")
        regexes.append(LORA_TARGETS[token])
    return LORA_TARGETS["all"] if "all" in tokens else "|".join(regexes)

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
    target_modules = _resolve_lora_scope(scope)

    lora_config = LoraConfig(
        r=int(getattr(args, "lora_r", 8)),
        lora_alpha=int(getattr(args, "lora_alpha", 16)),
        lora_dropout=float(getattr(args, "lora_dropout", 0.05)),
        bias=str(getattr(args, "lora_bias", "none")),
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
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
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file as safe_load_file
            set_peft_model_state_dict(
                model.thinker, safe_load_file(adapter_path), adapter_name="default"
            )
            print(f"[resume] loaded LoRA adapter weights from {adapter_path}")
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

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, self.model)
        self.optimizer = optimizer_cls(optim_groups, **optimizer_kwargs)
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
    def on_train_begin(self, args, state, control, **kwargs):
        if args.process_index != 0:
            return control
        path = os.path.join(args.output_dir, self.ranking_filename)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data.get("checkpoints", []):
                self.eval_losses[int(item["step"])] = float(item["eval_loss"])
        return control


    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if args.process_index != 0 or not metrics:
            return control
        loss = metrics.get("eval_loss")
        if loss is not None and math.isfinite(float(loss)):
            self.eval_losses[int(state.global_step)] = float(loss)
        return control

    def _write_ranking(self, output_dir: str, ranked: List[Tuple[float, int, str]]) -> None:
        path = os.path.join(output_dir, self.ranking_filename)
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

    def on_save(self, args, state, control, **kwargs):
        if args.process_index != 0:
            return control

        ranked = []
        for step, loss in self.eval_losses.items():
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
            if os.path.isdir(ckpt_dir):
                ranked.append((loss, step, ckpt_dir))
        ranked.sort()

        for _, step, ckpt_dir in ranked[self.limit:]:
            shutil.rmtree(ckpt_dir)
            print(
                f"[best-checkpoints] removed checkpoint-{step}; "
                f"outside best {self.limit} by eval_loss"
            )

        kept = ranked[: self.limit]
        self._write_ranking(args.output_dir, kept)
        summary = ", ".join(
            f"checkpoint-{step}={loss:.6f}" for loss, step, _ in kept
        )
        print(f"[best-checkpoints] retained: {summary}")
        return control
class StopOnEvalLossPlateauCallback(TrainerCallback):
    """Stop after repeated evals without a meaningful eval-loss improvement."""

    def __init__(self, patience: int, threshold: float):
        if patience < 1:
            raise ValueError("Early-stopping patience must be positive.")
        self.patience = patience
        self.threshold = threshold
        self.best_loss: Optional[float] = None
        self.bad_evals = 0

    def on_train_begin(self, args, state, control, **kwargs):
        path = os.path.join(args.output_dir, KeepBestCheckpointsCallback.ranking_filename)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            losses = [
                float(item["eval_loss"])
                for item in data.get("checkpoints", [])
                if item.get("eval_loss") is not None
            ]
            if losses:
                self.best_loss = min(losses)
                print(f"[early-stop] restored best eval_loss={self.best_loss:.6f}")
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or metrics.get("eval_loss") is None:
            return control
        loss = float(metrics["eval_loss"])
        improved = self.best_loss is None or loss < self.best_loss - self.threshold
        if improved:
            self.best_loss = loss
            self.bad_evals = 0
        else:
            self.bad_evals += 1
        print(
            f"[early-stop] step={state.global_step} eval_loss={loss:.6f} "
            f"best={self.best_loss:.6f} bad_evals={self.bad_evals}/{self.patience}"
        )
        if self.bad_evals >= self.patience:
            control.should_training_stop = True
            print("[early-stop] convergence criterion reached; stopping training.")
        return control






def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR LoRA Finetuning")

    # Paths
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--train_file", type=str, default="train.jsonl")
    p.add_argument("--eval_file", type=str, default="")
    p.add_argument("--extra_eval_set", action="append", default=[],
                   help="Additional reporting set as NAME=PATH (jsonl/csv).")
    p.add_argument("--output_dir", type=str, default="./qwen3-asr-finetuning-out")

    # Audio
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--strip_target_brackets", type=int, default=0, choices=(0, 1))

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
    p.add_argument(
        "--save_best_total_limit",
        type=int,
        default=0,
        help="When >0 with --eval_file, retain this many checkpoints by --save_best_metric.",
    )
    p.add_argument("--save_best_metric", default="eval_loss",
                   choices=("eval_loss", "wer", "cer"))
    p.add_argument("--wer_eval_samples", type=int, default=400,
                   help="Dev subset size for generation selection; 0 uses all samples.")
    p.add_argument("--wer_batch_size", type=int, default=4)
    p.add_argument("--wer_max_new_tokens", type=int, default=512)
    p.add_argument("--early_stopping_patience", type=int, default=0)
    p.add_argument("--early_stopping_threshold", type=float, default=0.0)

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

    # LoRA (default off)
    p.add_argument("--use_lora", type=int, default=0, choices=(0, 1))
    p.add_argument("--lora_scope", type=str, default="encoder_aligner",
                   help="LoRA target: encoder, aligner, encoder_aligner, "
                        "encoder_b4_aligner, llm, or all (PEFT all-linear). "
                        "Comma-separated combinations are supported (e.g. 'aligner,llm'); "
                        "all takes precedence. Parts without LoRA stay frozen.")
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
        max_new_tokens=args_cli.wer_max_new_tokens,
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    use_lora = apply_lora(model, args_cli)
    if use_lora:
        print("[lora] scope=%s r=%d alpha=%d dropout=%s"
              % (args_cli.lora_scope, args_cli.lora_r, args_cli.lora_alpha, args_cli.lora_dropout))

    # Load splits separately so manifests with different column sets (e.g.
    # RAFT rows) don't hit the multi-split schema-unification CastError.
    raw_ds = {
        "train": load_dataset("json", data_files=str(args_cli.train_file), split="train")
    }
    if args_cli.eval_file:
        raw_ds["validation"] = load_dataset(
            "json", data_files=str(args_cli.eval_file), split="train"
        )
    ds = {
        split: raw_split.map(make_preprocess_fn_prefix_only(processor), num_proc=1)
        for split, raw_split in raw_ds.items()
    }

    keep = {"prompt", "audio", "target", "prefix_text", "aug", "noise_aug"}
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    augment_cfg = AudioAugmentConfig.from_args(args_cli)
    train_collator = DataCollatorForQwen3ASRFinetuning(
        processor=processor,
        sampling_rate=args_cli.sr,
        augment=augment_cfg if augment_cfg.enabled else None,
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
                metric=args_cli.save_best_metric,
                output_dir=args_cli.output_dir,
                extra_eval_sets=[
                    _load_extra_eval_items(spec) for spec in args_cli.extra_eval_set
                ],
            ))
        else:
            callbacks.append(KeepBestCheckpointsCallback(limit=keep_best_limit))
    if args_cli.early_stopping_patience > 0:
        if not args_cli.eval_file:
            raise ValueError("--early_stopping_patience requires --eval_file")
        callbacks.append(
            StopOnEvalLossPlateauCallback(
                patience=args_cli.early_stopping_patience,
                threshold=args_cli.early_stopping_threshold,
            )
        )

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
