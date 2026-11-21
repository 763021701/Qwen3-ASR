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
# LoRA fine-tuning for Qwen3-ASR. The data recipe, trainer, callbacks and
# training run come from finetuning/qwen3_asr_sft.py; this file only adds the
# LoRA setup and adapter checkpoints.
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

if __package__:
    from .qwen3_asr_sft import CastFloatInputsTrainer, build_arg_parser, run_training
else:
    from qwen3_asr_sft import CastFloatInputsTrainer, build_arg_parser, run_training

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


def prepare_lora_model(model, args_cli) -> list:
    """``run_training`` hook: wrap the thinker with LoRA; no extra callbacks."""
    if apply_lora(model, args_cli):
        print("[lora] scope=%s r=%d alpha=%d dropout=%s"
              % (args_cli.lora_scope, args_cli.lora_r, args_cli.lora_alpha, args_cli.lora_dropout))
    return []


class LoraTrainer(CastFloatInputsTrainer):
    """Checkpoints and resumes the PEFT adapter that wraps ``model.thinker``."""

    @staticmethod
    def _lr_group(name: str) -> str:
        """Per-module LRs apply to LoRA weights only; other trainable params use --lr."""
        if "lora_" not in name:
            return "other"
        if any(x in name for x in ["audio_tower.conv_out", "audio_tower.proj1", "audio_tower.proj2"]):
            return "aligner"
        if "audio_tower.layers." in name:
            return "encoder"
        if "model.layers." in name:
            return "llm"
        return "other"

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


def parse_args():
    p = build_arg_parser("Qwen3-ASR LoRA Finetuning")
    p.add_argument("--lr_encoder", type=float, default=2e-5,
                   help="Learning rate for speech encoder LoRA params (per-module LR).")
    p.add_argument("--lr_aligner", type=float, default=2e-5,
                   help="Learning rate for audio-text aligner LoRA params (per-module LR).")
    p.add_argument("--lr_llm", type=float, default=2e-5,
                   help="Learning rate for LLM LoRA params (per-module LR).")

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
    return p.parse_args()


def main():
    run_training(parse_args(), prepare_model=prepare_lora_model, trainer_cls=LoraTrainer)


if __name__ == "__main__":
    main()
