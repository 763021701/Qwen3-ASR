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
import argparse
import importlib.util
import os
import re
import shutil
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import librosa
import torch
from datasets import load_dataset
from qwen_asr import Qwen3ASRModel
from transformers import (GenerationConfig, Trainer, TrainerCallback,
                          TrainingArguments)


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
        ctc_labels=None,
        ctc_label_lengths=None,
        return_ctc_logits=False,
        **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            ctc_labels=ctc_labels,
            ctc_label_lengths=ctc_label_lengths,
            return_ctc_logits=return_ctc_logits,
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


def default_ctc_vocab_path() -> str:
    candidates = [
        os.environ.get("CTC_VOCAB_PATH", ""),
        "/root/.cache/modelscope/hub/models/FunAudioLLM/Fun-ASR-Nano-2512/multilingual.tiktoken",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Fun-ASR-Nano-2512", "multilingual.tiktoken")),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return candidates[1]


def maybe_add_funasr_path(funasr_path: str):
    candidates = [
        funasr_path,
        os.environ.get("FUNASR_PATH", ""),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "FunASR")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Fun-ASR")),
    ]
    for path in candidates:
        if path and os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


def load_ctc_tokenizer(vocab_path: str, funasr_path: str = ""):
    maybe_add_funasr_path(funasr_path)
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"CTC vocab not found: {vocab_path}")

    roots = [
        funasr_path,
        os.environ.get("FUNASR_PATH", ""),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "FunASR")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Fun-ASR")),
    ]
    for root in roots:
        if not root:
            continue
        candidates = [
            os.path.join(root, "funasr", "models", "sense_voice", "whisper_lib", "tokenizer.py"),
            os.path.join(root, "models", "sense_voice", "whisper_lib", "tokenizer.py"),
        ]
        for tokenizer_py in candidates:
            if os.path.exists(tokenizer_py):
                spec = importlib.util.spec_from_file_location("_qwen3_asr_sensevoice_tokenizer", tokenizer_py)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module.get_tokenizer(
                    multilingual=True,
                    num_languages=8749,
                    vocab_path=vocab_path,
                )

    from funasr.tokenizer.whisper_tokenizer import SenseVoiceTokenizer

    return SenseVoiceTokenizer(
        vocab_path=vocab_path,
        is_multilingual=True,
        num_languages=8749,
    )


def enable_ctc_training(model, vocab_path: str):
    thinker = model.thinker if hasattr(model, "thinker") else model
    audio_config = getattr(getattr(thinker, "config", None), "audio_config", None)
    ctc_input_dim = getattr(audio_config, "d_model", 1280)
    ctc_config = {
        "enabled": True,
        "input_dim": int(ctc_input_dim),
        "model_dim": 512,
        "ffn_dim": 2048,
        "n_layer": 5,
        "attention_heads": 8,
        "dropout": 0.0,
        "vocab_size": 60515,
        "blank_id": 60514,
        "time_step_sec": 0.08,
        "tokenizer": {
            "name": "SenseVoiceTokenizer",
            "vocab_path": vocab_path,
            "is_multilingual": True,
            "num_languages": 8749,
        },
    }
    thinker.enable_ctc(ctc_config)
    if hasattr(model, "config") and hasattr(model.config, "thinker_config"):
        model.config.thinker_config.ctc_config = ctc_config
    for _, param in model.named_parameters():
        param.requires_grad = False
    for module in (thinker.ctc_decoder, thinker.ctc_head):
        for param in module.parameters():
            param.requires_grad = True
    return ctc_config


def build_prefix_messages(prompt: str, audio_array):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def make_preprocess_fn_prefix_only(processor, ctc: bool = False):
    def _preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ex.get("prompt", "")
        dummy_audio = None
        prefix_msgs = build_prefix_messages(prompt, dummy_audio)
        prefix_text = processor.apply_chat_template(
            [prefix_msgs], add_generation_prompt=True, tokenize=False
        )[0]
        result = {
            "prompt": prompt,
            "audio": ex["audio"],
            "target": ex["text"],
            "prefix_text": prefix_text,
        }
        if ctc:
            # Lazy import: non-CTC finetuning never pulls in the TN deps.
            from finetuning.tn.normalize import normalize_ctc_text

            result["ctc_target"] = normalize_ctc_text(ex["text"])
        return result

    return _preprocess


@dataclass
class DataCollatorForQwen3ASRFinetuning:
    processor: Any
    sampling_rate: int = 16000
    ctc_tokenizer: Any = None
    train_ctc_only: bool = False

    def _build_ctc_labels(self, targets: List[str]) -> Dict[str, torch.Tensor]:
        if self.ctc_tokenizer is None:
            return {}
        token_tensors = [torch.tensor(self.ctc_tokenizer.encode(text), dtype=torch.long) for text in targets]
        lengths = torch.tensor([tokens.numel() for tokens in token_tensors], dtype=torch.long)
        max_len = max(int(lengths.max().item()) if len(lengths) > 0 else 0, 1)
        labels = torch.full((len(token_tensors), max_len), fill_value=0, dtype=torch.long)
        for i, tokens in enumerate(token_tensors):
            if tokens.numel() > 0:
                labels[i, : tokens.numel()] = tokens
        return {"ctc_labels": labels, "ctc_label_lengths": lengths}

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_paths = [f["audio"] for f in features]
        prefix_texts = [f["prefix_text"] for f in features]
        targets = [f["target"] for f in features]
        ctc_targets = [f.get("ctc_target", f["target"]) for f in features]
        audios = [load_audio(p, sr=self.sampling_rate) for p in audio_paths]

        prefix_inputs = self.processor(
            text=prefix_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        if self.train_ctc_only:
            prefix_inputs.update(self._build_ctc_labels(ctc_targets))
            return prefix_inputs

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]
        full_inputs = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            labels[i, :pl] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        full_inputs["labels"] = labels
        full_inputs.update(self._build_ctc_labels(ctc_targets))
        return full_inputs


class CastFloatInputsTrainer(Trainer):
    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=model_dtype)
        return inputs


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
    p = argparse.ArgumentParser("Qwen3-ASR Finetuning")

    # Paths
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--train_file", type=str, default="train.jsonl")
    p.add_argument("--eval_file", type=str, default="")
    p.add_argument("--output_dir", type=str, default="./qwen3-asr-finetuning-out")

    # Audio
    p.add_argument("--sr", type=int, default=16000)

    # CTC auxiliary branch
    p.add_argument("--train_ctc_only", type=int, default=0)
    p.add_argument("--ctc_vocab_path", type=str, default="")
    p.add_argument("--funasr_path", type=str, default="")

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

    ctc_tokenizer = None
    if args_cli.train_ctc_only == 1:
        ctc_vocab_path = args_cli.ctc_vocab_path or default_ctc_vocab_path()
        ctc_tokenizer = load_ctc_tokenizer(ctc_vocab_path, args_cli.funasr_path)
        enable_ctc_training(model, ctc_vocab_path)

    raw_ds = load_dataset(
        "json",
        data_files={
            "train": args_cli.train_file,
            **({"validation": args_cli.eval_file} if args_cli.eval_file else {}),
        },
    )
    ctc_enabled = ctc_tokenizer is not None
    ds = raw_ds.map(make_preprocess_fn_prefix_only(processor, ctc=ctc_enabled), num_proc=1)

    keep = {"prompt", "audio", "target", "prefix_text", "ctc_target"}
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    collator = DataCollatorForQwen3ASRFinetuning(
        processor=processor,
        sampling_rate=args_cli.sr,
        ctc_tokenizer=ctc_tokenizer,
        train_ctc_only=(args_cli.train_ctc_only == 1),
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
        eval_steps=args_cli.save_steps,
        do_eval=bool(args_cli.eval_file),
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", None),
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=[MakeEveryCheckpointInferableCallback(base_model_path=args_cli.model_path)],
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
