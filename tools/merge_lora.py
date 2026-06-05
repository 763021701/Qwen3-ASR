#!/usr/bin/env python3
"""Merge a LoRA adapter into the base model and save as a standalone checkpoint.

The merged model can be loaded directly with Qwen3ASRModel.from_pretrained()
for inference, just like a full-finetune checkpoint.

Usage:
    python tools/merge_lora.py \
        --base_model Qwen/Qwen3-ASR-1.7B \
        --adapter outputs/ug_lora_all_aug/checkpoint-2000 \
        --output outputs/ug_lora_all_aug_merged
"""

import argparse
import os

import torch
from peft import PeftModel
from qwen_asr import Qwen3ASRModel


def main():
    p = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    p.add_argument("--base_model", required=True, help="Base model path or HF repo id")
    p.add_argument("--adapter", required=True, help="LoRA adapter directory (contains adapter_model.safetensors)")
    p.add_argument("--output", required=True, help="Output directory for merged model")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if not os.path.isdir(args.adapter):
        raise SystemExit(f"Adapter directory not found: {args.adapter}")
    adapter_file = os.path.join(args.adapter, "adapter_model.safetensors")
    if not os.path.isfile(adapter_file):
        raise SystemExit(f"adapter_model.safetensors not found in {args.adapter}")

    print(f"[1/4] Loading base model: {args.base_model}")
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    wrapper = Qwen3ASRModel.from_pretrained(
        args.base_model,
        dtype=dtype,
        device_map=None,  # load to CPU first for merging
    )
    model = wrapper.model  # Qwen3ASRForConditionalGeneration
    thinker = model.thinker

    print(f"[2/4] Loading LoRA adapter: {args.adapter}")
    thinker = PeftModel.from_pretrained(thinker, args.adapter, is_trainable=False)

    print(f"[3/4] Merging LoRA weights into base model...")
    thinker = thinker.merge_and_unload()
    model.thinker = thinker

    print(f"[4/4] Saving merged model to: {args.output}")
    # Fix generation_config: some base models have temperature set with do_sample=False
    # which newer transformers versions reject on save. Set do_sample=True to pass validation.
    if hasattr(model, "generation_config") and model.generation_config is not None:
        gc = model.generation_config
        if not getattr(gc, "do_sample", True):
            gc.do_sample = True
    model.save_pretrained(args.output, safe_serialization=True)
    # Save processor and tokenizer from the already-loaded wrapper
    wrapper.processor.save_pretrained(args.output)

    print(f"Done! Merged model saved to {args.output}")
    print(f"Load with: Qwen3ASRModel.from_pretrained('{args.output}')")


if __name__ == "__main__":
    main()
