#!/usr/bin/env python3
# coding=utf-8
"""Average two Qwen3-ASR checkpoints into a standalone, loadable model.

For every tensor:

    theta_C = alpha * theta_A + (1 - alpha) * theta_B

``alpha=1`` is a copy of A; ``alpha=0`` is a copy of B. Values outside [0, 1]
are allowed (extrapolation). Non-floating tensors must be identical and are
copied from A.

The output directory is a complete inference checkpoint: averaged
``model.safetensors`` plus tokenizer / processor / config files copied from A
(same set as a HuggingFace ``checkpoint-*`` minus trainer-only artifacts such
as ``optimizer.pt``). Load with ``Qwen3ASRModel.from_pretrained(output)``.

Usage:
    python finetuning/qwen3_asr_average.py \\
        --model_a outputs/<run_a>/checkpoint-250 \\
        --model_b outputs/<run_b>/checkpoint-250 \\
        --alpha 0.5 \\
        --output outputs/averaged_a_b_alpha0.5
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from typing import Dict, List, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Weight files we write ourselves (never copy from A/B).
_WEIGHT_EXACT = {
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
}

# Trainer resume artifacts — not part of an inference-ready averaged model.
_TRAINER_EXACT = {
    "optimizer.pt",
    "optimizer.bin",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Average two Qwen3-ASR checkpoints: "
            "C = alpha * A + (1 - alpha) * B."
        )
    )
    p.add_argument("--model_a", required=True, help="Checkpoint directory for theta_A.")
    p.add_argument("--model_b", required=True, help="Checkpoint directory for theta_B.")
    p.add_argument(
        "--alpha",
        "-a",
        type=float,
        default=0.5,
        help="Weight on A. C = alpha * A + (1 - alpha) * B. Default: 0.5.",
    )
    p.add_argument("--output", required=True, help="Output directory for the averaged model.")
    return p.parse_args()


def _is_weight_filename(name: str) -> bool:
    if name in _WEIGHT_EXACT:
        return True
    if name.startswith("model-") and name.endswith(".safetensors"):
        return True
    if name.startswith("pytorch_model-") and name.endswith(".bin"):
        return True
    if name in {"adapter_model.safetensors", "adapter_model.bin", "adapter_config.json"}:
        return True
    return False


def _is_trainer_filename(name: str) -> bool:
    if name in _TRAINER_EXACT:
        return True
    if name.startswith("rng_state"):
        return True
    return False


def list_safetensor_files(ckpt_dir: str) -> List[str]:
    """Return safetensor shard paths for a HuggingFace checkpoint directory."""
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    single_path = os.path.join(ckpt_dir, "model.safetensors")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid or empty weight_map in {index_path}")
        files = sorted({os.path.join(ckpt_dir, fname) for fname in weight_map.values()})
        missing = [p for p in files if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(f"Missing shard(s) listed in index: {missing}")
        return files
    if os.path.isfile(single_path):
        return [single_path]
    shards = sorted(glob.glob(os.path.join(ckpt_dir, "model-*-of-*.safetensors")))
    if shards:
        return shards
    adapter = os.path.join(ckpt_dir, "adapter_model.safetensors")
    if os.path.isfile(adapter):
        raise FileNotFoundError(
            f"No full-model weights in {ckpt_dir} (found a LoRA adapter). "
            "Merge the adapter first with tools/merge_lora.py."
        )
    raise FileNotFoundError(
        f"No model.safetensors (or shards) in {ckpt_dir}."
    )


# HuggingFace omits the tied lm_head from safetensors when tie_word_embeddings=true.
_TIED_HEAD_TO_EMBED = {
    "thinker.lm_head.weight": "thinker.model.embed_tokens.weight",
}


def fill_tied_weights(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Alias omitted tied lm_head <-> embed_tokens so two checkpoints can be aligned."""
    out = dict(state)
    for head, embed in _TIED_HEAD_TO_EMBED.items():
        if head not in out and embed in out:
            out[head] = out[embed]
        elif embed not in out and head in out:
            out[embed] = out[head]
    return out


def load_weight_tensors(ckpt_dir: str) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, str]]]:
    """Load all tensors from a checkpoint. Returns (state_dict, safetensors metadata)."""
    files = list_safetensor_files(ckpt_dir)
    state: Dict[str, torch.Tensor] = {}
    metadata: Optional[Dict[str, str]] = None
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            if metadata is None:
                raw = handle.metadata()
                metadata = dict(raw) if raw else None
            for key in handle.keys():
                if key in state:
                    raise ValueError(f"Duplicate tensor key {key!r} while reading {path}")
                state[key] = handle.get_tensor(key)
    if not state:
        raise ValueError(f"No tensors found under {ckpt_dir}")
    return state, metadata


def average_state_dicts(
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
    alpha: float,
) -> Dict[str, torch.Tensor]:
    """Return C = alpha * A + (1 - alpha) * B for every floating tensor."""
    keys_a = set(state_a)
    keys_b = set(state_b)
    if keys_a != keys_b:
        only_a = sorted(keys_a - keys_b)
        only_b = sorted(keys_b - keys_a)
        raise ValueError(
            "Checkpoint tensor keys do not match.\n"
            f"  only in A ({len(only_a)}): {only_a[:20]}\n"
            f"  only in B ({len(only_b)}): {only_b[:20]}"
        )
    averaged: Dict[str, torch.Tensor] = {}
    for key in state_a:
        tensor_a = state_a[key]
        tensor_b = state_b[key]
        if tensor_a.shape != tensor_b.shape:
            raise ValueError(
                f"Shape mismatch for {key!r}: A={tuple(tensor_a.shape)} B={tuple(tensor_b.shape)}"
            )
        if tensor_a.dtype != tensor_b.dtype:
            raise ValueError(
                f"Dtype mismatch for {key!r}: A={tensor_a.dtype} B={tensor_b.dtype}"
            )
        if tensor_a.is_floating_point():
            if alpha == 1.0:
                averaged[key] = tensor_a.clone()
            elif alpha == 0.0:
                averaged[key] = tensor_b.clone()
            else:
                mixed = alpha * tensor_a.float() + (1.0 - alpha) * tensor_b.float()
                averaged[key] = mixed.to(dtype=tensor_a.dtype)
        else:
            if not torch.equal(tensor_a, tensor_b):
                raise ValueError(
                    f"Non-floating tensor {key!r} differs between A and B; cannot average."
                )
            averaged[key] = tensor_a.clone()
    return averaged


def _check_configs_compatible(model_a: str, model_b: str) -> None:
    cfg_a_path = os.path.join(model_a, "config.json")
    cfg_b_path = os.path.join(model_b, "config.json")
    if not os.path.isfile(cfg_a_path) or not os.path.isfile(cfg_b_path):
        return
    with open(cfg_a_path, "r", encoding="utf-8") as f:
        cfg_a = json.load(f)
    with open(cfg_b_path, "r", encoding="utf-8") as f:
        cfg_b = json.load(f)
    for field in ("architectures", "model_type"):
        if cfg_a.get(field) != cfg_b.get(field):
            raise ValueError(
                f"config.json {field} mismatch: A={cfg_a.get(field)!r} B={cfg_b.get(field)!r}"
            )


def copy_sidecar_files(src_dir: str, dst_dir: str) -> List[str]:
    """Copy non-weight, non-trainer files from src into dst. Returns copied names."""
    copied: List[str] = []
    for name in sorted(os.listdir(src_dir)):
        src_path = os.path.join(src_dir, name)
        if not os.path.isfile(src_path):
            continue
        if _is_weight_filename(name) or _is_trainer_filename(name):
            continue
        shutil.copy2(src_path, os.path.join(dst_dir, name))
        copied.append(name)
    return copied


def average_checkpoints(model_a: str, model_b: str, output: str, alpha: float) -> str:
    """Average two checkpoints and write a complete model directory. Returns output path."""
    model_a = os.path.abspath(model_a)
    model_b = os.path.abspath(model_b)
    output = os.path.abspath(output)
    if output in {model_a, model_b}:
        raise ValueError("--output must be a new directory, not model_a or model_b.")
    _check_configs_compatible(model_a, model_b)

    print(f"[average] C = {alpha:g} * A + {1.0 - alpha:g} * B")
    print(f"[load] A={model_a}")
    state_a, metadata = load_weight_tensors(model_a)
    print(f"[load] B={model_b}")
    state_b, _ = load_weight_tensors(model_b)
    keys_on_disk_a = set(state_a)
    n_a, n_b = len(state_a), len(state_b)
    state_a = fill_tied_weights(state_a)
    state_b = fill_tied_weights(state_b)
    if len(state_a) != n_a or len(state_b) != n_b:
        print("[average] filled omitted tied lm_head from embed_tokens")
    print(f"[average] tensors={len(state_a)}")
    averaged = average_state_dicts(state_a, state_b, alpha)
    # Keep A's on-disk layout (do not write a tied lm_head A never stored).
    averaged = {key: tensor for key, tensor in averaged.items() if key in keys_on_disk_a}
    # Free source tensors before the safetensors write.
    del state_a, state_b

    os.makedirs(output, exist_ok=True)
    save_kwargs = {"metadata": metadata} if metadata else {}
    weight_path = os.path.join(output, "model.safetensors")
    save_file(averaged, weight_path, **save_kwargs)
    copied = copy_sidecar_files(model_a, output)
    print(f"[save] {weight_path}")
    print(f"[copy] {len(copied)} sidecar files from A")
    print(f"[done] output={output}")
    return output


def main() -> None:
    args = parse_args()
    average_checkpoints(args.model_a, args.model_b, args.output, args.alpha)


if __name__ == "__main__":
    main()
