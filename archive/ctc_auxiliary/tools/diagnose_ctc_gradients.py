#!/usr/bin/env python3
"""Measure CE-vs-CTC gradient conflict on the shared audio tower."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from typing import Any, Dict, List, Tuple

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from finetuning.ctc_auxiliary import (
    AudioTowerCapture,
    CTCVocabulary,
    SenseVoiceCTCTokenizer,
    attach_ctc_head,
    compute_ctc_loss,
)
from finetuning.nospeech_augmentation import NoSpeechAugmentConfig
from finetuning.qwen3_asr_sft import (
    AudioAugmentConfig,
    DataCollatorForQwen3ASRFinetuning,
    make_preprocess_fn_prefix_only,
)
from qwen_asr import Qwen3ASRModel
from safetensors.torch import load_file

_LAYER_RE = re.compile(r"audio_tower\.layers\.(\d+)\.")


def _load_head_weights(model_dir: str) -> Dict[str, torch.Tensor]:
    state = load_file(os.path.join(model_dir, "model.safetensors"))
    prefix = "thinker.ctc_head."
    weights = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if not weights:
        raise KeyError(f"No {prefix}* weights in {model_dir}")
    return weights


def _group(name: str) -> str:
    match = _LAYER_RE.search(name)
    if match:
        return "layer-%02d" % int(match.group(1))
    for part in (
        "conv2d1",
        "conv2d2",
        "conv2d3",
        "conv_out",
        "proj1",
        "proj2",
        "ln_post",
        "positional_embedding",
    ):
        if part in name:
            return part
    return "other"


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    a = left.float().flatten()
    b = right.float().flatten()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 0.0
    return float(torch.dot(a, b) / denom)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ctc_tokenizer", required=True)
    parser.add_argument("--ctc_vocab", default="")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--num_batches", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lambda_ctc", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--augment", type=int, default=0, choices=(0, 1))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model)
    tokenizer = SenseVoiceCTCTokenizer(args.ctc_tokenizer)
    vocab = CTCVocabulary.load(args.ctc_vocab or os.path.join(model_dir, "ctc_vocab.json"), tokenizer)

    wrapper = Qwen3ASRModel.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map=None, max_new_tokens=512
    )
    model = wrapper.model.to(args.device)
    processor = wrapper.processor
    head = attach_ctc_head(model, vocab.size)
    head.load_state_dict(_load_head_weights(model_dir))
    head.train()
    model.train()

    rows = [json.loads(line) for line in open(args.train_jsonl, encoding="utf-8") if line.strip()]
    random.Random(args.seed).shuffle(rows)
    rows = [make_preprocess_fn_prefix_only(processor)(row) for row in rows[: args.num_batches * args.batch_size]]
    augment_cfg = None
    if args.augment:
        augment_cfg = AudioAugmentConfig(
            enabled=True,
            augment_prob=1.0,
            speed_prob=0.5,
            speed_factors=(0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2),
            noise_prob=0.0,
            specaug_prob=0.5,
            specaug_time_mask_param=50,
            specaug_freq_mask_param=27,
            specaug_num_time_masks=2,
            specaug_num_freq_masks=2,
            nospeech=NoSpeechAugmentConfig(
                enabled=True, prob=0.3, pad_min_sec=0.5, pad_max_sec=3.0
            ),
        )
    collator = DataCollatorForQwen3ASRFinetuning(
        processor=processor,
        sampling_rate=16000,
        augment=augment_cfg,
        ctc_vocab=vocab,
    )
    capture = AudioTowerCapture(model.thinker.audio_tower)
    auT_params: List[Tuple[str, torch.nn.Parameter]] = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if "audio_tower" in name and parameter.requires_grad
    ]
    params = [parameter for _, parameter in auT_params]

    records: List[Dict[str, Any]] = []
    layer_records: Dict[str, List[Dict[str, float]]] = {}
    for batch_index in range(args.num_batches):
        batch_rows = rows[batch_index * args.batch_size : (batch_index + 1) * args.batch_size]
        batch = collator(batch_rows)
        ctc_targets = batch.pop("ctc_targets").to(args.device)
        ctc_lengths = batch.pop("ctc_target_lengths").to(args.device)
        batch = {
            key: (value.to(args.device) if torch.is_tensor(value) else value)
            for key, value in batch.items()
        }
        batch["input_features"] = batch["input_features"].to(torch.bfloat16)

        capture.start()
        outputs = model.thinker(**batch)
        capture.stop()
        loss_ce = outputs.loss
        loss_ctc = compute_ctc_loss(
            capture.hidden_states,
            head,
            ctc_targets,
            ctc_lengths,
            coverage_weight=0.0,
        )
        hidden_list = capture.hidden_states
        grads_ce_all = torch.autograd.grad(
            loss_ce, params + hidden_list, retain_graph=True, allow_unused=True
        )
        grads_ctc_all = torch.autograd.grad(
            loss_ctc, params + hidden_list, retain_graph=False, allow_unused=True
        )
        grads_ce = grads_ce_all[: len(params)]
        grads_ctc = grads_ctc_all[: len(params)]
        hidden_grads_ce = grads_ce_all[len(params) :]
        hidden_grads_ctc = grads_ctc_all[len(params) :]

        feature_cosines = []
        flat_hidden_ce = []
        flat_hidden_ctc = []
        for grad_ce, grad_ctc in zip(hidden_grads_ce, hidden_grads_ctc):
            if grad_ce is None or grad_ctc is None:
                continue
            feature_cosines.append(_cosine(grad_ce, grad_ctc))
            flat_hidden_ce.append(grad_ce.detach().float().flatten())
            flat_hidden_ctc.append(grad_ctc.detach().float().flatten())
        feature_cosine_concat = (
            _cosine(torch.cat(flat_hidden_ce), torch.cat(flat_hidden_ctc))
            if flat_hidden_ce
            else 0.0
        )
        feature_norm_ce = (
            float(torch.cat(flat_hidden_ce).norm()) if flat_hidden_ce else 0.0
        )
        feature_norm_ctc = (
            float(torch.cat(flat_hidden_ctc).norm()) if flat_hidden_ctc else 0.0
        )
        feature_weighted_ratio = (
            args.lambda_ctc * feature_norm_ctc / feature_norm_ce
            if feature_norm_ce
            else 0.0
        )

        flat_ce = []
        flat_ctc = []
        record: Dict[str, Any] = {
            "batch": batch_index,
            "loss_ce": float(loss_ce.detach()),
            "loss_ctc": float(loss_ctc.detach()),
            "layers": {},
        }
        for (name, _), grad_ce, grad_ctc in zip(auT_params, grads_ce, grads_ctc):
            if grad_ce is None or grad_ctc is None:
                continue
            flat_ce.append(grad_ce.detach().float().flatten())
            flat_ctc.append(grad_ctc.detach().float().flatten())
            group = _group(name)
            cos = _cosine(grad_ce, grad_ctc)
            norm_ce = float(grad_ce.detach().float().norm())
            norm_ctc = float(grad_ctc.detach().float().norm())
            layer_records.setdefault(group, []).append(
                {"cos": cos, "norm_ce": norm_ce, "norm_ctc": norm_ctc}
            )
            record["layers"].setdefault(group, []).append(
                {"cos": cos, "norm_ce": norm_ce, "norm_ctc": norm_ctc}
            )

        ce = torch.cat(flat_ce)
        ctc = torch.cat(flat_ctc)
        record["cosine"] = _cosine(ce, ctc)
        record["norm_ce"] = float(ce.norm())
        record["norm_ctc"] = float(ctc.norm())
        record["weighted_ratio"] = (
            args.lambda_ctc * record["norm_ctc"] / record["norm_ce"]
            if record["norm_ce"]
            else 0.0
        )
        record["feature_cosines"] = feature_cosines
        record["feature_cosine_mean"] = (
            sum(feature_cosines) / len(feature_cosines) if feature_cosines else 0.0
        )
        record["feature_cosine_concat"] = feature_cosine_concat
        record["feature_norm_ce"] = feature_norm_ce
        record["feature_norm_ctc"] = feature_norm_ctc
        record["feature_weighted_ratio"] = feature_weighted_ratio
        records.append(record)
        print(
            "[batch %d] loss_ce=%.3f loss_ctc=%.3f cos=%.3f feat_cos=%.3f |g_ce|=%.3f |g_ctc|=%.3f ratio=%.3f"
            % (
                batch_index,
                record["loss_ce"],
                record["loss_ctc"],
                record["cosine"],
                record["feature_cosine_mean"],
                record["norm_ce"],
                record["norm_ctc"],
                record["weighted_ratio"],
            )
        )

    layer_summary = {}
    for group, items in layer_records.items():
        cosines = [item["cos"] for item in items]
        norms_ce = [item["norm_ce"] for item in items]
        norms_ctc = [item["norm_ctc"] for item in items]
        layer_summary[group] = {
            "params": len(items),
            "cos_mean": sum(cosines) / len(cosines),
            "cos_min": min(cosines),
            "cos_max": max(cosines),
            "negative_fraction": sum(1 for c in cosines if c < 0) / len(cosines),
            "norm_ce_mean": sum(norms_ce) / len(norms_ce),
            "norm_ctc_mean": sum(norms_ctc) / len(norms_ctc),
            "weighted_ratio": (
                args.lambda_ctc * sum(norms_ctc) / sum(norms_ce) if sum(norms_ce) else 0.0
            ),
        }

    overall_cos = sum(record["cosine"] for record in records) / len(records)
    overall_ratio = sum(record["weighted_ratio"] for record in records) / len(records)
    summary = {
        "model": model_dir,
        "num_batches": len(records),
        "batch_size": args.batch_size,
        "lambda_ctc": args.lambda_ctc,
        "overall_cosine_mean": overall_cos,
        "overall_weighted_ratio_mean": overall_ratio,
        "overall_feature_cosine_mean": sum(
            record["feature_cosine_mean"] for record in records
        )
        / len(records),
        "overall_feature_cosine_concat": sum(
            record["feature_cosine_concat"] for record in records
        )
        / len(records),
        "overall_feature_weighted_ratio_mean": sum(
            record["feature_weighted_ratio"] for record in records
        )
        / len(records),
        "layers": dict(sorted(layer_summary.items())),
        "batches": records,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("\n%-24s %6s %8s %8s %8s %8s" % ("layer", "params", "cos_mean", "cos_min", "neg%", "w_ratio"))
    for group, item in sorted(layer_summary.items()):
        print(
            "%-24s %6d %8.3f %8.3f %7.1f%% %8.3f"
            % (
                group,
                item["params"],
                item["cos_mean"],
                item["cos_min"],
                100 * item["negative_fraction"],
                item["weighted_ratio"],
            )
        )
    print(
        "overall cosine mean %.3f, weighted ratio mean %.3f, feature cosine mean %.3f, feature ratio mean %.3f"
        % (
            overall_cos,
            overall_ratio,
            summary["overall_feature_cosine_mean"],
            summary["overall_feature_weighted_ratio_mean"],
        )
    )
    print("wrote", args.output)


if __name__ == "__main__":
    main()