#!/usr/bin/env python3
"""Separate pure CTC loss from coverage penalty on a fixed sample."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from typing import Any, Dict, List

import librosa
import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from finetuning.ctc_auxiliary import (
    CTCVocabulary,
    SenseVoiceCTCTokenizer,
    attach_ctc_head,
)
from qwen_asr import Qwen3ASRModel
from safetensors.torch import load_file


def _load_head_weights(model_dir: str) -> Dict[str, torch.Tensor]:
    state = load_file(os.path.join(model_dir, "model.safetensors"))
    prefix = "thinker.ctc_head."
    weights = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if not weights:
        raise KeyError(f"No {prefix}* weights in {model_dir}")
    return weights


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pct(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Checkpoint with ctc_head weights.")
    parser.add_argument("--ctc_tokenizer", required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--ctc_vocab", default="", help="Defaults to ctc_vocab.json in model dir.")
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model)
    tokenizer = SenseVoiceCTCTokenizer(args.ctc_tokenizer)
    vocab_path = args.ctc_vocab or os.path.join(model_dir, "ctc_vocab.json")
    vocab = CTCVocabulary.load(vocab_path, tokenizer)

    wrapper = Qwen3ASRModel.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map=None, max_new_tokens=512
    )
    model = wrapper.model.to(args.device).eval()
    processor = wrapper.processor
    head = attach_ctc_head(model, vocab.size)
    head.load_state_dict(_load_head_weights(model_dir))
    head.eval()

    rows = [json.loads(line) for line in open(args.train_jsonl, encoding="utf-8") if line.strip()]
    random.Random(1234).shuffle(rows)
    rows = rows[: args.num_samples]

    records = []
    for row in rows:
        wav, _ = librosa.load(row["audio"], sr=16000, mono=True)
        features = processor.feature_extractor(
            [wav], sampling_rate=16000, return_tensors="pt", padding=True, return_attention_mask=True
        )
        input_features = features["input_features"].to(args.device, dtype=torch.bfloat16)
        feature_mask = features["attention_mask"].to(args.device)
        length = int(feature_mask.sum().item())
        with torch.no_grad():
            hidden = model.thinker.audio_tower(
                input_features[0, :, :length],
                feature_lens=torch.tensor([length], device=args.device),
            ).last_hidden_state
            logits = head(hidden.float())
            log_probs = logits.float().log_softmax(dim=-1).unsqueeze(1).contiguous()

        ids = vocab.encode(row["text"])
        repeats = sum(1 for i in range(1, len(ids)) if ids[i] == ids[i - 1])
        target = torch.tensor(ids, dtype=torch.long, device=args.device).unsqueeze(0)
        t = hidden.size(0)
        u = len(ids)
        need = u + repeats
        pure = float(
            F.ctc_loss(
                log_probs,
                target,
                torch.tensor([t], device=args.device),
                torch.tensor([u], device=args.device),
                blank=0,
                reduction="none",
                zero_infinity=True,
            )
        )
        probs = log_probs[:, 0, :].exp()
        coverage = float((1.0 - probs[:, 0]).sum())
        penalty = max(0.0, u - coverage) ** 2
        blank_ratio = float((logits.argmax(dim=-1) == 0).float().mean())
        records.append(
            {
                "audio": row["audio"],
                "T": t,
                "U": u,
                "R": repeats,
                "need": need,
                "margin": t / need if need else 0.0,
                "pure_ctc": pure,
                "coverage": coverage,
                "coverage_penalty": penalty,
                "blank_ratio": blank_ratio,
                "zero_inf": u > 0 and pure == 0.0,
            }
        )

    summary = {
        "model": model_dir,
        "n": len(records),
        "pure_ctc_mean": _mean([r["pure_ctc"] for r in records]),
        "pure_ctc_p50": _pct([r["pure_ctc"] for r in records], 0.5),
        "coverage_mean": _mean([r["coverage"] for r in records]),
        "coverage_penalty_mean": _mean([r["coverage_penalty"] for r in records]),
        "coverage_penalty_max": max(r["coverage_penalty"] for r in records),
        "blank_ratio_mean": _mean([r["blank_ratio"] for r in records]),
        "zero_inf_samples": sum(1 for r in records if r["zero_inf"]),
        "need_gt_T": sum(1 for r in records if r["need"] > r["T"]),
        "u_gt_T": sum(1 for r in records if r["U"] > r["T"]),
        "margin_min": min(r["margin"] for r in records),
        "margin_p05": _pct([r["margin"] for r in records], 0.05),
        "margin_p50": _pct([r["margin"] for r in records], 0.5),
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "records": records}, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()