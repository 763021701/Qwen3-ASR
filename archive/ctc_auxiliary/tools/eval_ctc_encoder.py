#!/usr/bin/env python3
"""CTC-only greedy decode with the auxiliary head on the Qwen3-ASR audio tower.

This is a diagnostic: it answers "how much acoustic evidence does the AuT
retain?" independently of the LLM decoder.  The checkpoint must contain the
``thinker.ctc_head.*`` weights written by ``finetuning/qwen3_asr_sft.py
--ctc_loss_weight > 0``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import librosa
import torch
from safetensors.torch import load_file

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evaluation.english_medical.text_normalization import normalize_english
from finetuning.ctc_auxiliary import (
    CTCVocabulary,
    SenseVoiceCTCTokenizer,
    attach_ctc_head,
    ctc_greedy_decode,
    transcript_from_target,
)
from masr_eval_pkg import compute_wer
from qwen_asr import Qwen3ASRModel


def _find_sidecar(model_dir: str, name: str, explicit: str = "") -> str:
    candidates = [explicit] if explicit else []
    candidates.extend(
        [os.path.join(model_dir, name), os.path.join(os.path.dirname(model_dir), name)]
    )
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return ""


def _load_ctc_weights(model_dir: str) -> Dict[str, torch.Tensor]:
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.isfile(single):
        state = load_file(single)
    else:
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        bin_path = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.isfile(index_path):
            with open(index_path, "r", encoding="utf-8") as handle:
                index = json.load(handle)
            state = {}
            for shard in sorted(set(index["weight_map"].values())):
                state.update(load_file(os.path.join(model_dir, shard)))
        elif os.path.isfile(bin_path):
            state = torch.load(bin_path, map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(
                f"No model.safetensors, index, or pytorch_model.bin under {model_dir}"
            )

    prefix = "thinker.ctc_head."
    weights = {
        key[len(prefix) :]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    if not weights:
        raise KeyError(f"No {prefix}* weights found in {model_dir}")
    return weights


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


def _corpus_cer(references: List[str], hypotheses: List[str]) -> float:
    edits = 0
    total = 0
    for reference, hypothesis in zip(references, hypotheses):
        ref_chars = reference.replace(" ", "")
        hyp_chars = hypothesis.replace(" ", "")
        edits += _edit_distance(ref_chars, hyp_chars)
        total += len(ref_chars)
    return edits / total if total else 0.0


def _load_rows(path: str, limit: int) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Checkpoint with ctc_head weights.")
    parser.add_argument("--ctc_tokenizer", required=True, help="SenseVoice BPE model path.")
    parser.add_argument(
        "--ctc_vocab",
        default="",
        help="ctc_vocab.json path; defaults to the checkpoint or its run directory.",
    )
    parser.add_argument(
        "--ctc_head",
        default="",
        help="Standalone ctc_head.safetensors from tools/pretrain_ctc_head.py; "
             "when set, --model may be the base model.",
    )
    parser.add_argument("--eval_jsonl", required=True, help="Rows with audio + text.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    wrapper = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=None,
        max_new_tokens=512,
    )
    model = wrapper.model.to(args.device)
    model.eval()
    processor = wrapper.processor
    tokenizer = SenseVoiceCTCTokenizer(args.ctc_tokenizer)

    sidecar_dir = (
        os.path.dirname(os.path.abspath(args.ctc_head))
        if args.ctc_head
        else args.model
    )
    vocab_path = _find_sidecar(sidecar_dir, "ctc_vocab.json", args.ctc_vocab)
    if not vocab_path:
        raise FileNotFoundError(
            f"No ctc_vocab.json under {sidecar_dir}; pass --ctc_vocab explicitly."
        )
    vocab = CTCVocabulary.load(vocab_path, tokenizer)

    config_path = _find_sidecar(sidecar_dir, "ctc_config.json")
    bottleneck = 512
    if config_path:
        with open(config_path, "r", encoding="utf-8") as handle:
            bottleneck = int(json.load(handle).get("bottleneck", bottleneck))

    head = attach_ctc_head(model, vocab.size, bottleneck=bottleneck)
    weights = load_file(args.ctc_head) if args.ctc_head else _load_ctc_weights(args.model)
    if "proj.weight" in weights:
        raise ValueError(
            "This checkpoint uses the legacy linear CTC head; retrain with the "
            "compact-vocabulary MLP head before running CTC-only decoding."
        )
    head.load_state_dict(weights)
    head.eval()

    rows = _load_rows(args.eval_jsonl, args.limit)
    if not rows:
        raise ValueError(f"No rows loaded from {args.eval_jsonl}")

    os.makedirs(args.output_dir, exist_ok=True)
    predictions_path = os.path.join(args.output_dir, "ctc_predictions.jsonl")
    records = []
    for index, row in enumerate(rows, start=1):
        wav, _ = librosa.load(row["audio"], sr=16000, mono=True)
        features = processor.feature_extractor(
            [wav],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        input_features = features["input_features"].to(args.device, dtype=dtype)
        feature_mask = features["attention_mask"].to(args.device)

        with torch.no_grad():
            hidden = model.thinker.get_audio_features(
                input_features,
                feature_attention_mask=feature_mask,
            )
            logits = head(hidden.float()).unsqueeze(0)
        token_ids = ctc_greedy_decode(
            logits, torch.tensor([hidden.size(0)], device=logits.device)
        )[0]
        hypothesis = vocab.decode(token_ids)
        reference = transcript_from_target(row.get("text", ""))

        records.append(
            {
                "audio": row["audio"],
                "reference": reference,
                "hypothesis": hypothesis,
                "reference_normalized": normalize_english(reference),
                "hypothesis_normalized": normalize_english(hypothesis),
            }
        )
        if index % 10 == 0 or index == len(rows):
            print(f"[ctc-eval] {index}/{len(rows)}")

    with open(predictions_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    references = [record["reference_normalized"] for record in records]
    hypotheses = [record["hypothesis_normalized"] for record in records]
    summary = {
        "model": os.path.abspath(args.model),
        "ctc_tokenizer": os.path.abspath(args.ctc_tokenizer),
        "ctc_vocab": os.path.abspath(vocab_path),
        "vocab_size": vocab.size,
        "eval_jsonl": os.path.abspath(args.eval_jsonl),
        "n": len(records),
        "wer": float(compute_wer(references, hypotheses)["wer"]),
        "cer": _corpus_cer(references, hypotheses),
        "exact": sum(ref == hyp for ref, hyp in zip(references, hypotheses)),
        "predictions": os.path.abspath(predictions_path),
    }
    summary_path = os.path.join(args.output_dir, "ctc_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()