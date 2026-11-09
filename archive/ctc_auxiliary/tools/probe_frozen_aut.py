#!/usr/bin/env python3
"""Frozen-AuT CTC probe: does CTC joint training improve AuT decodability?

For each model, freeze the AuT, attach an identically initialized fresh head,
train the head on a fixed split with the same budget, and compare Train/Val CER.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List

import librosa
import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from finetuning.ctc_auxiliary import (
    CTCHead,
    CTCVocabulary,
    SenseVoiceCTCTokenizer,
    attach_ctc_head,
    compute_ctc_loss,
    normalize_ctc_text,
)
from qwen_asr import Qwen3ASRModel


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
    edits = total = 0
    for reference, hypothesis in zip(references, hypotheses):
        ref = reference.replace(" ", "")
        hyp = hypothesis.replace(" ", "")
        edits += _edit_distance(ref, hyp)
        total += len(ref)
    return edits / total if total else 0.0


def _load_rows(path: str) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def _features(model, processor, wav, device, dtype):
    feats = processor.feature_extractor(
        [wav], sampling_rate=16000, return_tensors="pt", padding=True, return_attention_mask=True
    )
    input_features = feats["input_features"].to(device, dtype=dtype)
    mask = feats["attention_mask"].to(device)
    length = int(mask.sum().item())
    with torch.no_grad():
        output = model.thinker.audio_tower(
            input_features[0, :, :length],
            feature_lens=torch.tensor([length], device=device),
        )
    return output.last_hidden_state.detach()


def _cache(model, processor, rows, device, dtype):
    hidden = []
    for row in rows:
        wav, _ = librosa.load(row["audio"], sr=16000, mono=True)
        hidden.append(_features(model, processor, wav, device, dtype))
    return hidden


def _batch_tensors(sequences, device):
    max_len = max([len(seq) for seq in sequences] + [1])
    targets = torch.zeros((len(sequences), max_len), dtype=torch.long, device=device)
    for i, seq in enumerate(sequences):
        if seq:
            targets[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long, device=device)
    return targets, lengths


def _evaluate(head, vocab, hidden_list, sequences, references, device):
    hypotheses = []
    pure = []
    coverage = []
    blanks = []
    for hidden, seq in zip(hidden_list, sequences):
        with torch.no_grad():
            logits = head(hidden.to(head.proj[1].weight.dtype))
            log_probs = logits.float().log_softmax(dim=-1).unsqueeze(1).contiguous()
            target = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
            t = torch.tensor([hidden.size(0)], device=device)
            u = torch.tensor([len(seq)], device=device)
            pure.append(
                float(
                    F.ctc_loss(
                        log_probs, target, t, u, blank=0, reduction="none", zero_infinity=True
                    )
                )
            )
            probs = log_probs[:, 0, :].exp()
            coverage.append(max(0.0, len(seq) - float((1.0 - probs[:, 0]).sum())) ** 2)
            blanks.append(float((logits.argmax(-1) == 0).float().mean()))
            prediction = logits.argmax(dim=-1)
            keep = torch.ones_like(prediction, dtype=torch.bool)
            keep[1:] = prediction[1:] != prediction[:-1]
            hypotheses.append(vocab.decode(prediction[keep].tolist()))
    return {
        "cer": _corpus_cer(references, hypotheses),
        "pure_ctc": sum(pure) / max(len(pure), 1),
        "coverage_penalty": sum(coverage) / max(len(coverage), 1),
        "blank_ratio": sum(blanks) / max(len(blanks), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--holdout_jsonl", default="")
    parser.add_argument("--ctc_tokenizer", required=True)
    parser.add_argument("--ctc_vocab", required=True)
    parser.add_argument("--train_samples", type=int, default=2000)
    parser.add_argument("--val_samples", type=int, default=512)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--coverage_weight", type=float, default=0.01)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer = SenseVoiceCTCTokenizer(args.ctc_tokenizer)
    vocab = CTCVocabulary.load(args.ctc_vocab, tokenizer)
    all_rows = _load_rows(args.train_jsonl)
    random.Random(args.seed).shuffle(all_rows)
    train_rows = all_rows[: args.train_samples]
    val_rows = all_rows[args.train_samples : args.train_samples + args.val_samples]
    holdout_rows = _load_rows(args.holdout_jsonl) if args.holdout_jsonl else []

    train_sequences = [vocab.encode(row["text"]) for row in train_rows]
    train_refs = [normalize_ctc_text(row["text"]) for row in train_rows]
    val_sequences = [vocab.encode(row["text"]) for row in val_rows]
    val_refs = [normalize_ctc_text(row["text"]) for row in val_rows]
    holdout_sequences = [vocab.encode(row["text"]) for row in holdout_rows]
    holdout_refs = [normalize_ctc_text(row["text"]) for row in holdout_rows]

    results: Dict[str, Any] = {"splits": {
        "train": len(train_rows), "val": len(val_rows), "holdout": len(holdout_rows)
    }, "models": {}}

    for spec in args.model:
        name, _, path = spec.partition("=")
        wrapper = Qwen3ASRModel.from_pretrained(
            path, dtype=torch.bfloat16, device_map=None, max_new_tokens=512
        )
        model = wrapper.model.to(args.device).eval()
        processor = wrapper.processor
        for parameter in model.parameters():
            parameter.requires_grad = False

        hidden_size = model.thinker.audio_tower.proj2.out_features
        torch.manual_seed(args.seed)
        ref_head = CTCHead(hidden_size, vocab.size)
        head_state = ref_head.state_dict()
        head = attach_ctc_head(model, vocab.size)
        head.load_state_dict(head_state)
        for parameter in head.parameters():
            parameter.requires_grad = True

        train_hidden = _cache(model, processor, train_rows, args.device, torch.bfloat16)
        val_hidden = _cache(model, processor, val_rows, args.device, torch.bfloat16)
        holdout_hidden = (
            _cache(model, processor, holdout_rows, args.device, torch.bfloat16)
            if holdout_rows
            else []
        )

        optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
        rng = random.Random(args.seed)
        order = list(range(len(train_hidden)))
        history = []
        for step in range(1, args.steps + 1):
            if not order:
                order = list(range(len(train_hidden)))
                rng.shuffle(order)
            take = order[: args.batch_size]
            order = order[args.batch_size :]
            batch_targets, batch_lengths = _batch_tensors(
                [train_sequences[i] for i in take], args.device
            )
            loss = compute_ctc_loss(
                [train_hidden[i] for i in take],
                head,
                batch_targets,
                batch_lengths,
                coverage_weight=args.coverage_weight,
                coverage_detach=True,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()

            if step % args.eval_every == 0 or step == args.steps:
                train_metrics = _evaluate(
                    head, vocab, train_hidden, train_sequences, train_refs, args.device
                )
                val_metrics = _evaluate(
                    head, vocab, val_hidden, val_sequences, val_refs, args.device
                )
                holdout_metrics = (
                    _evaluate(
                        head, vocab, holdout_hidden, holdout_sequences, holdout_refs, args.device
                    )
                    if holdout_rows
                    else {}
                )
                record = {
                    "step": step,
                    "train_cer": train_metrics["cer"],
                    "val_cer": val_metrics["cer"],
                    "holdout_cer": holdout_metrics.get("cer"),
                    "val_pure_ctc": val_metrics["pure_ctc"],
                    "val_blank": val_metrics["blank_ratio"],
                    "val_coverage_penalty": val_metrics["coverage_penalty"],
                }
                history.append(record)
                print(
                    "[%s] step=%d train_cer=%.4f val_cer=%.4f holdout_cer=%s blank=%.3f"
                    % (
                        name,
                        step,
                        record["train_cer"],
                        record["val_cer"],
                        "%.4f" % record["holdout_cer"] if record["holdout_cer"] is not None else "-",
                        record["val_blank"],
                    )
                )
        best_val = min(record["val_cer"] for record in history)
        results["models"][name] = {
            "path": os.path.abspath(path),
            "history": history,
            "best_val_cer": best_val,
            "final_val_cer": history[-1]["val_cer"],
            "final_train_cer": history[-1]["train_cer"],
            "final_holdout_cer": history[-1]["holdout_cer"],
        }
        print("[%s] best_val_cer=%.4f final_val=%.4f" % (name, best_val, history[-1]["val_cer"]))
        del model, head, train_hidden, val_hidden, holdout_hidden
        torch.cuda.empty_cache()

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print("wrote", args.output)


if __name__ == "__main__":
    main()