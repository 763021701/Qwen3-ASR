#!/usr/bin/env python3
"""Small-sample CTC overfit test: frozen AuT vs AuT+head, fresh head."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Tuple

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


def _features(model, processor, wav, device: str, dtype: torch.dtype, grad: bool):
    feats = processor.feature_extractor(
        [wav], sampling_rate=16000, return_tensors="pt", padding=True, return_attention_mask=True
    )
    input_features = feats["input_features"].to(device, dtype=dtype)
    mask = feats["attention_mask"].to(device)
    length = int(mask.sum().item())
    if grad:
        output = model.thinker.audio_tower(
            input_features[0, :, :length],
            feature_lens=torch.tensor([length], device=device),
        )
    else:
        with torch.no_grad():
            output = model.thinker.audio_tower(
                input_features[0, :, :length],
                feature_lens=torch.tensor([length], device=device),
            )
    return output.last_hidden_state


def _batch_tensors(sequences: List[List[int]], device: str):
    max_len = max([len(seq) for seq in sequences] + [1])
    targets = torch.zeros((len(sequences), max_len), dtype=torch.long, device=device)
    for i, seq in enumerate(sequences):
        if seq:
            targets[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long, device=device)
    return targets, lengths


def _evaluate(model, head, vocab, hidden_list, sequences, references, device) -> Dict[str, float]:
    hypotheses = []
    blanks = []
    pure_losses = []
    coverage_penalties = []
    for hidden, seq in zip(hidden_list, sequences):
        with torch.no_grad():
            logits = head(hidden.to(head.proj[1].weight.dtype).float())
            log_probs = logits.float().log_softmax(dim=-1).unsqueeze(1).contiguous()
            target = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
            t = torch.tensor([hidden.size(0)], device=device)
            u = torch.tensor([len(seq)], device=device)
            pure_losses.append(
                float(
                    F.ctc_loss(
                        log_probs, target, t, u, blank=0, reduction="none", zero_infinity=True
                    )
                )
            )
            probs = log_probs[:, 0, :].exp()
            coverage = float((1.0 - probs[:, 0]).sum())
            coverage_penalties.append(max(0.0, len(seq) - coverage) ** 2)
            blanks.append(float((logits.argmax(-1) == 0).float().mean()))
            prediction = logits.argmax(dim=-1)
            keep = torch.ones_like(prediction, dtype=torch.bool)
            keep[1:] = prediction[1:] != prediction[:-1]
            hypotheses.append(vocab.decode(prediction[keep].tolist()))
    return {
        "train_cer": _corpus_cer(references, hypotheses),
        "pure_ctc": sum(pure_losses) / max(len(pure_losses), 1),
        "coverage_penalty": sum(coverage_penalties) / max(len(coverage_penalties), 1),
        "blank_ratio": sum(blanks) / max(len(blanks), 1),
    }




def _train_head(model, head, hidden_list, sequences, references, vocab, args, device) -> List[Dict[str, float]]:
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr_head)
    history = []
    for step in range(1, args.head_steps + 1):
        idx = random.sample(range(len(hidden_list)), min(args.batch_size, len(hidden_list)))
        batch_hidden = [hidden_list[i] for i in idx]
        batch_targets, batch_lengths = _batch_tensors([sequences[i] for i in idx], device)
        loss = compute_ctc_loss(
            batch_hidden,
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
        if step % args.eval_every == 0 or step == args.head_steps:
            metrics = _evaluate(model, head, vocab, hidden_list, sequences, references, device)
            metrics["step"] = step
            history.append(metrics)
            print("[head] step=%d cer=%.4f pure_ctc=%.3f coverage=%.3f blank=%.3f"
                  % (step, metrics["train_cer"], metrics["pure_ctc"],
                     metrics["coverage_penalty"], metrics["blank_ratio"]))
    return history


def _train_full(model, head, rows, sequences, references, vocab, processor, args, device, dtype) -> List[Dict[str, float]]:
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.thinker.audio_tower.parameters() if p.requires_grad], "lr": args.lr_aut},
            {"params": list(head.parameters()), "lr": args.lr_head},
        ]
    )
    history = []
    for step in range(1, args.full_steps + 1):
        idx = random.sample(range(len(rows)), min(args.batch_size, len(rows)))
        batch_hidden = []
        batch_sequences = []
        for i in idx:
            row = rows[i]
            wav, _ = librosa.load(row["audio"], sr=16000, mono=True)
            batch_hidden.append(_features(model, processor, wav, device, dtype, grad=True))
            batch_sequences.append(sequences[i])
        targets, lengths = _batch_tensors(batch_sequences, device)
        loss = compute_ctc_loss(
            batch_hidden,
            head,
            targets,
            lengths,
            coverage_weight=args.coverage_weight,
            coverage_detach=True,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.thinker.audio_tower.parameters()] + list(head.parameters()), 1.0
        )
        optimizer.step()
        if step % args.eval_every == 0 or step == args.full_steps:
            eval_hidden = []
            for row in rows:
                wav, _ = librosa.load(row["audio"], sr=16000, mono=True)
                eval_hidden.append(_features(model, processor, wav, device, dtype, grad=False))
            metrics = _evaluate(model, head, vocab, eval_hidden, sequences, references, device)
            metrics["step"] = step
            history.append(metrics)
            print("[full] step=%d cer=%.4f pure_ctc=%.3f coverage=%.3f blank=%.3f"
                  % (step, metrics["train_cer"], metrics["pure_ctc"],
                     metrics["coverage_penalty"], metrics["blank_ratio"]))
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ctc_tokenizer", required=True)
    parser.add_argument("--ctc_vocab", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--modes", default="head,full")
    parser.add_argument("--head_steps", type=int, default=300)
    parser.add_argument("--full_steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_aut", type=float, default=2e-5)
    parser.add_argument("--coverage_weight", type=float, default=0.01)
    parser.add_argument("--eval_every", type=int, default=25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = _load_rows(args.samples)
    tokenizer = SenseVoiceCTCTokenizer(args.ctc_tokenizer)
    vocab = CTCVocabulary.load(args.ctc_vocab, tokenizer)
    wrapper = Qwen3ASRModel.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=None, max_new_tokens=512
    )
    model = wrapper.model.to(args.device)
    processor = wrapper.processor
    dtype = torch.bfloat16

    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()

    results: Dict[str, Any] = {"model": os.path.abspath(args.model), "samples": len(rows), "modes": {}}
    sequences = [vocab.encode(row["text"]) for row in rows]
    references = [normalize_ctc_text(row["text"]) for row in rows]

    if "head" in args.modes.split(","):
        head = attach_ctc_head(model, vocab.size)
        for parameter in head.parameters():
            parameter.requires_grad = True
        hidden_list = []
        for row in rows:
            wav, _ = librosa.load(row["audio"], sr=16000, mono=True)
            hidden_list.append(_features(model, processor, wav, args.device, dtype, grad=False))
        results["modes"]["head"] = _train_head(
            model, head, hidden_list, sequences, references, vocab, args, args.device
        )

    if "full" in args.modes.split(","):
        head = attach_ctc_head(model, vocab.size)
        for parameter in head.parameters():
            parameter.requires_grad = True
        for parameter in model.thinker.audio_tower.parameters():
            parameter.requires_grad = True
        model.train()
        results["modes"]["full"] = _train_full(
            model, head, rows, sequences, references, vocab, processor, args, args.device, dtype
        )

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print("wrote", args.output)


if __name__ == "__main__":
    main()