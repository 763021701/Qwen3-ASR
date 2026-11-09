#!/usr/bin/env python3
"""Pretrain the auxiliary CTC head on frozen Qwen3-ASR audio-tower features.

The joint SFT run can collapse the CTC head to blank + a few high-frequency
tokens before the encoder has a chance to adapt.  This tool trains only the
head against a fixed AuT, which is much cheaper than a full SFT step and gives
the joint run a non-degenerate starting point:

    python tools/pretrain_ctc_head.py \
        --model <base-model> \
        --train_jsonl data/poc_train_radiology_mix_m1/train.jsonl \
        --ctc_tokenizer <sensevoice-bpe> \
        --output_dir outputs/ctc_head_pretrain

Then pass ``--ctc_head_init outputs/ctc_head_pretrain/ctc_head.safetensors``
to ``finetuning/qwen3_asr_sft.py``.
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
from safetensors.torch import save_file

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from finetuning.ctc_auxiliary import (
    CTCVocabulary,
    SenseVoiceCTCTokenizer,
    attach_ctc_head,
    compute_ctc_loss,
)
from qwen_asr import Qwen3ASRModel


def _load_rows(path: str, max_samples: int) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples and len(rows) >= max_samples:
                break
    return rows


def _audio_features(
    model,
    processor,
    wav,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    features = processor.feature_extractor(
        [wav],
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    input_features = features["input_features"].to(device, dtype=dtype)
    feature_mask = features["attention_mask"].to(device)
    length = int(feature_mask.sum().item())
    feature = input_features[0, :, :length]
    with torch.no_grad():
        output = model.thinker.audio_tower(
            feature,
            feature_lens=torch.tensor([length], device=device),
        )
    return output.last_hidden_state.detach()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--ctc_tokenizer", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ctc_vocab", default="", help="Reuse an existing ctc_vocab.json.")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--coverage_weight", type=float, default=0.01)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)

    wrapper = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=None,
        max_new_tokens=512,
    )
    model = wrapper.model.to(args.device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    processor = wrapper.processor
    tokenizer = SenseVoiceCTCTokenizer(args.ctc_tokenizer)

    os.makedirs(args.output_dir, exist_ok=True)
    vocab_path = os.path.join(args.output_dir, "ctc_vocab.json")
    if args.ctc_vocab:
        vocab = CTCVocabulary.load(args.ctc_vocab, tokenizer)
    elif os.path.isfile(vocab_path):
        vocab = CTCVocabulary.load(vocab_path, tokenizer)
    else:
        rows = _load_rows(args.train_jsonl, 0)
        print("[ctc-pretrain] building vocabulary from %d transcripts" % len(rows))
        vocab = CTCVocabulary.from_texts(tokenizer, [row["text"] for row in rows])
        vocab.save(vocab_path)
    print(
        "[ctc-pretrain] vocabulary %d tokens -> %d classes"
        % (len(vocab.source_ids), vocab.size)
    )

    head = attach_ctc_head(model, vocab.size)
    head.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)

    rows = _load_rows(args.train_jsonl, args.max_samples)
    if not rows:
        raise ValueError(f"No rows loaded from {args.train_jsonl}")
    print("[ctc-pretrain] rows=%d steps=%d bs=%d" % (len(rows), args.steps, args.batch_size))

    step = 0
    cursor = 0
    order = list(range(len(rows)))
    while step < args.steps:
        if cursor == 0:
            random.shuffle(order)
        batch_rows = [rows[i] for i in order[cursor : cursor + args.batch_size]]
        cursor += args.batch_size
        if len(batch_rows) < args.batch_size:
            cursor = 0
            continue

        hidden_states = []
        target_sequences = []
        for row in batch_rows:
            wav, _ = librosa.load(row["audio"], sr=16000, mono=True)
            hidden_states.append(_audio_features(model, processor, wav, args.device, dtype))
            target_sequences.append(vocab.encode(row["text"]))

        max_len = max([len(seq) for seq in target_sequences] + [1])
        targets = torch.zeros(
            (len(target_sequences), max_len), dtype=torch.long, device=args.device
        )
        for i, seq in enumerate(target_sequences):
            if seq:
                targets[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=args.device)
        target_lengths = torch.tensor(
            [len(seq) for seq in target_sequences], dtype=torch.long, device=args.device
        )

        loss = compute_ctc_loss(
            hidden_states,
            head,
            targets,
            target_lengths,
            coverage_weight=args.coverage_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        step += 1
        if step % args.log_steps == 0:
            print(
                "[ctc-pretrain] step=%d/%d loss=%.4f"
                % (step, args.steps, float(loss.detach()))
            )

    head_path = os.path.join(args.output_dir, "ctc_head.safetensors")
    save_file(
        {name: tensor.detach().cpu() for name, tensor in head.state_dict().items()},
        head_path,
    )
    config = {
        "tokenizer": os.path.abspath(args.ctc_tokenizer),
        "vocab_file": os.path.abspath(vocab_path),
        "vocab_size": vocab.size,
        "blank_id": 0,
        "hidden_size": head.hidden_size,
        "bottleneck": head.bottleneck,
        "coverage_weight": args.coverage_weight,
        "steps": args.steps,
        "lr": args.lr,
        "source_model": os.path.abspath(args.model),
        "target_normalization": "acoustic-v1",
    }
    with open(
        os.path.join(args.output_dir, "ctc_config.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print("[ctc-pretrain] wrote %s" % head_path)


if __name__ == "__main__":
    main()