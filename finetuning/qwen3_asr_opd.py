#!/usr/bin/env python3
# coding=utf-8
"""On-policy distillation (OPD) post-training for Qwen3-ASR.

Mechanism (Thinking Machines, "on-policy distillation"): a *student* (a
domain-finetuned checkpoint) samples its own ASR rollouts for each audio prompt;
a *teacher* (typically the base Qwen3-ASR, which retains general robustness)
scores those exact student trajectories token-by-token. The training signal is
the per-token reverse KL between the two distributions, used as a dense
advantage under a policy-gradient update — no reward, no reference transcript
enters the loss. On-policy means the student is corrected on the states it
actually visits, so a failure mode the student produces (e.g. repetition loops)
is penalized wherever it recurs, while regions where the student already matches
the teacher see little gradient and domain knowledge is preserved.

This is structurally the sibling of ``qwen3_asr_grpo.py`` (same single-GPU
rollout/logprob/freeze machinery). The differences:
  - advantage = -kl_coef * (student_logp - teacher_logp)  [dense, per token]
      instead of GRPO's group-normalized -WER reward;
  - the reference model is replaced by an explicit *teacher* (different weights);
  - teacher and student both reuse the STUDENT's detached audio features (the
    frozen encoder+aligner path) so the KL reflects only the LM, not audio drift.

By default the audio tower (encoder + aligner) is frozen and only the LLM + lm_head
are updated (``--freeze_modules`` overrides).

Usage (CLI args, like the GRPO script — not a YAML pipeline config):
    python finetuning/qwen3_asr_opd.py \
        --student_path outputs/<grpo-or-sft>/checkpoint-1500 \
        --teacher_path /root/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/<hash> \
        --train_file data/gigaspeech/train.jsonl \
        --output_dir outputs/<opd_run> \
        --lr 1e-6 --num_generations 4 --temperature 1.0 --max_steps 1500
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import GenerationConfig

# Allow running directly as a script (``python finetuning/qwen3_asr_opd.py``) from
# the repo root: put this file's directory on sys.path so sibling imports resolve.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwen3_asr_grpo import (  # noqa: E402  (reuse the GRPO skeleton)
    JsonlDataset,
    RolloutBatch,
    _collate_rows,
    _prepare_prefix_inputs,
    _resolve_frozen_parts,
    completion_logps,
    extract_reference_text,
    rollout_groups,
    seed_everything,
    word_error_rate,
)
from module_freeze import (  # noqa: E402
    PARTS,
    count_part_parameters,
    set_part_freeze,
)
from qwen_asr import Qwen3ASRModel  # noqa: E402


def _save_opd_checkpoint(
    model: Any,
    processor: Any,
    optimizer: AdamW,
    output_dir: str,
    global_step: int,
    epoch: int,
) -> str:
    """Save student weights + processor + trainer state (OPD-named state file)."""
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    generation_config = model.generation_config
    if not getattr(generation_config, "do_sample", True):
        generation_config.do_sample = True  # downstream re-sampling should sample
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    processor.save_pretrained(checkpoint_dir)
    torch.save(
        {"optimizer": optimizer.state_dict(), "global_step": global_step, "epoch": epoch},
        os.path.join(checkpoint_dir, "opd_trainer_state.pt"),
    )
    return checkpoint_dir


def _discounted_future_sum(values: torch.Tensor, gamma: float) -> torch.Tensor:
    """Reverse cumulative discounted sum along the token axis (dim=1).

    Mirrors tinker_cookbook's ``discounted_future_sum_vectorized``: out[t] =
    sum_{k>=t} gamma^(k-t) * values[k]. gamma=0 is a no-op.
    """
    if gamma == 0.0:
        return values
    out = torch.zeros_like(values)
    carry = torch.zeros_like(out[:, :1])
    for t in range(out.shape[1] - 1, -1, -1):
        carry = values[:, t : t + 1] + gamma * carry
        out[:, t : t + 1] = carry
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="On-policy distillation for Qwen3-ASR (teacher -> student via per-token reverse KL)."
    )
    p.add_argument("--student_path", required=True, help="Domain-finetuned checkpoint to recover robustness in.")
    p.add_argument("--teacher_path", required=True, help="Teacher checkpoint (typically the base Qwen3-ASR).")
    p.add_argument("--train_file", required=True, help="JSONL with audio (+optional text for WER logging).")
    p.add_argument("--output_dir", required=True)
    # Rollout sampling
    p.add_argument("--batch_size", type=int, default=1, help="Audio prompts per rollout batch.")
    p.add_argument("--num_generations", type=int, default=4, help="Student rollouts per audio (>=1).")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=512)
    # Distillation objective
    p.add_argument("--kl_coef", type=float, default=1.0, help="Coefficient on the per-token reverse-KL advantage.")
    p.add_argument("--kl_discount", type=float, default=0.0, help="Discount for cumulative future KL (0 = per-token).")
    p.add_argument(
        "--teacher_own_audio", type=int, default=0, choices=(0, 1),
        help="1 = teacher scores rollouts with its OWN audio tower (recompute features via the "
             "live_audio path). REQUIRED when the student's audio_tower/aligner differ from the "
             "teacher's (e.g. student was SFT'd with audio trainable). 0 (default) = teacher reuses "
             "the student's detached audio features (valid only when both share an identical frozen "
             "audio tower).",
    )
    # Optimizer
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_acc", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=1)
    # Freezing
    p.add_argument(
        "--freeze_modules", default=None,
        help="Comma-separated parts to freeze: encoder,aligner,llm. Unset = encoder,aligner (LLM-only).",
    )
    # Loop control / logging
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--log_steps", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=0, help="0 runs all epochs.")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run", type=int, default=0, choices=(0, 1))
    return p.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch_size must be positive.")
    if args.num_generations < 1:
        raise ValueError("--num_generations must be at least 1 (GRPO-style group advantage is not used here).")
    if args.grad_acc < 1:
        raise ValueError("--grad_acc must be positive.")
    if args.max_new_tokens < 1:
        raise ValueError("--max_new_tokens must be positive.")
    if args.temperature <= 0:
        raise ValueError("--temperature must be greater than zero when sampling.")
    if not 0.0 <= args.kl_discount < 1.0:
        raise ValueError("--kl_discount must be in [0, 1).")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    seed_everything(args.seed)
    dataset = JsonlDataset(args.train_file, max_samples=args.max_samples)
    print(f"[data] rows={len(dataset)} train_file={args.train_file}")

    frozen_parts = _resolve_frozen_parts(args)
    trainable_parts = [part for part in PARTS if part not in frozen_parts]
    print(f"[freeze] frozen={','.join(frozen_parts) or 'none'}; trainable={','.join(trainable_parts) or 'none'}")

    if args.dry_run:
        print("[dry-run] student=%s teacher=%s" % (args.student_path, args.teacher_path))
        print("[dry-run] objective=per-token reverse-KL(student||teacher); kl_coef=%g" % args.kl_coef)
        print("[dry-run] teacher_own_audio=%d (student_audio_trainable=%s)"
              % (args.teacher_own_audio, not ("encoder" in frozen_parts and "aligner" in frozen_parts)))
        print("[dry-run] first_audio=%s" % dataset[0]["audio"])
        return

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-ASR OPD requires CUDA.")
    device = torch.device("cuda:0")
    use_bf16 = torch.cuda.get_device_capability(device)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    # Student (policy) — the trainable model, started from the finetuned checkpoint.
    print(f"[load] student={args.student_path}")
    student_wrapper = Qwen3ASRModel.from_pretrained(args.student_path, dtype=dtype, device_map=None)
    student_model = student_wrapper.model.to(device)
    processor = student_wrapper.processor
    set_part_freeze(student_model, frozen_parts)
    part_counts = count_part_parameters(student_model)
    for part in PARTS:
        total, trainable = part_counts[part]
        print(f"[freeze] {part}: total={total:,} trainable={trainable:,}")
    audio_trainable = "encoder" not in frozen_parts or "aligner" not in frozen_parts
    if audio_trainable:
        # OPD assumes a FIXED shared audio evidence across teacher/student; trainable
        # audio would make the two models see different features. Keep the default.
        print("[warn] audio part is trainable; teacher will reuse STUDENT audio features "
              "so the KL still isolates the LM, but a base-aligned audio path is recommended.")

    # Teacher — frozen, different weights (typically the base model).
    print(f"[load] teacher={args.teacher_path}")
    teacher_wrapper = Qwen3ASRModel.from_pretrained(args.teacher_path, dtype=dtype, device_map=None)
    teacher_model = teacher_wrapper.model.to(device).eval()
    teacher_model.requires_grad_(False)
    teacher_model.thinker.audio_tower.eval()

    optimizer = AdamW(
        [param for param in student_model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate_rows,
    )
    eos_token_ids = student_model.generation_config.eos_token_id or [151645, 151643]
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=eos_token_ids,
        pad_token_id=student_model.generation_config.pad_token_id or eos_token_ids[-1],
        return_dict_in_generate=False,
    )

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    should_stop = False
    for epoch in range(args.epochs):
        for rows in loader:
            student_model.eval()
            prefix_inputs = _prepare_prefix_inputs(rows, processor, device, dtype, args.sr)
            rollout = rollout_groups(
                student_model.thinker,
                processor,
                prefix_inputs,
                args.num_generations,
                generation_config,
            )

            # Log-only WER (does not enter the loss). GigaSpeech/train jsonl has 'text'.
            references = [extract_reference_text(str(row["text"])) for row in rows for _ in range(args.num_generations)]
            try:
                roll_wer = sum(word_error_rate(ref, comp) for ref, comp in zip(references, rollout.raw_completions))
                roll_wer = roll_wer / max(len(references), 1)
            except Exception:
                roll_wer = float("nan")

            # Teacher scores the student's exact rollouts. With --teacher_own_audio=1 the
            # teacher runs its OWN audio tower on the raw input (live_audio path) — required
            # when the student's audio_tower/aligner differ from the teacher's (audio was
            # trainable during the student's SFT). Otherwise (0) both reuse the student's
            # detached features (valid only when the frozen audio towers are identical).
            with torch.no_grad():
                if args.teacher_own_audio:
                    teacher_audio = (prefix_inputs["input_features"], prefix_inputs["feature_attention_mask"])
                else:
                    teacher_audio = None
                teacher_logps = completion_logps(teacher_model.thinker, rollout, teacher_audio).detach()

            student_model.train()
            if not audio_trainable:
                student_model.thinker.audio_tower.eval()
            student_logps = completion_logps(student_model.thinker, rollout)

            mask = rollout.completion_mask.to(dtype=student_logps.dtype)
            # Per-token reverse KL estimate on sampled tokens: kl = student_logp - teacher_logp.
            reverse_kl = (student_logps.detach() - teacher_logps)
            if args.kl_discount > 0.0:
                reverse_kl = _discounted_future_sum(reverse_kl, args.kl_discount)
            advantage = -args.kl_coef * reverse_kl  # dense, detached (a constant)
            # Policy gradient: maximize advantage * log pi_student  -> minimize reverse KL.
            per_token_loss = -advantage * student_logps
            sequence_loss = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            loss = sequence_loss.mean()
            (loss / args.grad_acc).backward()

            global_step += 1
            if global_step % args.grad_acc == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in student_model.parameters() if p.requires_grad],
                    args.max_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if global_step % args.log_steps == 0:
                mean_len = rollout.completion_mask.sum(dim=1).float().mean().item()
                mean_kl = ((student_logps.detach() - teacher_logps) * mask).sum().item() / mask.sum().item()
                print(
                    "[step %d] loss=%.6f rev_kl=%.5f roll_wer=%.4f len=%.1f"
                    % (global_step, loss.item(), mean_kl, roll_wer, mean_len),
                    flush=True,
                )

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                student_model.eval()
                path = _save_opd_checkpoint(student_model, processor, optimizer, args.output_dir, global_step, epoch)
                print(f"[save] {path}")
                student_model.train()

            if args.max_steps > 0 and global_step >= args.max_steps:
                should_stop = True
                break
        if should_stop:
            break

    if global_step % args.grad_acc:
        torch.nn.utils.clip_grad_norm_(
            [p for p in student_model.parameters() if p.requires_grad], args.max_grad_norm
        )
        optimizer.step()
    final_path = _save_opd_checkpoint(student_model, processor, optimizer, args.output_dir, global_step, epoch)
    print(f"[done] final_checkpoint={final_path}")


if __name__ == "__main__":
    main()
