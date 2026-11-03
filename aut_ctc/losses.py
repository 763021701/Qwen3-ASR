"""CTC losses (DESIGN.md §9.1): 0.7 * CTC(final) + 0.3 * CTC(inter), fp32, normalized by target tokens."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def ctc_loss(logits, lens, targets, target_lens, blank: int = 0):
    """targets: (B, S) right-padded. log_softmax and CTC in fp32; summed then divided by the number of
    target tokens, which keeps mixed zh/en batches stable."""
    valid = torch.arange(targets.size(1), device=targets.device)[None] < target_lens[:, None]
    if (targets[valid] == blank).any():  # e.g. reusing SenseVoice ids, where <unk> (0) leaks into targets
        raise ValueError("blank id found in CTC targets")
    logp = logits.float().log_softmax(-1).transpose(0, 1)
    loss = F.ctc_loss(logp, targets, lens, target_lens, blank=blank, reduction="sum", zero_infinity=True)
    return loss / target_lens.sum().clamp(min=1)


def total_loss(out, targets, target_lens, inter_weight: float = 0.3):
    """Returns (loss, final CTC, inter CTC or None)."""
    final = ctc_loss(out["logits"], out["lens"], targets, target_lens)
    if out["inter_logits"] is None:
        return final, final, None
    inter = ctc_loss(out["inter_logits"], out["inter_lens"], targets, target_lens)
    return (1 - inter_weight) * final + inter_weight * inter, final, inter


def ctc_feasible(token_ids: list[int], num_frames: int) -> bool:
    """CTC feasibility: T >= U + adjacent repeats (a blank must separate repeated tokens)."""
    repeats = sum(a == b for a, b in zip(token_ids, token_ids[1:]))
    return len(token_ids) + repeats <= num_frames
