"""CTC greedy decoding with AuT frame timestamps (DESIGN.md §10.1)."""
from __future__ import annotations

import torch

from .frontend import aut_frame_times


@torch.no_grad()
def ctc_greedy(logits, lens, blank: int = 0, upsample: int = 1, num_queries: int = 0):
    """Per utterance [(token_id, time_s or None), ...].

    Times are the non-uniform AuT frame centres of each token's first frame (CTC spikes usually lag the true
    onset; use Qwen3-ForcedAligner for precise times). Tags emitted at query positions get time None.
    """
    best = logits.argmax(-1)
    results = []
    for b in range(best.size(0)):
        n = int(lens[b])
        seq = best[b, :n]
        n_audio = n // upsample - num_queries
        times = [None] * (num_queries * upsample) + aut_frame_times(n_audio, upsample).tolist()
        first = torch.ones(1, dtype=torch.bool, device=seq.device)
        keep = (seq != blank) & torch.cat([first, seq[1:] != seq[:-1]])
        results.append([(int(seq[k]), times[k]) for k in keep.nonzero().flatten().tolist()])
    return results


def hyps_to_text(tokenizer, hyps) -> list[str]:
    return [tokenizer.decode([tid for tid, _ in hyp]) for hyp in hyps]
