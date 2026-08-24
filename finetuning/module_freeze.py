# coding=utf-8
"""Per-part parameter freezing for the Qwen3-ASR finetuning scripts.

The Qwen3-ASR thinker parameters are split into three parts:

- ``encoder``: ``thinker.audio_tower`` minus the aligner projections
  (conv2d1/2/3, layers, ln_post, positional_embedding)
- ``aligner``: ``thinker.audio_tower.{conv_out,proj1,proj2}``
  (the audio->LLM projection layers)
- ``llm``: ``thinker.model`` + ``thinker.lm_head``
"""

from typing import Dict, Iterable, Tuple

ALIGNER_SUBMODULES = ("conv_out", "proj1", "proj2")
PARTS = ("encoder", "aligner", "llm")


def parse_parts(spec) -> Tuple[str, ...]:
    """Parse a comma-separated part list (e.g. ``"encoder,aligner"``).

    ``None`` or empty input yields ``()``. Unknown names raise ``ValueError``.
    """
    if not spec:
        return ()
    parts = []
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        if token not in PARTS:
            raise ValueError(f"Unknown freeze part {token!r}. Choices: {', '.join(PARTS)}")
        if token not in parts:
            parts.append(token)
    return tuple(parts)


def _audio_child_part(name: str) -> str:
    return "aligner" if name in ALIGNER_SUBMODULES else "encoder"


def set_part_freeze(model, frozen_parts: Iterable[str]) -> None:
    """Set requires_grad (and eval mode for frozen parts) per thinker part.

    Idempotent: parts not in ``frozen_parts`` are explicitly set trainable.
    """
    thinker = model.thinker
    if (
        not hasattr(thinker, "audio_tower")
        or not hasattr(thinker, "model")
        or not hasattr(thinker, "lm_head")
    ):
        raise RuntimeError("Expected a Qwen3-ASR thinker with audio_tower, model and lm_head.")
    frozen = set(frozen_parts)
    for name, module in thinker.audio_tower.named_children():
        trainable = _audio_child_part(name) not in frozen
        module.requires_grad_(trainable)
        if not trainable:
            module.eval()
    llm_trainable = "llm" not in frozen
    thinker.model.requires_grad_(llm_trainable)
    thinker.lm_head.requires_grad_(llm_trainable)
    if not llm_trainable:
        thinker.model.eval()
        thinker.lm_head.eval()


def count_part_parameters(model) -> Dict[str, Tuple[int, int]]:
    """Return ``{part: (total_params, trainable_params)}`` for the three parts."""
    thinker = model.thinker
    counts: Dict[str, list] = {}
    for name, module in thinker.audio_tower.named_children():
        entry = counts.setdefault(_audio_child_part(name), [0, 0])
        for p in module.parameters():
            entry[0] += p.numel()
            if p.requires_grad:
                entry[1] += p.numel()
    for module in (thinker.model, thinker.lm_head):
        entry = counts.setdefault("llm", [0, 0])
        for p in module.parameters():
            entry[0] += p.numel()
            if p.requires_grad:
                entry[1] += p.numel()
    return {part: (total, trainable) for part, (total, trainable) in counts.items()}
