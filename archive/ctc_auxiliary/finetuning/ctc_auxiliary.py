# coding=utf-8
"""Auxiliary CTC head on the Qwen3-ASR audio tower.

The head reuses the SenseVoice tokenizer (multi-lingual BPE, 25,055 pieces)
so that every audio frame is supervised against the actual transcript.  The
CTC loss is added to the usual LLM cross-entropy loss during fine-tuning; the
LLM still produces the final transcription.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import Counter
from typing import List, Optional, Sequence

import torch
from torch import nn

_ASR_TEXT_TAG = "<asr_text>"
_HYPHEN_RE = re.compile(r"[-‐‑‒–—―−]")
_NON_DECIMAL_PERIOD_RE = re.compile(r"(?<!\d)\.|\.(?!\d)")
_WS_RE = re.compile(r"\s+")


def transcript_from_target(text: str) -> str:
    """Return the transcript part of a ``language X<asr_text>...`` target."""
    value = (text or "").strip()
    if _ASR_TEXT_TAG in value:
        value = value.split(_ASR_TEXT_TAG, 1)[1]
    return value.strip()


def normalize_ctc_text(text: str) -> str:
    """Acoustic normalization for CTC targets.

    Case and punctuation are not observable from the waveform, so they are
    removed.  Hyphens become spaces (``x-ray`` -> ``x ray``); decimal points
    are kept; CJK characters and digits are preserved as-is.
    """
    value = unicodedata.normalize("NFKC", transcript_from_target(text))
    value = value.replace("’", "'")
    value = _HYPHEN_RE.sub(" ", value)
    value = value.lower()

    kept: List[str] = []
    for ch in value:
        if ch.isalnum() or ch.isspace() or ch == "'" or ch == ".":
            kept.append(ch)
        else:
            kept.append(" ")
    value = "".join(kept)
    value = _NON_DECIMAL_PERIOD_RE.sub(" ", value)
    return _WS_RE.sub(" ", value).strip()


class SenseVoiceCTCTokenizer:
    """SentencePiece tokenizer from SenseVoiceSmall.

    SenseVoice uses ``blank_id=0`` and a 25,055-piece vocabulary.  The
    SentencePiece ``<unk>`` piece is also id 0, so unknown ids are dropped from
    CTC targets instead of being fed as blanks.
    """

    def __init__(self, model_path: str):
        import sentencepiece as spm

        self.model_path = model_path
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.vocab_size = self.sp.get_piece_size()
        self.blank_id = 0

    def encode(self, text: str) -> List[int]:
        ids = self.sp.encode(text, out_type=int)
        return [idx for idx in ids if idx != self.blank_id]

    def decode(self, ids: Sequence[int]) -> str:
        return self.sp.decode([idx for idx in ids if idx != self.blank_id])


class CTCVocabulary:
    """Compact CTC vocabulary built from the training transcripts.

    Only SentencePiece pieces that occur in the training labels are kept.  This
    shrinks the classifier from the full 25,055-piece SenseVoice vocabulary to
    a few thousand classes and avoids the blank/unigram collapse observed with
    the full vocabulary.  Compact ids are 1-based so that id 0 remains blank.
    """

    def __init__(
        self,
        tokenizer: "SenseVoiceCTCTokenizer",
        source_ids: Sequence[int],
        counts: Optional[Sequence[int]] = None,
    ):
        self.tokenizer = tokenizer
        self.blank_id = 0
        self.source_ids = list(source_ids)
        self.counts = list(counts) if counts is not None else None
        self._orig_to_compact = {
            source_id: index + 1 for index, source_id in enumerate(self.source_ids)
        }
        self.size = len(self.source_ids) + 1

    @classmethod
    def from_texts(
        cls,
        tokenizer: "SenseVoiceCTCTokenizer",
        texts: Sequence[str],
        min_count: int = 1,
    ) -> "CTCVocabulary":
        counter: Counter = Counter()
        for text in texts:
            counter.update(tokenizer.encode(normalize_ctc_text(text)))
        source_ids = [
            source_id
            for source_id, count in sorted(counter.items())
            if count >= min_count and source_id != tokenizer.blank_id
        ]
        return cls(tokenizer, source_ids, [counter[i] for i in source_ids])

    def encode(self, text: str) -> List[int]:
        ids = self.tokenizer.encode(normalize_ctc_text(text))
        return [self._orig_to_compact[i] for i in ids if i in self._orig_to_compact]

    def decode(self, compact_ids: Sequence[int]) -> str:
        source_ids = [
            self.source_ids[compact_id - 1]
            for compact_id in compact_ids
            if compact_id != self.blank_id and 1 <= compact_id <= len(self.source_ids)
        ]
        return self.tokenizer.decode(source_ids)

    def save(self, path: str) -> None:
        payload = {
            "tokenizer": os.path.abspath(self.tokenizer.model_path),
            "blank_id": self.blank_id,
            "source_ids": self.source_ids,
            "pieces": [self.tokenizer.sp.id_to_piece(i) for i in self.source_ids],
            "counts": self.counts,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)

    @classmethod
    def load(cls, path: str, tokenizer: "SenseVoiceCTCTokenizer") -> "CTCVocabulary":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        source_ids = [int(value) for value in payload["source_ids"]]
        if source_ids and max(source_ids) >= tokenizer.vocab_size:
            raise ValueError(
                "CTC vocabulary contains ids outside the loaded tokenizer: "
                f"max id {max(source_ids)}, vocab {tokenizer.vocab_size}"
            )
        return cls(tokenizer, source_ids, payload.get("counts"))


class CTCHead(nn.Module):
    """Bottleneck MLP frame classifier over a compact CTC vocabulary."""

    def __init__(self, hidden_size: int, vocab_size: int, bottleneck: int = 512):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.bottleneck = bottleneck
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, bottleneck),
            nn.GELU(),
            nn.LayerNorm(bottleneck),
            nn.Linear(bottleneck, vocab_size),
        )
        for module in self.proj:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)


def attach_ctc_head(model, vocab_size: int, bottleneck: int = 512) -> CTCHead:
    """Attach a fresh fp32 CTC head to the thinker's audio tower output."""
    thinker = model.thinker
    hidden_size = thinker.audio_tower.proj2.out_features
    head = CTCHead(hidden_size, vocab_size, bottleneck=bottleneck)
    device = next(thinker.audio_tower.parameters()).device
    thinker.ctc_head = head.to(device=device, dtype=torch.float32)
    return thinker.ctc_head


class AudioTowerCapture:
    """Capture the per-sample audio-tower output of the current forward pass."""

    def __init__(self, audio_tower: nn.Module):
        self.hidden_states: List[torch.Tensor] = []
        self.enabled = False
        self._handle = audio_tower.register_forward_hook(self._hook)

    def _hook(self, module, args, output) -> None:
        if not self.enabled:
            return
        hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
        self.hidden_states.append(hidden)

    def start(self) -> None:
        self.hidden_states.clear()
        self.enabled = True

    def stop(self) -> None:
        self.enabled = False

    def remove(self) -> None:
        self._handle.remove()


def compute_ctc_loss(
    hidden_states: Sequence[torch.Tensor],
    head: CTCHead,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    coverage_weight: float = 0.0,
    coverage_detach: bool = True,
) -> torch.Tensor:
    """CTC loss over one batch of per-sample audio-tower outputs."""
    if len(hidden_states) != targets.size(0):
        raise RuntimeError(
            "CTC head saw %d audio-tower outputs for a batch of %d"
            % (len(hidden_states), targets.size(0))
        )
    padded = nn.utils.rnn.pad_sequence(hidden_states, batch_first=True)
    targets = targets.to(padded.device)
    target_lengths = target_lengths.to(padded.device)
    lengths = torch.tensor(
        [hidden.size(0) for hidden in hidden_states],
        dtype=torch.long,
        device=padded.device,
    )
    if torch.any(target_lengths > lengths):
        too_long = [
            (int(length), int(target_length))
            for length, target_length in zip(lengths.tolist(), target_lengths.tolist())
            if target_length > length
        ]
        raise ValueError(
            "CTC target longer than encoder output (frames, tokens): %s" % too_long[:5]
        )

    amp_dtype = torch.bfloat16 if padded.dtype == torch.bfloat16 else torch.float16
    with torch.autocast(
        device_type=padded.device.type,
        dtype=amp_dtype,
        enabled=padded.is_cuda,
    ):
        logits = head(padded)
    log_probs = logits.float().log_softmax(dim=-1).transpose(0, 1)
    loss = nn.functional.ctc_loss(
        log_probs,
        targets,
        lengths,
        target_lengths,
        blank=0,
        reduction="mean",
        zero_infinity=True,
    )
    if coverage_weight > 0:
        if coverage_detach:
            # The coverage term only shapes the head; letting it reach the
            # encoder biases the shared representation toward emitting more
            # tokens (observed as "follow up period ..." loops).
            coverage_log_probs = (
                head(padded.detach().to(dtype=head.proj[1].weight.dtype))
                .float()
                .log_softmax(dim=-1)
                .transpose(0, 1)
            )
        else:
            coverage_log_probs = log_probs
        non_blank = 1.0 - coverage_log_probs[:, :, 0].exp()
        coverage = non_blank.sum(dim=0)
        required = target_lengths.to(dtype=coverage.dtype)
        loss = loss + coverage_weight * torch.relu(required - coverage).pow(2).mean()
    return loss


def ctc_greedy_decode(logits: torch.Tensor, input_lengths: torch.Tensor) -> List[List[int]]:
    """Greedy CTC decode (collapse repeats, keep blanks for the caller to drop)."""
    predictions = logits.argmax(dim=-1)
    decoded: List[List[int]] = []
    for row, length in zip(predictions, input_lengths.tolist()):
        sequence = row[: int(length)]
        if sequence.numel() == 0:
            decoded.append([])
            continue
        keep = torch.ones_like(sequence, dtype=torch.bool)
        keep[1:] = sequence[1:] != sequence[:-1]
        decoded.append(sequence[keep].tolist())
    return decoded