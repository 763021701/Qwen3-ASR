"""Manifests, transcript cleaning, weighted duration-bucketed batches and collate (DESIGN.md §7, §8).

Manifest: JSONL, one utterance per line, e.g.
  {"id": "aishell1-BAC009S0002W0122", "audio": "/data/.../BAC009S0002W0122.wav", "start": 0.0, "end": null,
   "duration": 4.21, "text": "而对楼市成交抑制作用最大的限购", "lang": "zh", "style": "woitn",
   "source": "aishell1", "label_type": "human"}
``text`` is the raw transcript; it is normalized at load time so the rules can change later.
"""
from __future__ import annotations

import json
import random
import re
import warnings
from collections import Counter

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, Sampler

from .augment import resample, spec_augment
from .frontend import SAMPLE_RATE, aut_output_lengths, compute_features
from .tokenizer import CJK, normalize_text, to_halfwidth_alnum

# Dataset-specific markers (<SIL>, GigaSpeech <COMMA>, [noise], ...) are removed for every source.
_MARKUP_RE = re.compile(r"<[A-Z_]+>|\[[A-Za-z_ ]+\]")
_CJK_RE = re.compile(f"[{CJK}]")
_DIGIT_RE = re.compile(r"\d")

try:
    from opencc import OpenCC

    _t2s = OpenCC("t2s").convert
except ImportError:  # optional: the pipeline then keeps traditional characters as they are
    _t2s = None

_cn_tn = None


def _chinese_tn(text: str) -> str:
    global _cn_tn
    if _cn_tn is None:
        from .cn_tn import TextNorm

        _cn_tn = TextNorm()  # defaults: verbalize numbers, punctuation -> space, keep spaces and English
    return _cn_tn(text)


def clean_transcript(text: str, lang=None, style: str = "woitn") -> tuple[str | None, str]:
    """Raw transcript -> (normalized text or None, reason).

    TN runs before normalize_text because it needs the punctuation (decimal points, %, dates) that
    normalize_text removes. Chinese (lang contains "zh", or CJK text when lang is unknown) uses cn_tn;
    there is no English TN here, so English text that still has digits is dropped (DESIGN §7.2 rule 5).
    """
    text = _MARKUP_RE.sub(" ", text or "")
    if _t2s is not None:
        text = _t2s(text)
    if style == "woitn" and _DIGIT_RE.search(text):
        chinese = "zh" in lang if lang else bool(_CJK_RE.search(text))
        if chinese:
            text = _chinese_tn(to_halfwidth_alnum(text))
    text, has_digit = normalize_text(text, style)
    if style == "woitn" and has_digit:
        return None, "digits_without_tn"
    return text, "ok"


def utterance_duration(ex: dict) -> float:
    if ex.get("duration"):
        return float(ex["duration"])
    start = float(ex.get("start") or 0.0)
    if ex.get("end") is not None:
        return float(ex["end"]) - start
    return sf.info(ex["audio"]).duration - start


def aut_frames_for_duration(duration: float) -> int:
    return int(aut_output_lengths(torch.tensor(int(round(duration * SAMPLE_RATE)) // 160)))


def prepare_example(ex: dict, tokenizer, style="woitn", min_duration=0.3, max_duration=30.0, drop_unk=True,
                    upsample=1, num_queries=0, check_feasible=True):
    """Filter + clean + tokenize one manifest entry -> (target ids or None, reason, info).

    info carries what the M2 report needs: duration, AuT frames T, U' (targets + repeats), raw digits.
    Evaluation sets pass min_duration=0, max_duration=inf, drop_unk=False, check_feasible=False so that only
    references that cannot be normalized (digits without TN) are dropped.
    """
    info = {"duration": utterance_duration(ex), "raw_has_digit": bool(_DIGIT_RE.search(ex.get("text") or ""))}
    if not min_duration <= info["duration"] <= max_duration:
        return None, "duration", info
    text, reason = clean_transcript(ex.get("text", ""), ex.get("lang"), style)
    if text is None:
        return None, reason, info
    ids = tokenizer.encode(text)
    info.update(text=text, num_tokens=len(ids), num_unk=ids.count(tokenizer.UNK))
    if drop_unk and info["num_unk"]:
        return None, "unk", info
    info["T"] = (aut_frames_for_duration(info["duration"]) + num_queries) * upsample
    info["U'"] = len(ids) + num_queries + sum(a == b for a, b in zip(ids, ids[1:]))  # targets include query tags
    if check_feasible and info["U'"] > info["T"]:
        return None, "ctc_infeasible", info
    return ids, "ok" if ids else "ok_empty", info


def load_manifest(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_audio(path: str, start: float = 0.0, end=None, sr: int = SAMPLE_RATE) -> np.ndarray:
    with sf.SoundFile(path) as f:
        file_sr = f.samplerate
        f.seek(int(round(start * file_sr)))
        frames = -1 if end is None else int(round((float(end) - start) * file_sr))
        wav = f.read(frames, dtype="float32", always_2d=True).mean(1)
    return resample(wav, file_sr, sr)


class ManifestDataset(Dataset):
    """Text is cleaned and tokenized up front so filtering is exact; ``stats`` counts the reasons."""

    def __init__(self, manifests, tokenizer, style="woitn", min_duration=0.3, max_duration=30.0, drop_unk=True,
                 upsample=1, num_queries=0, check_feasible=True, augment=None):
        if _t2s is None:
            warnings.warn("opencc is not installed: traditional characters are not converted (DESIGN §7.2 rule 2)")
        self.items, self.sources, self.durations, self.stats = [], [], [], Counter()
        for src, path in enumerate(manifests):
            for ex in load_manifest(path):
                ids, reason, info = prepare_example(ex, tokenizer, style, min_duration, max_duration, drop_unk,
                                                    upsample, num_queries, check_feasible)
                self.stats[reason] += 1
                if ids is None:
                    continue
                self.items.append((ex["id"], ex["audio"], float(ex.get("start") or 0.0), ex.get("end"),
                                   info["text"], ids))
                self.sources.append(src)
                self.durations.append(info["duration"])
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        uid, audio, start, end, text, ids = self.items[i]
        wav = read_audio(audio, start, end)
        if self.augment is not None:  # torch seeds every DataLoader worker differently
            wav = self.augment(wav, np.random.default_rng(int(torch.randint(2**31, ()))))
        return {"id": uid, "wav": wav, "text": text, "targets": ids}


def duration_batches(indices, durations, max_batch_seconds):
    """Sort by duration and cut so that batch size x longest clip <= max_batch_seconds."""
    batches, cur = [], []
    for i in sorted(indices, key=lambda i: durations[i]):
        if cur and (len(cur) + 1) * durations[i] > max_batch_seconds:
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


class WeightedBucketBatchSampler(Sampler):
    """Infinite stream of duration-bucketed batches from several manifests.

    Every step picks a manifest with probability proportional to its weight (equal weights give each manifest
    the same share of steps whatever its size), then takes that manifest's next batch; a manifest's batch
    order is reshuffled on every pass. The trainer stops at total_steps.
    """

    def __init__(self, durations, sources, weights, max_batch_seconds, seed=0):
        n_src = max(sources) + 1 if sources else 0
        if len(weights) != n_src:
            raise ValueError(f"{len(weights)} sampling weights for {n_src} manifests")
        self.batches = [duration_batches([i for i, s in enumerate(sources) if s == src], durations,
                                         max_batch_seconds) for src in range(n_src)]
        self.weights = [float(w) for w in weights]
        self.rng = random.Random(seed)

    def num_batches(self) -> int:
        return sum(len(b) for b in self.batches)

    def __iter__(self):
        active = [s for s, b in enumerate(self.batches) if b and self.weights[s] > 0]
        weights = [self.weights[s] for s in active]
        queues = {s: [] for s in active}
        while True:
            s = self.rng.choices(active, weights=weights)[0]
            if not queues[s]:
                queues[s] = self.rng.sample(self.batches[s], len(self.batches[s]))
            yield queues[s].pop()


class Collator:
    """Batch log-mel (padding="longest") + optional SpecAugment + right-padded targets."""

    def __init__(self, fe, specaug=None):
        self.fe, self.specaug = fe, specaug

    def __call__(self, items):
        feats, mel_lens = compute_features(self.fe, [it["wav"] for it in items])
        if self.specaug:
            spec_augment(feats, mel_lens, np.random.default_rng(int(torch.randint(2**31, ()))), **self.specaug)
        target_lens = torch.tensor([len(it["targets"]) for it in items], dtype=torch.long)
        targets = torch.zeros(len(items), max(1, int(target_lens.max())), dtype=torch.long)
        for b, it in enumerate(items):
            targets[b, : len(it["targets"])] = torch.tensor(it["targets"], dtype=torch.long)
        return {"ids": [it["id"] for it in items], "texts": [it["text"] for it in items], "input_features": feats,
                "mel_lens": mel_lens, "targets": targets, "target_lens": target_lens}
