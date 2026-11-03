"""Manifests, transcript cleaning, weighted duration-bucketed batches and collate (DESIGN.md §7, §8).

Manifest: JSONL, one utterance per line, e.g.
  {"id": "aishell1-BAC009S0002W0122", "audio": "/data/.../BAC009S0002W0122.wav", "start": 0.0, "end": null,
   "duration": 4.21, "text": "而对楼市成交抑制作用最大的限购", "lang": "zh", "style": "woitn",
   "source": "aishell1", "label_type": "human"}
``text`` is the raw transcript; it is normalized at load time so the rules can change later.
Optional ``"t2s"``: OpenCC config ("hk2s" / "tw2s" / "t2s") for sources written in traditional characters.
Optional ``"tar_offset"`` / ``"tar_size"``: ``audio`` is a WebDataset shard and the clip is the member at that byte
range (see tools/wds_manifest.py), so shards are used without extracting them.
"""
from __future__ import annotations

import io
import json
import random
import re
import sys
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
# OpenCC rewrites these Cantonese characters into rare variants (嗰→𠮶, 喎→㖞, 餸→𩠌, 噚→㖊; the last three are not
# in the vocab), so they are kept as written.
_OPENCC_KEEP_RE = re.compile("([嗰喎餸噚])")
_opencc, _cn_tn = {}, None


def to_simplified(text: str, config: str = "t2s") -> str:
    """OpenCC traditional -> simplified (config t2s / hk2s / tw2s). For traditional-script sources only: on simplified
    text it is not a no-op (inside some phrases 坏 becomes 坯 and 沈 becomes 沉)."""
    if config not in _opencc:
        from opencc import OpenCC

        _opencc[config] = OpenCC(config).convert
    return "".join(p if k % 2 else _opencc[config](p) for k, p in enumerate(_OPENCC_KEEP_RE.split(text)))


def _chinese_tn(text: str) -> str:
    global _cn_tn
    if _cn_tn is None:
        from .cn_tn import TextNorm

        _cn_tn = TextNorm()  # defaults: verbalize numbers, punctuation -> space, keep spaces and English
    return _cn_tn(text)


def _english_tn(text: str) -> str:
    try:
        from .en_tn import english_tn
    except ImportError:  # num2words is optional: the digits stay and the sample is dropped
        return text
    return english_tn(text) or text


def clean_transcript(text: str, lang=None, style: str = "woitn", t2s=None) -> tuple[str | None, str]:
    """Raw transcript -> (normalized text or None, reason).

    t2s: OpenCC config for traditional-script sources (manifest field "t2s"); None keeps the characters.
    TN runs before normalize_text because it needs the punctuation (decimal points, %, dates) that normalize_text
    removes. Text with Han characters, or with a non-English lang (zh, yue, wuyu, ...), uses cn_tn; other text uses
    en_tn (num2words). Text whose digits cannot be verbalized is dropped (DESIGN §7.2 rule 5).
    """
    text = _MARKUP_RE.sub(" ", text or "")
    if t2s:
        text = to_simplified(text, t2s)
    if style == "woitn" and _DIGIT_RE.search(text):
        text = to_halfwidth_alnum(text)
        if _CJK_RE.search(text) or (lang and not lang.startswith("en")):
            text = _chinese_tn(text)
        else:
            text = _english_tn(text)
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
    text, reason = clean_transcript(ex.get("text", ""), ex.get("lang"), style, ex.get("t2s"))
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


def read_audio(path: str, start: float = 0.0, end=None, sr: int = SAMPLE_RATE, tar_offset=None, tar_size=None) -> np.ndarray:
    """tar_offset / tar_size: the clip is a member of an uncompressed tar (WebDataset shard) at that byte range."""
    if tar_offset is not None:
        with open(path, "rb") as f:
            f.seek(tar_offset)
            path = io.BytesIO(f.read(tar_size))
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
        self.items, self.sources, self.durations, self.stats = [], [], [], Counter()
        for src, path in enumerate(manifests):
            for ex in load_manifest(path):
                ids, reason, info = prepare_example(ex, tokenizer, style, min_duration, max_duration, drop_unk,
                                                    upsample, num_queries, check_feasible)
                self.stats[reason] += 1
                if ids is None:
                    continue
                # int32 targets and interned paths keep millions of entries small (and shared with forked workers)
                self.items.append((ex["id"], sys.intern(ex["audio"]), float(ex.get("start") or 0.0), ex.get("end"),
                                   info["text"], np.asarray(ids, dtype=np.int32), ex.get("tar_offset"), ex.get("tar_size")))
                self.sources.append(src)
                self.durations.append(info["duration"])
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        uid, audio, start, end, text, ids, tar_offset, tar_size = self.items[i]
        wav = read_audio(audio, start, end, tar_offset=tar_offset, tar_size=tar_size)
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
