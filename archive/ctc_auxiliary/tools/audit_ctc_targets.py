#!/usr/bin/env python3
"""Audit CTC supervision: frame/token margin, UNK, zero_infinity, decoding."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Any, Dict, List

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from finetuning.ctc_auxiliary import (
    CTCVocabulary,
    SenseVoiceCTCTokenizer,
    ctc_greedy_decode,
    normalize_ctc_text,
)
from qwen_asr.core.transformers_backend.processing_qwen3_asr import (
    _get_feat_extract_output_lengths,
)


def _frames(duration: float, speed: float = 1.0) -> int:
    mel = max(1, int(round(duration * 100.0 / speed)))
    return int(_get_feat_extract_output_lengths(torch.tensor([mel])).item())


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def _audit(name: str, rows: List[Dict[str, Any]], vocab: CTCVocabulary, tokenizer) -> Dict[str, Any]:
    stats = []
    unk_tokens = 0
    dropped_tokens = 0
    total_tokens = 0
    roundtrip_ok = 0
    roundtrip_bad = []
    for row in rows:
        text = row.get("text", "")
        norm = normalize_ctc_text(text)
        raw_ids = tokenizer.sp.encode(norm, out_type=int)
        ids = vocab.encode(text)
        unk = sum(1 for i in raw_ids if i == tokenizer.blank_id)
        dropped = sum(1 for i in raw_ids if i != tokenizer.blank_id and i not in vocab._orig_to_compact)
        unk_tokens += unk
        dropped_tokens += dropped
        total_tokens += len(raw_ids)

        repeats = sum(1 for i in range(1, len(ids)) if ids[i] == ids[i - 1])
        u = len(ids)
        need = u + repeats
        duration = float(row.get("duration_sec") or 0.0)
        t = _frames(duration)
        t12 = _frames(duration, 1.2)
        margin = t / need if need else float("inf")
        margin12 = t12 / need if need else float("inf")

        decoded = vocab.decode(vocab.encode(text))
        if decoded == norm:
            roundtrip_ok += 1
        elif len(roundtrip_bad) < 5:
            roundtrip_bad.append({"text": norm, "decoded": decoded})

        stats.append(
            {
                "audio": row.get("audio", ""),
                "duration_sec": duration,
                "source_group": str(row.get("source_group", "")),
                "T": t,
                "T_speed12": t12,
                "U": u,
                "R": repeats,
                "need": need,
                "margin": margin,
                "margin_speed12": margin12,
                "unk": unk,
                "dropped": dropped,
                "unalignable": need > t,
                "unalignable_speed12": need > t12,
                "u_gt_t": u > t,
            }
        )

    margins = [s["margin"] for s in stats if s["need"]]
    margins12 = [s["margin_speed12"] for s in stats if s["need"]]
    n = len(stats)
    unalign = sum(1 for s in stats if s["unalignable"])
    unalign12 = sum(1 for s in stats if s["unalignable_speed12"])
    u_gt_t = sum(1 for s in stats if s["u_gt_t"])
    tight12 = sum(1 for s in stats if s["margin"] < 1.2)
    tight15 = sum(1 for s in stats if s["margin"] < 1.5)
    tight12_speed = sum(1 for s in stats if s["margin_speed12"] < 1.2)
    tight15_speed = sum(1 for s in stats if s["margin_speed12"] < 1.5)
    summary = {
        "set": name,
        "n": n,
        "T_median": _percentile([s["T"] for s in stats], 0.5),
        "T_p05": _percentile([s["T"] for s in stats], 0.05),
        "U_median": _percentile([s["U"] for s in stats], 0.5),
        "R_median": _percentile([s["R"] for s in stats], 0.5),
        "need_median": _percentile([s["need"] for s in stats], 0.5),
        "margin_p01": _percentile(margins, 0.01),
        "margin_p05": _percentile(margins, 0.05),
        "margin_p50": _percentile(margins, 0.5),
        "margin_min": min(margins) if margins else 0.0,
        "unalignable": unalign,
        "unalignable_pct": 100.0 * unalign / n if n else 0.0,
        "unalignable_speed12": unalign12,
        "unalignable_speed12_pct": 100.0 * unalign12 / n if n else 0.0,
        "u_gt_t": u_gt_t,
        "u_gt_t_pct": 100.0 * u_gt_t / n if n else 0.0,
        "margin_lt1.2": tight12,
        "margin_lt1.2_pct": 100.0 * tight12 / n if n else 0.0,
        "margin_lt1.5": tight15,
        "margin_lt1.5_pct": 100.0 * tight15 / n if n else 0.0,
        "margin_speed12_lt1.2_pct": 100.0 * tight12_speed / n if n else 0.0,
        "margin_speed12_lt1.5_pct": 100.0 * tight15_speed / n if n else 0.0,
        "unk_tokens": unk_tokens,
        "unk_pct": 100.0 * unk_tokens / total_tokens if total_tokens else 0.0,
        "dropped_tokens": dropped_tokens,
        "dropped_pct": 100.0 * dropped_tokens / total_tokens if total_tokens else 0.0,
        "roundtrip_ok": roundtrip_ok,
        "roundtrip_bad": roundtrip_bad,
        "margin_speed12_p01": _percentile(margins12, 0.01),
        "margin_speed12_p50": _percentile(margins12, 0.5),
    }
    return summary, stats


def _load_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _greedy_check(vocab: CTCVocabulary) -> Dict[str, Any]:
    logits = torch.full((1, 8, 8), -10.0)
    sequence = [1, 1, 0, 2, 2, 3, 0, 3]
    for frame, token in enumerate(sequence):
        logits[0, frame, token] = 10.0
    decoded = ctc_greedy_decode(logits, torch.tensor([len(sequence)]))[0]
    reference = []
    previous = -1
    for frame in range(len(sequence)):
        token = int(logits[0, frame].argmax())
        if token != previous:
            reference.append(token)
        previous = token
    return {
        "sequence": sequence,
        "ctc_greedy_decode": decoded,
        "reference_collapse": reference,
        "match": decoded == reference,
        "decoded_text": vocab.decode(decoded),
        "nonblank": [token for token in decoded if token != 0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--ctc_tokenizer", required=True)
    parser.add_argument("--ctc_vocab", required=True)
    parser.add_argument("--eval_jsonl", action="append", default=[], help="NAME=PATH")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    tokenizer = SenseVoiceCTCTokenizer(args.ctc_tokenizer)
    vocab = CTCVocabulary.load(args.ctc_vocab, tokenizer)
    os.makedirs(args.output_dir, exist_ok=True)

    sets = {"train": _load_rows(args.train_jsonl)}
    for spec in args.eval_jsonl:
        name, _, path = spec.partition("=")
        sets[name] = _load_rows(path)

    report = {
        "vocab": {
            "path": os.path.abspath(args.ctc_vocab),
            "size": vocab.size,
            "tokens": len(vocab.source_ids),
            "tokenizer": os.path.abspath(args.ctc_tokenizer),
        },
        "greedy_check": _greedy_check(vocab),
        "sets": {},
    }
    for name, rows in sets.items():
        summary, stats = _audit(name, rows, vocab, tokenizer)
        report["sets"][name] = summary
        with open(os.path.join(args.output_dir, f"targets_{name}.json"), "w", encoding="utf-8") as handle:
            json.dump({"summary": summary, "rows": stats}, handle, ensure_ascii=False, indent=2)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    with open(os.path.join(args.output_dir, "targets_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print("greedy_check:", json.dumps(report["greedy_check"], ensure_ascii=False))


if __name__ == "__main__":
    main()