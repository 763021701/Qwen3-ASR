#!/usr/bin/env python3
"""Combine two models' predictions by keeping the shorter hypothesis (tie -> A).

Unsupervised system combination: no references are used for selection. Empirically
on radiology95 it improves 41.19/41.37 -> 39.36 corpus WER, and on holdout19
59.75/59.75 -> 54.68. Writes the combined jsonl and a WER/CER summary.

    python tools/combine_predictions_shortest.py --a a.jsonl --b b.jsonl \
        --out combined.jsonl --summary summary.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "finetuning"))

from evaluation.english_medical.text_normalization import normalize_english  # noqa: E402
from masr_eval_pkg import compute_wer  # noqa: E402
from qwen3_asr_grpo import cer_decomposition  # noqa: E402


def load(path: str) -> dict[str, dict]:
    return {json.loads(l)["audio"]: json.loads(l) for l in open(path, encoding="utf-8") if l.strip()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", default="")
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    keys = [k for k in A if k in B]
    rows = [A[k] if len(A[k]["hypothesis"]) <= len(B[k]["hypothesis"]) else B[k] for k in keys]
    refs = [normalize_english(r["reference"]) for r in rows]
    hyps = [normalize_english(r["hypothesis"]) for r in rows]
    s = d = i = n = 0
    for r in rows:
        x, y, z, m = cer_decomposition(r["reference"], r["hypothesis"])
        s += x; d += y; i += z; n += m
    wer = compute_wer(refs, hyps)["wer"]
    exact = sum(compute_wer([r], [h])["wer"] == 0 for r, h in zip(refs, hyps))
    lines = [
        "=== shortest-hypothesis combination ===",
        f"A: {args.a}",
        f"B: {args.b}",
        f"Samples: {len(rows)}",
        f"Corpus WER:          {100 * wer:.2f}%",
        f"Corpus CER:          {100 * (s + d + i) / max(n, 1):.2f}%",
        f"Substitutions:       {s}",
        f"Deletions:           {d}",
        f"Insertions:          {i}",
        f"Ref tokens (total):  {sum(len(x.split()) for x in refs)}",
        f"Sentence exact acc:  {100 * exact / max(len(rows), 1):.2f}% ({exact}/{len(rows)})",
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for r in rows:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = "\n".join(lines)
    print(summary)
    if args.summary:
        Path(args.summary).write_text(summary + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()