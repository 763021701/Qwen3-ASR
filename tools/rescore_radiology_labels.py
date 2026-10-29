#!/usr/bin/env python3
"""Rescore saved radiology predictions against an updated reference CSV.

Saved predictions keep the raw hypothesis, so a reference-only change (label fix in
raw/POC_test/radiology.csv) can be re-scored without re-running inference. For each
NAME=PREDS.jsonl, rows are joined to the new references by absolute audio path; corpus
WER (masr_eval_pkg) and corpus CER (shared normalize + cer_decomposition) plus per-clip
stats are recomputed with the CURRENT normalizer.

Usage:
    python tools/rescore_radiology_labels.py NAME=path/predictions.jsonl [NAME=...] \
        [--csv raw/POC_test/radiology.csv] [--out outputs/.../rescore.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "finetuning"))

from evaluation.english_medical.text_normalization import normalize_english  # noqa: E402
from masr_eval_pkg import compute_wer  # noqa: E402
from qwen3_asr_grpo import cer_decomposition  # noqa: E402


def load_refs(path: Path) -> dict[str, str]:
    refs = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            refs[str(Path(str(row["audio_path"])).resolve())] = str(row["text"])
    return refs


def score(pred_path: Path, refs: dict[str, str]) -> dict:
    unmatched = 0
    refs_n: list[str] = []
    hyps_n: list[str] = []
    per_clip: list[float] = []
    s = d = i = n = 0
    for line in pred_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        ref = refs.get(str(Path(r["audio"]).resolve()))
        if ref is None:
            unmatched += 1
            continue
        rn = normalize_english(ref)
        hn = normalize_english(r["hypothesis"])
        refs_n.append(rn)
        hyps_n.append(hn)
        per_clip.append(compute_wer([rn], [hn])["wer"])
        a, b, c, m = cer_decomposition(ref, r["hypothesis"])
        s += a; d += b; i += c; n += m
    if not refs_n:
        return {"error": "no matched rows", "unmatched": unmatched}
    med = sorted(per_clip)[len(per_clip) // 2]
    return {
        "n": len(refs_n),
        "unmatched": unmatched,
        "corpus_wer": round(compute_wer(refs_n, hyps_n)["wer"], 4),
        "corpus_cer": round((s + d + i) / max(n, 1), 4),
        "median_wer": round(med, 4),
        "lt30": sum(w < 0.30 for w in per_clip),
        "gt100": sum(w > 1.00 for w in per_clip),
        "exact": sum(w == 0 for w in per_clip),
        "ref_tokens": sum(len(x.split()) for x in refs_n),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("specs", nargs="+", help="NAME=path/to/predictions.jsonl")
    ap.add_argument("--csv", default="raw/POC_test/radiology.csv")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    refs = load_refs(Path(args.csv))
    print(f"references: {len(refs)} clips from {args.csv}")
    results = []
    for spec in args.specs:
        name, _, path = spec.partition("=")
        res = score(Path(path), refs)
        res["name"] = name
        results.append(res)
        if "error" in res:
            print(f"{name:34s} {res['error']} (unmatched={res['unmatched']})")
        else:
            print(
                "%-34s n=%-3d WER=%7.2f%% CER=%7.2f%% median=%6.1f%% <30%%=%2d >100%%=%2d exact=%2d refT=%d"
                % (name, res["n"], 100 * res["corpus_wer"], 100 * res["corpus_cer"],
                   100 * res["median_wer"], res["lt30"], res["gt100"], res["exact"], res["ref_tokens"])
            )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"csv": args.csv, "results": results}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print("wrote", args.out)


if __name__ == "__main__":
    main()