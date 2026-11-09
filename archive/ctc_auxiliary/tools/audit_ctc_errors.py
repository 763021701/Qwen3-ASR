#!/usr/bin/env python3
"""Break down CTC-only errors by Cantonese / English / code-switch / medical."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evaluation.english_medical.text_normalization import normalize_english
from finetuning.qwen3_asr_grpo import cer_decomposition
from masr_eval_pkg import compute_wer

CANTO_MARKERS = set("嘅咁係唔嗰喺咗啲冇嚟睇佢哋畀諗啱啦囉呀喎咩噉點廿誒")
MEDICAL_TERMS = [
    "lymph", "nodule", "lesion", "fracture", "opacity", "consolidation",
    "effusion", "bronchiectasis", "mediastinal", "pleural", "thickening",
    "hyperdensity", "lucency", "sinus", "orbital", "cavity", "segment",
    "lobe", "ct", "x-ray", "x ray", "mri", "ultrasound", "biopsy",
    "carcinoma", "metastasis", "mass", "edema", "hemorrhage", "calcification",
    "vertebral", "body", "fossa", "sella", "ventricle", "parenchyma",
]


def _has_cjk(text: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in text)


def _has_latin(text: str) -> bool:
    return any("a" <= ch.lower() <= "z" for ch in text)


def _cer(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    s = d = i = n = 0
    for row in rows:
        a, b, c, m = cer_decomposition(row["reference"], row["hypothesis"])
        s += a
        d += b
        i += c
        n += m
    refs = [normalize_english(row["reference"]) for row in rows]
    hyps = [normalize_english(row["hypothesis"]) for row in rows]
    return {
        "n": len(rows),
        "cer": (s + d + i) / n if n else 0.0,
        "wer": float(compute_wer(refs, hyps)["wer"]),
        "S": s,
        "D": d,
        "I": i,
        "N": n,
        "I_over_N": i / n if n else 0.0,
    }


def _term_recall(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = 0
    found = 0
    per_term: Dict[str, List[int]] = {}
    for row in rows:
        ref = normalize_english(row["reference"])
        hyp = normalize_english(row["hypothesis"])
        for term in MEDICAL_TERMS:
            if term in ref:
                total += 1
                hit = int(term in hyp)
                found += hit
                per_term.setdefault(term, [0, 0])
                per_term[term][0] += hit
                per_term[term][1] += 1
    return {
        "occurrences": total,
        "recall": found / total if total else 0.0,
        "per_term": {term: {"hit": hit, "total": count} for term, (hit, count) in per_term.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.predictions, encoding="utf-8") if line.strip()]
    categories: Dict[str, List[Dict[str, Any]]] = {
        "all": rows,
        "english": [],
        "cantonese": [],
        "code_switch": [],
        "medical": [],
    }
    for row in rows:
        ref = row["reference"]
        cjk = _has_cjk(ref)
        latin = _has_latin(ref)
        if not cjk:
            categories["english"].append(row)
        if cjk and any(ch in CANTO_MARKERS for ch in ref):
            categories["cantonese"].append(row)
        if cjk and latin:
            categories["code_switch"].append(row)
        lowered = ref.lower()
        if any(term in lowered for term in MEDICAL_TERMS):
            categories["medical"].append(row)

    report = {
        "predictions": os.path.abspath(args.predictions),
        "categories": {name: _cer(subset) for name, subset in categories.items()},
        "medical_terms": _term_recall(categories["medical"]),
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("%-12s %4s %8s %8s %8s %8s %8s" % ("subset", "n", "CER", "WER", "S", "D", "I/N"))
    for name, metrics in report["categories"].items():
        print(
            "%-12s %4d %7.2f%% %7.2f%% %8d %8d %7.2f%%"
            % (
                name,
                metrics["n"],
                100 * metrics["cer"],
                100 * metrics["wer"],
                metrics["S"],
                metrics["D"],
                100 * metrics["I_over_N"],
            )
        )
    medical = report["medical_terms"]
    print("medical term recall: %d/%d = %.2f%%" % (
        0 if not medical["occurrences"] else round(medical["recall"] * medical["occurrences"]),
        medical["occurrences"],
        100 * medical["recall"],
    ))


if __name__ == "__main__":
    main()