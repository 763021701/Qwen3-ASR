#!/usr/bin/env python3
# coding=utf-8
"""Build empty/gold/distractor prompt copies from a Qwen3-ASR JSONL manifest."""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluation.english_medical.text_normalization import normalize_english

GOLD_CATEGORIES = {
    "procedures_specimen_types",
    "specimen_labels",
    "lymph_nodes_and_ids",
    "lesions",
}
GOLD_GROSSING = {
    "frozen and paraffin sections",
    "sampled for frozen and paraffin sections",
    "scrape cytology",
    "serially sectioned",
}
SKIP_GOLD_NORM = {
    "mass",
    "cyst",
    "polyp",
    "tumour",
    "tumor",
    "nodule",
    "bulla",
    "intramural",
    "fibroid",
    "fibroids",
}
CONFUSION = {
    "hemithyroidectomy": "endometrial cone",
    "hemithyroidectomy specimen": "endometrial cone",
    "frozen and paraffin sections": "full turn",
    "sampled for frozen and paraffin sections": "full turn",
    "scrape cytology": "full turn",
    "non-sn": "sn 3",
    "non sn": "sn 3",
    "sn 1": "specimen 1",
    "sn 2": "sn 3",
    "subserosal fibroids": "serial fibroids",
    "coronal full slabs": "whole node",
    "sentinel node": "right colon",
    "left sentinel node": "right colon",
    "tubal mass": "superomax",
    "lumpectomy": "mastectomy",
}
EXTRA_GOLD_RE = re.compile(r"\b(?:non\s+sn|sn\s+\d+)\b", re.IGNORECASE)
ASR_TAG = "<asr_text>"
REAL_SOURCE_RE = re.compile(r"real", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_jsonl", required=True)
    p.add_argument("--keywords_json", default=str(_REPO / "docs/real_domain_medical_keywords.json"))
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--report_json", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--real_only",
        type=int,
        default=1,
        help="Keep rows whose sampling_source matches /real/i.",
    )
    p.add_argument("--max_gold_phrases", type=int, default=2)
    p.add_argument("--gold_frac", type=float, default=0.60)
    p.add_argument("--gold_distractor_frac", type=float, default=0.25)
    p.add_argument(
        "--keep_empty",
        type=int,
        default=1,
        help="Write original rows with empty prompt.",
    )
    return p.parse_args()


def transcript_body(text: str) -> str:
    if ASR_TAG in (text or ""):
        return text.split(ASR_TAG, 1)[1]
    return text or ""


def contains(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    return f" {needle} " in f" {haystack} "


def load_phrases(path: Path) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    gold: List[Tuple[str, str, str]] = []
    pool: List[str] = []
    seen_pool = set()
    for cat in data.get("categories", []):
        cat_id = str(cat.get("id") or "")
        for item in cat.get("items") or []:
            raw = str(item.get("phrase") or "").strip()
            if not raw:
                continue
            norm = normalize_english(raw)
            if len(norm) < 4 or norm in SKIP_GOLD_NORM or len(norm) > 40:
                continue
            if cat_id in GOLD_CATEGORIES or (
                cat_id == "grossing_actions" and raw.lower() in GOLD_GROSSING
            ):
                gold.append((raw, norm, cat_id))
            if cat_id in GOLD_CATEGORIES and norm not in seen_pool:
                pool.append(raw)
                seen_pool.add(norm)
    gold.sort(key=lambda row: len(row[1]), reverse=True)
    return gold, pool


def extract_gold(body_norm: str, catalog: Sequence[Tuple[str, str, str]], max_n: int) -> List[str]:
    hits: List[str] = []
    used = ""
    for raw, norm, _cat in catalog:
        if contains(body_norm, norm) and not contains(used, norm):
            hits.append(raw)
            used = (used + " " + norm).strip()
            if len(hits) >= max_n:
                break
    if len(hits) < max_n:
        for match in EXTRA_GOLD_RE.finditer(body_norm):
            token = match.group(0)
            display = "non-SN" if token.startswith("non") else "SN " + token.split()[-1]
            if display not in hits and contains(body_norm, normalize_english(display)):
                hits.append(display)
            if len(hits) >= max_n:
                break
    return hits[:max_n]


def pick_distractors(
    body_norm: str,
    gold: Sequence[str],
    pool: Sequence[str],
    rng: random.Random,
    k: int = 2,
) -> List[str]:
    out: List[str] = []
    for phrase in gold:
        mapped = CONFUSION.get(phrase.lower()) or CONFUSION.get(normalize_english(phrase))
        if not mapped:
            continue
        if contains(body_norm, normalize_english(mapped)):
            continue
        if mapped not in out:
            out.append(mapped)
        if len(out) >= k:
            return out
    gold_norm = {normalize_english(p) for p in gold}
    candidates = [
        p
        for p in pool
        if normalize_english(p) not in gold_norm and not contains(body_norm, normalize_english(p))
    ]
    rng.shuffle(candidates)
    for phrase in candidates:
        if phrase not in out:
            out.append(phrase)
        if len(out) >= k:
            break
    return out


def is_real_row(row: Dict[str, Any]) -> bool:
    return bool(REAL_SOURCE_RE.search(str(row.get("sampling_source") or "")))


def choose_kind(rng: random.Random, gold_frac: float, mixed_frac: float) -> str:
    draw = rng.random()
    if draw < gold_frac:
        return "gold"
    if draw < gold_frac + mixed_frac:
        return "gold_distractor"
    return "distractor"


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    catalog, pool = load_phrases(Path(args.keywords_json))
    src_counts: Counter[str] = Counter()
    n_in = 0
    n_real = 0
    n_gold_eligible = 0
    n_empty = 0
    n_prompted = 0
    kind_counts: Counter[str] = Counter()
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Path(args.input_jsonl).open(encoding="utf-8") as handle, out_path.open(
        "w", encoding="utf-8"
    ) as out:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_in += 1
            src_counts[str(row.get("sampling_source") or "?")] += 1
            if args.real_only and not is_real_row(row):
                continue
            n_real += 1
            body_norm = normalize_english(transcript_body(str(row.get("text") or "")))
            gold = extract_gold(body_norm, catalog, args.max_gold_phrases)
            if gold:
                n_gold_eligible += 1

            if args.keep_empty:
                empty = dict(row)
                empty["prompt"] = ""
                empty["prompt_kind"] = "empty"
                out.write(json.dumps(empty, ensure_ascii=False) + "\n")
                n_empty += 1

            if not gold:
                continue
            distractors = pick_distractors(body_norm, gold, pool, rng)
            kind = choose_kind(rng, args.gold_frac, args.gold_distractor_frac)
            if kind == "gold":
                prompt_phrases = list(gold)
            elif kind == "gold_distractor":
                prompt_phrases = list(gold) + distractors[:1]
            else:
                prompt_phrases = distractors or list(gold)
                if not distractors:
                    kind = "gold"
                    prompt_phrases = list(gold)
            prompted = dict(row)
            prompted["prompt"] = " ".join(prompt_phrases)
            prompted["prompt_kind"] = kind
            prompted["prompt_gold"] = gold
            prompted["prompt_distractor"] = distractors
            out.write(json.dumps(prompted, ensure_ascii=False) + "\n")
            n_prompted += 1
            kind_counts[kind] += 1

    total = n_empty + n_prompted
    report = {
        "input_jsonl": args.input_jsonl,
        "output_jsonl": str(out_path),
        "n_input": n_in,
        "n_real": n_real,
        "n_gold_eligible": n_gold_eligible,
        "n_empty": n_empty,
        "n_prompted": n_prompted,
        "n_output": total,
        "prompted_row_frac": (n_prompted / total) if total else 0.0,
        "kind_counts": dict(kind_counts),
        "sampling_source_counts": dict(src_counts),
        "n_gold_catalog": len(catalog),
        "n_distractor_pool": len(pool),
    }
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
