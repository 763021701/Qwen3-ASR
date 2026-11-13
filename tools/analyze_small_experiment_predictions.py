#!/usr/bin/env python3
"""Compare prediction patterns of the three small-data probes on POC53.

Reads eval_best/poc53_predictions.jsonl and eval_best/summary.json from:
    outputs/poc_small_realraw_noaug_1ep
    outputs/poc_small_realdenoise_aug_1ep
    outputs/poc_small_synth_ab_aug_1ep
Writes outputs/poc_small_experiments_analysis.json and prints a compact report.
"""
from __future__ import annotations

import argparse
import collections
import difflib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = {
    "realraw_noaug": "outputs/poc_small_realraw_noaug_1ep",
    "realdenoise_aug": "outputs/poc_small_realdenoise_aug_1ep",
    "synth_ab_aug": "outputs/poc_small_synth_ab_aug_1ep",
}
PATTERNS = {
    "specimen_id": re.compile(r"\b\d{4,}\b"),
    "sn": re.compile(r"\bsn\s*\d|\bsn\b"),
    "block_letter": re.compile(r"\bblock\s*[a-z]\b|\bblocks?\s+[a-z]\s+to\s+[a-z]\b"),
    "frozen": re.compile(r"\bfrozen\b"),
    "paraffin": re.compile(r"\bparaffin\b"),
    "coronal": re.compile(r"\bcoronal\b|full slab"),
    "margins": re.compile(r"\bmargin|perpendicular"),
    "measurement": re.compile(r"\d+(?:\.\d+)?\s*(?:cm|mm|gram|g)\b"),
    "specimen_type": re.compile(
        r"hemithyroidectomy|lumpectomy|hysterectomy|sentinel node|mastectomy|nephrectomy"),
    "orientation": re.compile(r"orientation|stitch"),
}
CJK = re.compile(r"[\u4e00-\u9fff]")


def load(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    return {row["audio"]: row for row in (json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip())}


def edit_distance(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def ref_hyp(row: dict) -> tuple[str, str]:
    return row["reference_normalized"], row["hypothesis_normalized"]


def corpus(rows: dict[str, dict], char: bool = False) -> float:
    errors = total = 0
    for row in rows.values():
        ref, hyp = ref_hyp(row)
        if char:
            ref, hyp = ref.replace(" ", ""), hyp.replace(" ", "")
            errors += edit_distance(list(ref), list(hyp))
            total += len(ref)
        else:
            ref, hyp = ref.split(), hyp.split()
            errors += edit_distance(ref, hyp)
            total += len(ref)
    return round(100.0 * errors / max(total, 1), 2)


def clip_wer(row: dict) -> float:
    ref, hyp = ref_hyp(row)
    ref, hyp = ref.split(), hyp.split()
    return edit_distance(ref, hyp) / max(len(ref), 1)


def ops(row: dict):
    ref, hyp = ref_hyp(row)
    a, b = ref.split(), hyp.split()
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    inserted, deleted, replaced = [], [], []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            inserted += b[j1:j2]
        elif tag == "delete":
            deleted += a[i1:i2]
        elif tag == "replace":
            replaced += list(zip(a[i1:i2], b[j1:j2]))
    return inserted, deleted, replaced


def phenomena(raw: str, norm: str) -> set[str]:
    hits = {name for name, pattern in PATTERNS.items() if pattern.search(norm)}
    if CJK.search(raw):
        hits.add("cantonese")
    return hits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/poc_small_experiments_analysis.json")
    args = ap.parse_args()
    models = {}
    for name, run in RUNS.items():
        preds = load(REPO / run / "eval_best/poc53_predictions.jsonl")
        summary_path = REPO / run / "eval_best/summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        models[name] = {"preds": preds, "summary": summary}
        print(f"[load] {name}: rows={len(preds)} summary_sets={list(summary.get('sets', {}))}")

    common = set.intersection(*(set(m["preds"]) for m in models.values() if m["preds"]))
    print(f"[align] common clips={len(common)}")
    if not common:
        raise SystemExit("no common POC53 predictions")

    report = {"clips": len(common), "models": {}}
    wers = {}
    for name, model in models.items():
        preds = {a: r for a, r in model["preds"].items() if a in common}
        wers[name] = {a: clip_wer(r) for a, r in preds.items()}
        ins = collections.Counter()
        dele = collections.Counter()
        sub = collections.Counter()
        modes = collections.Counter()
        for row in preds.values():
            i, d, s = ops(row)
            ins.update(i)
            dele.update(d)
            sub.update(f"{a}->{b}" for a, b in s)
            if len(i) >= 3:
                modes["insertion"] += 1
            if len(d) >= 3:
                modes["deletion"] += 1
            if len(s) >= 3:
                modes["substitution"] += 1
            if len(row["hypothesis_normalized"].split()) < 0.6 * len(row["reference_normalized"].split()):
                modes["truncation"] += 1
            if re.search(r"(\b\w+\b)( \1){2,}", row["hypothesis_normalized"]):
                modes["loop"] += 1
            if CJK.search(row["reference"]) and not CJK.search(row["hypothesis"]):
                modes["cantonese_drop"] += 1
            if any(any(ch.isdigit() for ch in token) for token in i + d + [x for pair in s for x in pair]):
                modes["numeric"] += 1
        hard = sorted(((value, audio) for audio, value in wers[name].items() if value >= 0.2),
                      reverse=True)
        report["models"][name] = {
            "poc53_wer": corpus(preds),
            "poc53_cer": corpus(preds, char=True),
            "summary_sets": model["summary"].get("sets", {}),
            "modes": dict(sorted(modes.items())),
            "hard_clips": [{"audio": Path(a).name, "wer": round(v, 3)} for v, a in hard],
            "top_insertions": ins.most_common(10),
            "top_deletions": dele.most_common(10),
            "top_substitutions": sub.most_common(12),
            "categories": {},
        }
        for category in list(PATTERNS) + ["cantonese"]:
            errors = total = 0
            for row in preds.values():
                ref, hyp = ref_hyp(row)
                if category in phenomena(row["reference"], ref):
                    ref_tokens, hyp_tokens = ref.split(), hyp.split()
                    errors += edit_distance(ref_tokens, hyp_tokens)
                    total += len(ref_tokens)
            if total:
                report["models"][name]["categories"][category] = {
                    "wer": round(100.0 * errors / total, 1), "ref_tokens": total}

    names = list(models)
    pairs = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            wins = sum(wers[a][c] < wers[b][c] - 1e-9 for c in common)
            ties = sum(abs(wers[a][c] - wers[b][c]) < 1e-9 for c in common)
            losses = len(common) - wins - ties
            pairs[f"{a}_vs_{b}"] = {"wins": wins, "ties": ties, "losses": losses}
    report["pairwise_clip_wer"] = pairs

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\nPOC53:")
    for name in names:
        m = report["models"][name]
        print(f"  {name:18s} WER={m['poc53_wer']:6.2f} CER={m['poc53_cer']:6.2f} "
              f"hard={len(m['hard_clips']):2d} modes={m['modes']}")
    print("\nPairwise wins/ties/losses:", json.dumps(pairs, ensure_ascii=False))
    for category in list(PATTERNS) + ["cantonese"]:
        vals = {name: report["models"][name]["categories"].get(category, {}).get("wer")
                for name in names}
        if any(v is not None for v in vals.values()):
            print(f"  {category:14s} " + " ".join(
                f"{name}={vals[name]}" for name in names))
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
