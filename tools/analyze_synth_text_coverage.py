#!/usr/bin/env python3
"""Diagnose synthetic-text coverage against POC53 and the top hard clips.

For each source in the unified direct-mix manifest:
  - text diversity (rows, unique texts, voices/text, entropy, top 5-gram share)
  - row/unique-text coverage of the POC53 error phenomena
  - 8-gram coverage of POC53 references and of the top-N hard clips
Writes outputs/poc_unified7_text_coverage.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from evaluation.english_medical.text_normalization import normalize_english  # noqa: E402

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
ASCII_RE = re.compile(r"[A-Za-z]")
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
REAL_GROUPS = {"real_raw", "real_denoised", "real_0821_raw", "real_0821_denoised"}


def body(text: str) -> str:
    return (text or "").split("<asr_text>", 1)[-1].strip()


def tokens(text: str) -> list[str]:
    return normalize_english(text).split()


def ngrams(tokens_: list[str], n: int) -> list[str]:
    return [" ".join(tokens_[i:i + n]) for i in range(max(0, len(tokens_) - n + 1))]


def entropy(counts: collections.Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((v / total) * math.log2(v / total) for v in counts.values() if v)


def phenomenon_hits(raw: str, norm: str) -> dict[str, bool]:
    hits = {name: bool(pat.search(norm)) for name, pat in PATTERNS.items()}
    hits["all_not"] = bool(re.search(r"\ball\b", norm)) and bool(re.search(r"\bnot\b", norm))
    hits["cantonese"] = bool(CJK_RE.search(raw))
    hits["code_switch"] = bool(CJK_RE.search(raw)) and bool(ASCII_RE.search(raw))
    return hits


def wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    prev = list(range(len(h) + 1))
    for i, x in enumerate(r, 1):
        cur = [i]
        for j, y in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1] / max(len(r), 1)


def group_of(source: str) -> str:
    return "real_target_domain" if source in REAL_GROUPS else source


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train_jsonl", default="data/poc_train_unified7/direct_mix/train.jsonl")
    ap.add_argument("--dev_jsonl", default="data/poc_train_unified7/direct_mix/dev.jsonl")
    ap.add_argument("--predictions", default="outputs/poc_unified7_stage2_real75_1ep/eval_best/poc53_predictions.jsonl")
    ap.add_argument("--hard_top_n", type=int, default=8)
    ap.add_argument("--out", default="outputs/poc_unified7_text_coverage.json")
    args = ap.parse_args()

    dev = [json.loads(line) for line in open(args.dev_jsonl, encoding="utf-8") if line.strip()]
    dev_text = {row["audio"]: body(row["text"]) for row in dev}
    preds = [json.loads(line) for line in open(args.predictions, encoding="utf-8") if line.strip()]
    hard = sorted(
        ({"audio": p["audio"], "wer": wer(p["reference_normalized"], p["hypothesis_normalized"]),
          "text": dev_text[p["audio"]]} for p in preds if p["audio"] in dev_text),
        key=lambda item: item["wer"], reverse=True)[:args.hard_top_n]

    poc_tokens = {audio: tokens(text) for audio, text in dev_text.items()}
    poc_grams = {audio: set(ngrams(tokens_, 8)) for audio, tokens_ in poc_tokens.items()}
    poc_all_grams = set().union(*poc_grams.values())
    hard_grams = {item["audio"]: set(ngrams(tokens(item["text"]), 8)) for item in hard}
    hard_all_grams = set().union(*hard_grams.values())

    sources: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    for line in open(args.train_jsonl, encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        sources[group_of(row["sampling_source"])].append((body(row["text"]), row["audio"]))

    report = {
        "poc53": {"clips": len(dev_text),
                  "phenomena": {name: sum(1 for t in dev_text.values()
                                          if phenomenon_hits(t, normalize_english(t))[name])
                                for name in list(PATTERNS) + ["all_not", "cantonese", "code_switch"]}},
        "hard_clips": [{"audio": item["audio"].split("/")[-1], "wer": round(item["wer"], 3),
                        "text": item["text"][:200]} for item in hard],
        "sources": {},
    }

    for source, rows in sorted(sources.items()):
        unique_norm: dict[str, str] = {}
        uni = collections.Counter()
        tri = collections.Counter()
        five = collections.Counter()
        phen_rows = collections.Counter()
        phen_unique = collections.Counter()
        covered_poc_grams: set[str] = set()
        covered_hard_grams: set[str] = set()
        total_tokens = 0

        for raw, _audio in rows:
            norm = normalize_english(raw)
            unique_norm.setdefault(norm, raw)
            toks = norm.split()
            total_tokens += len(toks)
            uni.update(toks)
            tri.update(ngrams(toks, 3))
            five.update(ngrams(toks, 5))
            for g in ngrams(toks, 8):
                if g in poc_all_grams:
                    covered_poc_grams.add(g)
                if g in hard_all_grams:
                    covered_hard_grams.add(g)
            for name, hit in phenomenon_hits(raw, norm).items():
                if hit:
                    phen_rows[name] += 1

        for norm, raw in unique_norm.items():
            for name, hit in phenomenon_hits(raw, norm).items():
                if hit:
                    phen_unique[name] += 1

        n = len(rows)
        top5 = five.most_common(5)
        top5_rows = sum(v for _, v in top5)
        top50_share = sum(v for _, v in five.most_common(50)) / max(sum(five.values()), 1)
        poc_covered_clips = sum(1 for gs in poc_grams.values() if gs & covered_poc_grams)
        hard_covered_clips = sum(1 for gs in hard_grams.values() if gs & covered_hard_grams)

        report["sources"][source] = {
            "rows": n,
            "unique_texts": len(unique_norm),
            "rows_per_unique": round(n / max(len(unique_norm), 1), 2),
            "avg_tokens": round(total_tokens / max(n, 1), 1),
            "unigram_entropy": round(entropy(uni), 2),
            "trigram_entropy": round(entropy(tri), 2),
            "top5gram_row_share": round(top5_rows / max(n, 1), 3),
            "top5grams": [{"gram": g, "rows": v} for g, v in top5],
            "top50_5gram_token_share": round(top50_share, 3),
            "poc53_gram_coverage": round(len(covered_poc_grams) / max(len(poc_all_grams), 1), 3),
            "poc53_clip_coverage": round(poc_covered_clips / len(poc_grams), 3),
            "hard_gram_coverage": round(len(covered_hard_grams) / max(len(hard_all_grams), 1), 3),
            "hard_clip_coverage": round(hard_covered_clips / max(len(hard_grams), 1), 3),
            "phenomena_rows_pct": {k: round(100 * v / max(n, 1), 2) for k, v in sorted(phen_rows.items())},
            "phenomena_unique_pct": {k: round(100 * v / max(len(unique_norm), 1), 2)
                                     for k, v in sorted(phen_unique.items())},
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("POC53 phenomena clips:", report["poc53"]["phenomena"])
    print("\nhard clips:", [h["audio"] for h in report["hard_clips"]])
    print(f"\n{'source':28s} {'rows':>7s} {'uniq':>7s} {'r/u':>5s} {'POC53g':>7s} {'POC53c':>7s} "
          f"{'hardg':>6s} {'hardc':>6s} {'top5%':>6s} {'H1':>5s} {'H3':>6s}")
    for name, s in report["sources"].items():
        print(f"{name:28s} {s['rows']:7d} {s['unique_texts']:7d} {s['rows_per_unique']:5.2f} "
              f"{s['poc53_gram_coverage']:7.3f} {s['poc53_clip_coverage']:7.3f} "
              f"{s['hard_gram_coverage']:6.3f} {s['hard_clip_coverage']:6.3f} "
              f"{s['top5gram_row_share']:6.3f} {s['unigram_entropy']:5.2f} {s['trigram_entropy']:6.2f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
