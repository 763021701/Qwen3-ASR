#!/usr/bin/env python3
"""Classify and compare per-token errors on the 53-clip POC test between two checkpoints.

Both prediction files are re-normalized with the CURRENT normalize_english (saved
*_normalized fields may be stale). Tokens are aligned per clip with difflib; every S/D/I
is bucketed by token nature so the two models' error *mixes* are comparable, and the
per-clip WER delta shows which clips moved.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "evaluation"))

from english_medical.text_normalization import normalize_english  # noqa: E402
from masr_eval_pkg import compute_wer  # noqa: E402

CJK_RE = re.compile(r"[㐀-䶿一-鿿]")
NUM_RE = re.compile(r"\d")
NUM_WORDS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
}
UNIT_WORDS = {"cm", "mm", "um", "nm", "g", "kg", "ml", "cc", "micron", "microns", "gram", "grams"}
ID_RE = re.compile(r"^\d*[a-z]*\d{3,}$|^\d{2}\s?ss")
STOP_WORDS = {"period", "comma", "stop"}


def classify(token: str) -> str:
    if CJK_RE.search(token):
        return "cjk"
    if NUM_RE.search(token):
        return "specimen_id" if ID_RE.match(token) else "numeric"
    if token in NUM_WORDS:
        return "numeric_word"
    if token in UNIT_WORDS:
        return "unit"
    if token in STOP_WORDS:
        return "discourse_marker"
    if len(token) <= 2 and token.isalpha():
        return "single_letter"
    return "word"


def load(path: Path) -> dict[str, tuple[str, str]]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[r["audio"]] = (normalize_english(r["reference"]), normalize_english(r["hypothesis"]))
    return out


def bucket_errors(data: dict[str, tuple[str, str]]) -> tuple[Counter, Counter, dict]:
    """Returns (S+D by category, I by category, per-clip WER)."""
    sub_del, ins, per_clip = Counter(), Counter(), {}
    for audio, (ref, hyp) in data.items():
        rt, ht = ref.split(), hyp.split()
        w = compute_wer([ref], [hyp])
        per_clip[audio] = w["wer"]
        sm = difflib.SequenceMatcher(a=rt, b=ht, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            if tag in ("replace", "delete"):
                for t in rt[i1:i2]:
                    sub_del[classify(t)] += 1
            if tag in ("replace", "insert"):
                # attributions of inserted tokens use the hypothesis side
                for t in ht[j1:j2]:
                    ins[classify(t)] += 1
        if w["n_ref_tokens"] == 0:
            continue
    return sub_del, ins, per_clip


CATS = ["cjk", "numeric", "numeric_word", "unit", "specimen_id", "single_letter", "discourse_marker", "word"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="outputs/eval_poc_fullvoice_yt_plus0821/checkpoint-3900/predictions.jsonl",
                    help="reference model predictions (champion)")
    ap.add_argument("--b", default="outputs/eval_poc_yue_ba_sl/checkpoint-2000/predictions.jsonl",
                    help="candidate model predictions")
    ap.add_argument("--a_name", default="champ-3900")
    ap.add_argument("--b_name", default="yue_ba-2000")
    args = ap.parse_args()

    da, db = load(REPO / args.a), load(REPO / args.b)
    assert set(da) == set(db), "prediction files cover different audio sets"

    sda, ina, wa = bucket_errors(da)
    sdb, ins_b, wb = bucket_errors(db)
    tot_a, tot_b = sum(sda.values()) + sum(ina.values()), sum(sdb.values()) + sum(ins_b.values())

    print(f"{'category':16} {args.a_name+' S/D':>12} {args.a_name+' I':>8} {'%':>6}   {args.b_name+' S/D':>12} {args.b_name+' I':>8} {'%':>6}")
    for c in CATS:
        a1, a2, b1, b2 = sda[c], ina[c], sdb[c], ins_b[c]
        print(f"{c:16} {a1:12d} {a2:8d} {100*(a1+a2)/max(tot_a,1):5.1f}%   {b1:12d} {b2:8d} {100*(b1+b2)/max(tot_b,1):5.1f}%")
    print(f"{'TOTAL':16} {sum(sda.values()):12d} {sum(ina.values()):8d}          {sum(sdb.values()):12d} {sum(ins_b.values()):8d}")

    # CJK-specific view: how many CJK ref tokens did each model recover?
    for name, d in ((args.a_name, da), (args.b_name, db)):
        ref_cjk = hyp_cjk = hit = 0
        for ref, hyp in d.values():
            rt = [t for t in ref.split() if CJK_RE.search(t)]
            ht = [t for t in hyp.split() if CJK_RE.search(t)]
            ref_cjk += len(rt)
            hyp_cjk += len(ht)
            hit += sum(1 for t in rt if t in set(hyp.split()))
        print(f"{name}: CJK ref tokens {ref_cjk}, output {hyp_cjk}, 命中 {hit} ({hit/max(ref_cjk,1):.0%})")

    # per-clip movement
    rows = sorted(((wb[a] - wa[a], a, wa[a], wb[a]) for a in da), key=lambda x: x[0])
    better = [r for r in rows if r[0] < -0.01]
    worse = [r for r in rows if r[0] > 0.01]
    print(f"\nclip 变动: 变好 {len(better)} 条 / 变差 {len(worse)} 条 / 其余持平; "
          f"WER {sum(wa.values())/len(wa):.2%} -> {sum(wb.values())/len(wb):.2%} (per-clip 均值)")
    for tag, sel in (("  TOP 变好", better[:4]), ("  TOP 变差", worse[-4:][::-1])):
        print(tag)
        for dlt, a, x, y in sel:
            print(f"    {dlt:+.1%} {x:.1%}->{y:.1%} {Path(a).name}")
            print(f"      ref : {da[a][0][:110]}")
            print(f"      {args.a_name}: {da[a][1][:110]}")
            print(f"      {args.b_name}: {db[a][1][:110]}")


if __name__ == "__main__":
    main()
