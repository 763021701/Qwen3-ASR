#!/usr/bin/env python3
"""Rescore saved radiology predictions with the CURRENT normalize_english (round-2).

Saved *_normalized fields use whatever normalizer was active at training time, so both
runs must be re-normalized from raw reference/hypothesis. Reports corpus WER per model
and a per-clip diff (same 95 audios) between the plus0821 champion (step-3900) and the
yue_ba_sl run (step-2000), plus digit/CJK-token error decomposition.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "evaluation"))

from english_medical.text_normalization import normalize_english  # noqa: E402
from masr_eval_pkg import compute_wer  # noqa: E402

P3900 = REPO / "outputs/poc_train_real_llm_syn_cantonese_num_measure_tts_pathology_full_testdev_cer_sft_nodual_curated_fullvoice_clean_v2synth_yt_plus0821_fromscratch600/generation_eval/step-3900_radiology_predictions.jsonl"
PNEW = REPO / "outputs/poc_train_yue_ba_sl_no_yt_plus0821_fromscratch/generation_eval"


def has_digit_or_cjk(token: str) -> bool:
    return any(ch.isdigit() or "一" <= ch <= "鿿" for ch in token)


def score(path: Path) -> dict[str, dict]:
    per: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        ref = normalize_english(r["reference"])
        hyp = normalize_english(r["hypothesis"])
        w = compute_wer([ref], [hyp])
        per[r["audio"]] = {
            "wer": w["wer"], "n": w["n_ref_tokens"],
            "s": w["substitutions"], "d": w["deletions"], "i": w["insertions"],
            "ref": ref, "hyp": hyp,
        }
    return per


def corpus(per: dict[str, dict]) -> dict:
    s = d = i = n = 0
    exact = 0
    for v in per.values():
        s += v["s"]; d += v["d"]; i += v["i"]; n += v["n"]
        if v["wer"] == 0.0:
            exact += 1
    return {"wer": (s + d + i) / n if n else 0.0, "s": s, "d": d, "i": i,
            "n": n, "exact": exact, "clips": len(per)}


def digit_stats(per: dict[str, dict]) -> str:
    hits = {"ref": 0, "hyp": 0, "wrong": 0}
    for v in per.values():
        ref_t = v["ref"].split()
        for tok in v["hyp"].split():
            if has_digit_or_cjk(tok):
                hits["hyp"] += 1
                if tok not in ref_t:
                    hits["wrong"] += 1
        hits["ref"] += sum(1 for t in ref_t if has_digit_or_cjk(t))
    return f"digit/CJK hyp tokens {hits['hyp']} (not-in-ref {hits['wrong']}, {hits['wrong'] / max(hits['hyp'], 1):.0%}); ref pool {hits['ref']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new_steps", default="1300,1900,2000,4557")
    args = ap.parse_args()

    champ = score(P3900)
    c = corpus(champ)
    print(f"== plus0821 ckpt-3900 (champion): WER {c['wer']:.2%}  S/D/I {c['s']}/{c['d']}/{c['i']}  n={c['n']}  exact {c['exact']}/{c['clips']}")
    print(f"   {digit_stats(champ)}")

    news = {}
    for step in args.new_steps.split(","):
        per = score(PNEW / f"step-{step}_radiology_predictions.jsonl")
        assert set(per) == set(champ), f"audio set mismatch at step {step}"
        news[step] = per
        k = corpus(per)
        print(f"== yue_ba_sl ckpt-{step}: WER {k['wer']:.2%}  S/D/I {k['s']}/{k['d']}/{k['i']}  exact {k['exact']}/{k['clips']}")
        print(f"   {digit_stats(per)}")

    best = min(news, key=lambda s: corpus(news[s])["wer"])
    per = news[best]
    print(f"\n== best new step on radiology: ckpt-{best}; per-clip diff vs champion (ΔWER>2pp):")
    rows = []
    for a in champ:
        d = per[a]["wer"] - champ[a]["wer"]
        if abs(d) > 0.02:
            rows.append((d, a, champ[a], per[a]))
    rows.sort()
    for d, a, old, new in rows:
        print(f"  {d:+.1%} {Path(a).name}")
        print(f"     ref : {old['ref'][:95]}")
        print(f"     3900: {old['hyp'][:95]}")
        print(f"     new : {new['hyp'][:95]}")


if __name__ == "__main__":
    main()
