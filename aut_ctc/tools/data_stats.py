"""M2 data report (DESIGN.md §8.4), optionally writing the filtered manifests.

  python -m aut_ctc.tools.data_stats --sp_model /path/to/chn_jpn_yue_eng_ko_spectok.bpe.model \
      --manifests data/train_zh.jsonl data/train_en.jsonl --report exp/data_report.json --filtered_dir data/filtered

The same prepare_example() as training decides what is kept, so the counts match the training run.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..data import load_manifest, prepare_example
from ..tokenizer import load_tokenizer

DURATION_BINS = [0, 0.3, 1, 2, 5, 10, 20, 30, 60, float("inf")]


def manifest_report(path, tokenizer, args, filtered_out=None) -> dict:
    reasons, hours, ratios = Counter(), Counter(), defaultdict(list)
    durations, tokens, unk, raw_digit, lost_boundaries = [], 0, 0, 0, 0
    for ex in load_manifest(path):
        ids, reason, info = prepare_example(ex, tokenizer, args.style, args.min_duration, args.max_duration,
                                            drop_unk=False, upsample=args.upsample, num_queries=args.num_queries)
        if ids is not None and args.drop_unk and info["num_unk"]:
            ids, reason = None, "unk"
        reasons[reason] += 1
        hours[reason] += info["duration"] / 3600
        durations.append(info["duration"])
        raw_digit += info["raw_has_digit"]
        if "text" in info:
            tokens += info["num_tokens"]
            unk += info["num_unk"]
            lost_boundaries += tokenizer.dropped_boundary_count(info["text"])
        if "T" in info and info["U'"]:
            ratios[ex.get("lang") or "unknown"].append(info["T"] / info["U'"])
        if ids is not None and filtered_out is not None:
            filtered_out.write(json.dumps(ex, ensure_ascii=False) + "\n")
    hist, _ = np.histogram(durations, bins=DURATION_BINS)
    t_over_u = {}
    for lang, r in ratios.items():
        r = np.asarray(r)
        t_over_u[lang] = {"n": len(r), "p1": float(np.percentile(r, 1)), "p5": float(np.percentile(r, 5)),
                          "p50": float(np.percentile(r, 50)), "share_below_1.5": float((r < 1.5).mean())}
    all_r = np.concatenate([np.asarray(r) for r in ratios.values()]) if ratios else np.zeros(0)
    return {
        "utts": len(durations),
        "hours": round(sum(durations) / 3600, 3),
        "kept_utts": reasons["ok"] + reasons["ok_empty"],
        "kept_hours": round(hours["ok"] + hours["ok_empty"], 3),
        "filter_reasons": dict(reasons),
        "filter_hours": {k: round(v, 3) for k, v in hours.items()},
        "duration_hist": {f"{lo}-{hi}s": int(c) for lo, hi, c in zip(DURATION_BINS, DURATION_BINS[1:], hist)},
        "oov_rate": unk / max(tokens, 1),
        "raw_digit_ratio": raw_digit / max(len(durations), 1),
        "lost_word_boundaries": lost_boundaries,  # isolated ▁ before a non-Han piece (DESIGN §6.6)
        "T_over_U": t_over_u,
        # DESIGN §8.4: switch to upsample=2 when more than 1% of samples have T/U' < 1.5
        "suggest_upsample_2": bool(len(all_r) and (all_r < 1.5).mean() > 0.01),
    }


def main():
    p = argparse.ArgumentParser("AuT-CTC M2 data report")
    p.add_argument("--sp_model", required=True)
    p.add_argument("--manifests", nargs="+", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--filtered_dir", default=None, help="write the kept entries of each manifest here")
    p.add_argument("--style", default="woitn")
    p.add_argument("--min_duration", type=float, default=0.3)
    p.add_argument("--max_duration", type=float, default=30.0)
    p.add_argument("--drop_unk", type=int, default=1)
    p.add_argument("--upsample", type=int, default=1)
    p.add_argument("--num_queries", type=int, default=0)
    args = p.parse_args()

    tokenizer = load_tokenizer(args.sp_model)
    report = {"vocab_size": tokenizer.vocab_size, "manifests": {}}
    for path in args.manifests:
        out = None
        if args.filtered_dir:
            Path(args.filtered_dir).mkdir(parents=True, exist_ok=True)
            out = open(Path(args.filtered_dir) / Path(path).name, "w", encoding="utf-8")
        report["manifests"][path] = manifest_report(path, tokenizer, args, out)
        if out:
            out.close()
        print(json.dumps({path: report["manifests"][path]}, ensure_ascii=False, indent=1), flush=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
