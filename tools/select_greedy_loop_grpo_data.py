"""Loop-aware GRPO manifest selection for --advantage_mode greedy.

Reads a mining report (tools/mine_qwen3_asr_grpo_data.py) and the source
manifest, computes per-clip loop activity with the SAME penalty the trainer
reward uses (finetuning.qwen3_asr_grpo.loop_penalty_ratio), and emits an
oversampled training manifest.

Greedy-baseline advantages A_i = reward_i - reward_greedy do not require
within-group reward variance, so the 4-class CER taxonomy is re-mixed:
loop-active clips from every category (except suspect without a clean
rollout, where the label itself is suspect) join the pool.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from finetuning.qwen3_asr_grpo import loop_penalty_ratio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mining_report", required=True, help="mining_report.jsonl")
    parser.add_argument("--input_jsonl", required=True, help="Source manifest (supplies text/audio fields)")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--weight_hard_loop", type=int, default=4, help="hard AND loop-active: the core mass")
    parser.add_argument("--weight_hard", type=int, default=2, help="hard without loops: CER signal mass")
    parser.add_argument("--weight_easy_loop", type=int, default=2, help="easy BUT loop-active (sporadic loopers)")
    parser.add_argument("--weight_anchored_loop", type=int, default=2,
                        help="(weak|suspect) AND loop-active AND a clean rollout exists")
    parser.add_argument("--loop_pen_min", type=float, default=0.2,
                        help="A clip is loop-active when any rollout's loop_pen coverage >= this.")
    parser.add_argument("--clean_cer_max", type=float, default=0.30,
                        help="A clip has a clean rollout when min(group CER) <= this (same bar as capable_best_max).")
    parser.add_argument("--dedupe_by_audio", type=int, default=1, choices=(0, 1),
                        help="Keep the first selected row per exact audio path (drops raw/denoised twins).")
    return parser.parse_args()


def loop_penalties(reference: str, hypotheses: List[str]) -> List[float]:
    return [loop_penalty_ratio(hyp, reference) for hyp in hypotheses]


def decide_weight(
    category: str,
    loop_active: bool,
    clean_exists: bool,
    args: argparse.Namespace,
) -> int:
    if category == "hard":
        return args.weight_hard_loop if loop_active else args.weight_hard
    if category == "easy":
        return args.weight_easy_loop if loop_active else 0
    if category in ("weak", "suspect"):
        return args.weight_anchored_loop if (loop_active and clean_exists) else 0
    return 0


def select_records(report_rows: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Attach per-record loop stats and selection weight to each report row (in input order)."""
    out = []
    for record in report_rows:
        pens = loop_penalties(str(record["reference"]), [str(h) for h in record["hypotheses"]])
        loop_active = bool(pens) and max(pens) >= args.loop_pen_min
        cers = [float(c) for c in record["cers"]]
        clean_exists = bool(cers) and min(cers) <= args.clean_cer_max
        category = str(record["category"])
        weight = decide_weight(category, loop_active, clean_exists, args)
        out.append({
            "record": record,
            "loop_pen_max": max(pens) if pens else 0.0,
            "loop_active": loop_active,
            "clean_exists": clean_exists,
            "weight": weight,
        })
    return out


def main() -> None:
    args = parse_args()
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        source_rows = [json.loads(line) for line in f if line.strip()]
    with open(args.mining_report, "r", encoding="utf-8") as f:
        report_rows = [json.loads(line) for line in f if line.strip()]
    by_index = {int(r["miner_source_index"]): r for r in report_rows}
    if len(by_index) != len(report_rows):
        raise ValueError("mining report contains duplicate miner_source_index values")
    if len(by_index) < len(source_rows):
        print(f"[warn] incomplete report: {len(by_index)}/{len(source_rows)} rows mined", flush=True)
    ordered = [by_index[i] for i in sorted(by_index)]
    for record in ordered:
        i = int(record["miner_source_index"])
        if record["audio"] != source_rows[i]["audio"]:
            raise ValueError(f"report/source audio mismatch at index {i}: {record['audio']!r}")

    selected = select_records(ordered, args)

    matrix: Counter[str] = Counter()
    for item in selected:
        tag = f"{'loop_' if item['loop_active'] else ''}{item['record']['category']}"
        matrix[f"{tag}:{'clean' if item['clean_exists'] else 'dirty'}"] += 1

    out_rows: List[Dict[str, Any]] = []
    seen_audio = set()
    for item in selected:
        if item["weight"] <= 0:
            continue
        record = item["record"]
        if args.dedupe_by_audio:
            audio = str(record["audio"])
            if audio in seen_audio:
                continue
            seen_audio.add(audio)
        row = dict(source_rows[int(record["miner_source_index"])])
        row["grpo_category"] = record["category"]
        row["grpo_loop_pen_max"] = round(item["loop_pen_max"], 6)
        row["grpo_clean_exists"] = int(item["clean_exists"])
        for _ in range(item["weight"]):
            out_rows.append(dict(row))

    parent = os.path.dirname(os.path.abspath(args.output_jsonl))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[categories] {dict(sorted(matrix.items()))}")
    n_kept = sum(1 for it in selected if it["weight"] > 0)
    print(
        f"[done] mined={len(selected)} clips kept={n_kept} "
        f"rows={len(out_rows)} unique_audio={len({str(r['audio']) for r in out_rows})} "
        f"-> {args.output_jsonl}"
    )


if __name__ == "__main__":
    main()
