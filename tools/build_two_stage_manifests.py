#!/usr/bin/env python3
"""Split the balanced POC manifest into nested two-stage training manifests.

Stage 1 (synth-dominant): all synth rows + real rows capped at 20% of audio hours
         (all real_target_domain, remaining hours sampled from Wenet/YouTube).
Stage 2 (real-dominant): all real rows + synth rows capped at 25% of audio hours
         (sampled proportionally from pathology/radiology/llm/supp).

Stage-1 real rows are a subset of stage-2 real; stage-2 synth rows are a subset of
stage-1 synth, so the second stage is a strict domain-shift curriculum.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SYNTH = {"tts_pathology_vox", "tts_pathology_supp", "tts_radiology", "llm_text_syn"}
REAL_TARGET = {"real_raw", "real_denoised", "real_0821_raw", "real_0821_denoised"}


def hours(rows: list[dict]) -> float:
    return sum(float(row["duration_sec"]) for row in rows) / 3600.0


def take_hours(rows: list[dict], target_hours: float, seed: int) -> list[dict]:
    order = list(rows)
    random.Random(seed).shuffle(order)
    picked, total = [], 0.0
    for row in order:
        if total >= target_hours:
            break
        picked.append(row)
        total += float(row["duration_sec"]) / 3600.0
    return picked


def scale_sources(rows: list[dict], target_hours: float, seed: int) -> tuple[list[dict], dict]:
    by_source = collections.defaultdict(list)
    for row in rows:
        by_source[row["sampling_source"]].append(row)
    pool_hours = {source: hours(items) for source, items in by_source.items()}
    total_pool = sum(pool_hours.values())
    picked, detail = [], {}
    for index, (source, items) in enumerate(sorted(by_source.items())):
        target = target_hours * pool_hours[source] / max(total_pool, 1e-9)
        chosen = take_hours(items, target, seed + index)
        picked += chosen
        detail[source] = {"pool_hours": round(pool_hours[source], 3),
                          "target_hours": round(target, 3),
                          "picked_rows": len(chosen),
                          "picked_hours": round(hours(chosen), 3)}
    return picked, detail


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dir", default="data/poc_train_unified_v2/dist_balanced")
    ap.add_argument("--out-root", default="data/poc_train_unified_v2/two_stage")
    ap.add_argument("--stage1-real-share", type=float, default=0.20)
    ap.add_argument("--stage2-synth-share", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=20261112)
    args = ap.parse_args()

    source_dir = Path(args.source_dir).resolve()
    out_root = Path(args.out_root).resolve()
    rows = [json.loads(line) for line in (source_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    synth = [row for row in rows if row["sampling_source"] in SYNTH]
    real = [row for row in rows if row["sampling_source"] not in SYNTH]
    real_target = [row for row in real if row["sampling_source"] in REAL_TARGET]
    other_real = [row for row in real if row["sampling_source"] not in REAL_TARGET]
    synth_h, real_h = hours(synth), hours(real)

    stage1_real_target_h = synth_h * args.stage1_real_share / (1.0 - args.stage1_real_share)
    stage1_other_h = max(0.0, stage1_real_target_h - hours(real_target))
    stage1_other, stage1_detail = scale_sources(other_real, stage1_other_h, args.seed + 1)
    stage1 = list(synth) + list(real_target) + stage1_other

    stage2_synth_target_h = real_h * args.stage2_synth_share / (1.0 - args.stage2_synth_share)
    stage2_synth, stage2_detail = scale_sources(synth, stage2_synth_target_h, args.seed + 2)
    stage2 = list(real) + stage2_synth

    stage1_ids = {row["segment_id"] for row in stage1}
    stage2_ids = {row["segment_id"] for row in stage2}
    stage1_synth_ids = {row["segment_id"] for row in stage1 if row["sampling_source"] in SYNTH}
    stage2_synth_ids = {row["segment_id"] for row in stage2 if row["sampling_source"] in SYNTH}
    stage1_real_ids = {row["segment_id"] for row in stage1 if row["sampling_source"] not in SYNTH}
    stage2_real_ids = {row["segment_id"] for row in stage2 if row["sampling_source"] not in SYNTH}
    if stage1_synth_ids != {row["segment_id"] for row in synth}:
        raise RuntimeError("stage1 must contain the full synth pool")
    if not stage2_synth_ids <= stage1_synth_ids:
        raise RuntimeError("stage2 synth is not a subset of stage1 synth")
    if stage2_real_ids != {row["segment_id"] for row in real}:
        raise RuntimeError("stage2 must contain the full real pool")
    if not stage1_real_ids <= stage2_real_ids:
        raise RuntimeError("stage1 real is not a subset of stage2 real")

    for name, split in (("stage1_synth_dominant", stage1), ("stage2_real_dominant", stage2)):
        directory = out_root / name
        random.Random(args.seed + 10).shuffle(split)
        write_jsonl(directory / "train.jsonl", split)
        for part in ("dev", "test"):
            (directory / f"{part}.jsonl").write_bytes((source_dir / f"{part}.jsonl").read_bytes())

    def summary(items: list[dict]) -> dict:
        by_source = collections.defaultdict(lambda: {"rows": 0, "hours": 0.0})
        for row in items:
            entry = by_source[row["sampling_source"]]
            entry["rows"] += 1
            entry["hours"] += float(row["duration_sec"]) / 3600.0
        for entry in by_source.values():
            entry["hours"] = round(entry["hours"], 3)
        synth_hours = sum(entry["hours"] for source, entry in by_source.items() if source in SYNTH)
        total = sum(entry["hours"] for entry in by_source.values())
        return {"rows": len(items), "hours": round(total, 3),
                "synth_hours": round(synth_hours, 3), "real_hours": round(total - synth_hours, 3),
                "by_source": dict(sorted(by_source.items()))}

    report = {
        "source": str(source_dir / "train.jsonl"),
        "seed": args.seed,
        "ratios": {"stage1_real_share": args.stage1_real_share,
                   "stage2_synth_share": args.stage2_synth_share},
        "source_totals": {"synth_hours": round(synth_h, 3), "real_hours": round(real_h, 3)},
        "stage1": {"summary": summary(stage1), "real_sample_detail": stage1_detail,
                   "real_target_hours": round(hours(real_target), 3)},
        "stage2": {"summary": summary(stage2), "synth_sample_detail": stage2_detail},
        "nesting": {
            "stage1_synth_is_full": True,
            "stage2_synth_subset_stage1": len(stage2_synth_ids) == len(stage2_synth_ids & stage1_synth_ids),
            "stage2_real_is_full": True,
            "stage1_real_subset_stage2": len(stage1_real_ids) == len(stage1_real_ids & stage2_real_ids),
            "stage1_total_ids": len(stage1_ids), "stage2_total_ids": len(stage2_ids),
        },
    }
    (out_root / "stage_stats.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    for name in ("stage1", "stage2"):
        item = report[name]["summary"]
        print(f"[{name}] rows={item['rows']} hours={item['hours']:.2f} "
              f"synth={item['synth_hours']:.2f} real={item['real_hours']:.2f}")
    print(f"[done] {out_root}")


if __name__ == "__main__":
    main()
