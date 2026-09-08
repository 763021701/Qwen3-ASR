#!/usr/bin/env python3
"""Build a reachable-hard RAFT training set from a mining report (offline).

The mining report already stores each clip's G sampled completions and the
training reference. For clips the model solves only sometimes (unstable /
recoverable), pick the sampled completion closest to the reference; keep the
clip when that best sample's normalized CER is under --max_cer and emit the
RAW completion (with its own language prefix) as the target, so continued SFT
distills the model's own rarely-greedy success. Optionally mix in stabilizer
rows (reference targets) sampled from an existing manifest.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.english_medical.text_normalization import normalize_english
from qwen_asr.inference.utils import parse_asr_output


def char_cer(reference: str, hypothesis: str) -> float:
    ref = normalize_english(reference).replace(" ", "")
    hyp = normalize_english(hypothesis).replace(" ", "")
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(ref)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mining_report", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--categories", default="unstable,recoverable")
    parser.add_argument("--max_cer", type=float, default=0.15)
    parser.add_argument("--raft_weight", type=int, default=3)
    parser.add_argument("--stabilizer_manifest", default="",
                        help="Optional manifest to sample reference-target rows from.")
    parser.add_argument("--stabilizer_rows", type=int, default=1500)
    parser.add_argument("--flags_manifest", default="",
                        help="Manifest used to inherit per-row aug/noise_aug flags by audio path.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    wanted = {c.strip() for c in args.categories.split(",") if c.strip()}
    kept, skipped_hard = [], 0
    cer_hist = []
    flags_by_audio = {}
    if args.flags_manifest:
        for line in open(args.flags_manifest, "r", encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            flags_by_audio[row["audio"]] = (row.get("aug", 1), row.get("noise_aug", 1))

    with open(args.mining_report, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("category") not in wanted:
                continue
            reference = str(row.get("reference") or "")
            best_raw, best_cer = None, 1e9
            for raw in row.get("raw_completions") or []:
                try:
                    _, hyp = parse_asr_output(raw, user_language=None)
                except Exception:
                    continue
                cer = char_cer(reference, hyp)
                if cer < best_cer:
                    best_raw, best_cer = raw, cer
            if best_raw is None or best_cer > args.max_cer:
                skipped_hard += 1
                continue
            cer_hist.append(best_cer)
            aug, noise_aug = flags_by_audio.get(row["audio"], (1, 1))
            base_row = {
                "audio": row["audio"],
                "text": best_raw,
                "aug": int(aug),
                "noise_aug": int(noise_aug),
            }
            for _ in range(args.raft_weight):
                kept.append(base_row)

    stabilizer_report = ""

    stabilizer_lines = []
    if args.stabilizer_manifest:
        rng = random.Random(args.seed)
        pool = [
            l if l.endswith("\n") else l + "\n"
            for l in open(args.stabilizer_manifest, "r", encoding="utf-8")
            if l.strip()
        ]
        rng.shuffle(pool)
        stabilizer_lines = pool[: args.stabilizer_rows]
        stabilizer_report = f" stabilizer_rows={len(stabilizer_lines)}"

    rng = random.Random(args.seed + 1)
    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.writelines(stabilizer_lines)

    kept_clips = len(kept) // max(args.raft_weight, 1)
    print(
        f"[done] kept_clips={kept_clips} rows={len(kept)} skipped_hard={skipped_hard} "
        f"mean_best_cer={sum(cer_hist)/max(len(cer_hist),1):.4f}{stabilizer_report}"
    )


if __name__ == "__main__":
    main()
