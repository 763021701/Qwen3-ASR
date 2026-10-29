#!/usr/bin/env python3
"""Evaluate every checkpoint of a radiology experiment on the standard sets.

Sets (new-label references):
  holdout19   — data/poc_radiology_oracle/holdout.jsonl (unseen recordings)
  radiology95 — train76 + holdout19 concatenated (all real clips; none seen by synth runs)
  poc53       — pathology selection set (forgetting)

    python tools/eval_radiology_checkpoints.py --run-dir outputs/<run> \
        [--sets holdout19,radiology95,poc53] [--max-new-tokens 512]
Writes <run>/radiology_eval_summary.json and prints one row per checkpoint×set.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "finetuning"))

from evaluation.english_medical.text_normalization import normalize_english  # noqa: E402
from masr_eval_pkg import compute_wer  # noqa: E402
from qwen3_asr_grpo import cer_decomposition  # noqa: E402

EVAL = REPO / "evaluation/english_medical/eval_english_medical_asr_jsonl.py"
DATA = REPO / "data/poc_radiology_oracle"
SET_PATHS = {
    "holdout19": DATA / "holdout.jsonl",
    "poc53": REPO / "data/poc_train_fullvoice_clean_v2synth_yt_plus0821/test.jsonl",
}


def radiology95(run_dir: Path) -> Path:
    p = run_dir / "radiology95.jsonl"
    if not p.is_file():
        lines = [ln for part in (DATA / "train.jsonl", DATA / "holdout.jsonl")
                 for ln in part.read_text(encoding="utf-8").splitlines() if ln.strip()]
        p.write_text("".join(ln + "\n" for ln in lines), encoding="utf-8")
    return p


def evaluate(ckpt: Path, name: str, jsonl: Path, run_dir: Path, max_new_tokens: int) -> dict:
    outdir = run_dir / "eval_final" / f"{ckpt.name}_{name}"
    outdir.mkdir(parents=True, exist_ok=True)
    pred = outdir / "predictions.jsonl"
    if not pred.is_file():
        subprocess.run(
            [sys.executable, str(EVAL), "--jsonl", str(jsonl), "--model", str(ckpt),
             "--language", "None", "--batch_size", "4", "--max_new_tokens", str(max_new_tokens),
             "--output_predictions", str(pred), "--output_summary", str(outdir / "summary.txt")],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rows = [json.loads(ln) for ln in pred.read_text(encoding="utf-8").splitlines() if ln.strip()]
    refs = [normalize_english(r["reference"]) for r in rows]
    hyps = [normalize_english(r["hypothesis"]) for r in rows]
    per_clip = [compute_wer([a], [b])["wer"] for a, b in zip(refs, hyps)]
    s = d = i = n = 0
    for r in rows:
        a, b, c, m = cer_decomposition(r["reference"], r["hypothesis"])
        s += a; d += b; i += c; n += m
    med = sorted(per_clip)[len(per_clip) // 2]
    return {"checkpoint": ckpt.name, "set": name, "n": len(rows),
            "corpus_wer": round(compute_wer(refs, hyps)["wer"], 4),
            "corpus_cer": round((s + d + i) / max(n, 1), 4),
            "median_wer": round(med, 4),
            "lt30": sum(w < 0.30 for w in per_clip),
            "gt100": sum(w > 1.00 for w in per_clip),
            "exact": sum(w == 0 for w in per_clip)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--sets", default="holdout19,radiology95,poc53")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--summary", default="radiology_eval_summary.json")
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    ckpts = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.rsplit("-", 1)[1]))
    if not ckpts:
        raise SystemExit(f"no checkpoints under {run_dir}")
    paths = dict(SET_PATHS)
    paths["radiology95"] = radiology95(run_dir)
    results = []
    for ckpt in ckpts:
        for name in [s for s in args.sets.split(",") if s]:
            row = evaluate(ckpt, name, paths[name], run_dir, args.max_new_tokens)
            results.append(row)
            print("%-16s %-12s n=%-3d WER=%7.2f%% median=%6.1f%% <30%%=%2d >100%%=%2d exact=%d CER=%7.2f%%"
                  % (row["checkpoint"], row["set"], row["n"], 100 * row["corpus_wer"],
                     100 * row["median_wer"], row["lt30"], row["gt100"], row["exact"],
                     100 * row["corpus_cer"]), flush=True)
    (run_dir / args.summary).write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote", run_dir / args.summary)


if __name__ == "__main__":
    main()