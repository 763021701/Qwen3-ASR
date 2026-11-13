#!/usr/bin/env python3
"""Evaluate one unified checkpoint on POC53, radiology95 and holdout19.

Writes per-set predictions/summaries and a collated summary.json with both
corpus WER and corpus MER.

MER contract (see AGENTS.md / evaluation/README.md):
    normalize_english (medical: c m -> cm, 乘 -> x, alphanumeric boundaries)
    -> masr_eval_pkg ChineseNormalizer.normalize_for_mer
    -> compute_corpus_mer(..., text_preprocessed=True)
The default ChineseNormalizer must not be applied to raw medical text.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation.english_medical.text_normalization import normalize_english  # noqa: E402
from masr_eval_pkg import compute_corpus_mer  # noqa: E402
from masr_eval_pkg.normalizers import get_normalizer  # noqa: E402

EVAL_SCRIPT = REPO / "evaluation/english_medical/eval_english_medical_asr_jsonl.py"
SETS = {
    "poc53": REPO / "data/poc_train_unified_v2/test.jsonl",
    "radiology95": REPO / "data/eval_unified7/radiology95.jsonl",
    "holdout19": REPO / "data/eval_unified7/holdout19.jsonl",
}
WER_RE = re.compile(r"Corpus WER:\s+([0-9.]+)%")
MER_CONTRACT = ("normalize_english -> masr_eval_pkg.normalize_for_mer "
                "-> compute_corpus_mer(text_preprocessed=True)")
_MER_NORMALIZER = get_normalizer("zh", zh_convert="none", number_normalize="to_arabic")


def mer_metrics(predictions_path: Path) -> dict:
    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise RuntimeError(f"no predictions in {predictions_path}")
    references, hypotheses = [], []
    for index, row in enumerate(rows, 1):
        if "reference" not in row or "hypothesis" not in row:
            raise RuntimeError(f"{predictions_path}:{index}: missing reference/hypothesis")
        references.append(_MER_NORMALIZER.normalize_for_mer(normalize_english(row["reference"])))
        hypotheses.append(_MER_NORMALIZER.normalize_for_mer(normalize_english(row["hypothesis"])))
    result = compute_corpus_mer(references, hypotheses, text_preprocessed=True)
    return {
        "percent": round(100.0 * result.mer, 2),
        "substitutions": int(result.total_substitutions),
        "deletions": int(result.total_deletions),
        "insertions": int(result.total_insertions),
        "ref_tokens": int(result.total_n_ref_tokens),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--sets", default="poc53,radiology95,holdout19")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"model": args.model, "mer_contract": MER_CONTRACT, "sets": {}}
    for name in [s for s in args.sets.split(",") if s]:
        manifest = SETS[name]
        predictions = out_dir / f"{name}_predictions.jsonl"
        summary_path = out_dir / f"{name}_summary.txt"
        cmd = [
            sys.executable, str(EVAL_SCRIPT),
            "--jsonl", str(manifest),
            "--model", args.model,
            "--output_predictions", str(predictions),
            "--output_summary", str(summary_path),
            "--batch_size", str(args.batch_size),
            "--max_new_tokens", str(args.max_new_tokens),
        ]
        print(f"[eval] {name}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True, cwd=REPO)

        match = WER_RE.search(summary_path.read_text(encoding="utf-8"))
        if not match:
            raise RuntimeError(f"could not parse WER from {summary_path}")
        mer = mer_metrics(predictions)
        with summary_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"\nMER normalization:   {MER_CONTRACT}\n"
                f"Corpus MER:          {mer['percent']:.2f}%\n"
                f"MER S/D/I:           {mer['substitutions']}/{mer['deletions']}/{mer['insertions']} "
                f"(ref tokens {mer['ref_tokens']})\n"
            )
        results["sets"][name] = {
            "wer_percent": float(match.group(1)),
            "mer_percent": mer["percent"],
            "mer": mer,
            "summary": str(summary_path),
            "predictions": str(predictions),
        }
        print(
            f"[eval] {name}: WER={results['sets'][name]['wer_percent']:.2f}% "
            f"MER={mer['percent']:.2f}% "
            f"S/D/I={mer['substitutions']}/{mer['deletions']}/{mer['insertions']}",
            flush=True,
        )

    (out_dir / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
