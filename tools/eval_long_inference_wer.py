#!/usr/bin/env python3
# coding=utf-8
"""Score long_inference.py logs against raw/POC_test_long references.

Parses the "Full text:" block from each inference log, normalizes reference
and hypothesis with evaluation/english_medical/text_normalization.py (same
normalization as the generation_eval step-3900_predictions.jsonl), and
computes corpus WER with masr_eval_pkg.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.english_medical.text_normalization import normalize_english
from masr_eval_pkg import compute_wer


def parse_full_text(log_path: Path) -> str:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        start = lines.index("Full text:")
    except ValueError:
        raise SystemExit(f"No 'Full text:' block in {log_path}")
    end = len(lines)
    while end > start and set(lines[end - 1]) == {"="}:
        end -= 1
    return "\n".join(lines[start + 1 : end]).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", required=True, type=Path)
    parser.add_argument("--audio-root", required=True, type=Path,
                        help="e.g. raw/POC_test_long (wav+txt pairs inside)")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    records = []
    wavs = []
    for dirpath, _, filenames in os.walk(args.audio_root, followlinks=True):
        wavs.extend(Path(dirpath) / fn for fn in filenames if fn.endswith(".wav"))
    for wav in sorted(wavs):
        stem = wav.stem
        log = args.logs_dir / f"{stem}.log"
        if not log.is_file():
            raise SystemExit(f"Missing log for {stem}: {log}")
        ref = (wav.with_suffix(".txt")).read_text(encoding="utf-8").strip()
        hyp = parse_full_text(log)
        records.append({
            "audio": str(wav.resolve()),
            "reference": ref,
            "reference_normalized": normalize_english(ref),
            "hypothesis": hyp,
            "hypothesis_normalized": normalize_english(hyp),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    refs = [r["reference_normalized"] for r in records]
    hyps = [r["hypothesis_normalized"] for r in records]
    wer_result = compute_wer(refs, hyps, per_sample=True)
    per_sample = wer_result.get("per_sample", [])
    exact = sum(1 for s in per_sample if s.get("wer", 1.0) == 0.0)

    summary_lines = [
        "=== Long-audio (POC_test_long) ASR Evaluation ===",
        f"Logs:                {args.logs_dir}",
        f"Predictions:         {predictions_path}",
        f"Samples:             {len(records)}",
        "Normalization:       NFKC + Chinese/English numerals to Arabic + preserve decimal points + "
        "x/乘 -> x + merge spaced letters + split alphanumeric boundaries + preserve parenthesized content + gram variants + paraffine -> paraffin + "
        "remove numeric 個 classifier + close dictionary-listed hyphenated compounds + map other dashes to spaces + lowercase + strip other ASCII punctuation + collapse whitespace",
        f"Corpus WER:          {wer_result['wer'] * 100:.2f}%",
        f"Substitutions:       {wer_result['substitutions']}",
        f"Deletions:           {wer_result['deletions']}",
        f"Insertions:          {wer_result['insertions']}",
        f"Ref tokens (total):  {wer_result['n_ref_tokens']}",
        f"Sentence exact acc:  {exact / max(len(per_sample), 1) * 100:.2f}% ({exact}/{len(per_sample)})",
        "",
        "Per-file WER:",
    ]
    for rec, s in zip(records, per_sample):
        name = Path(rec["audio"]).stem
        summary_lines.append(f"  {name:12s} WER {s['wer'] * 100:6.2f}%  "
                             f"(S {s.get('substitutions', '-')}, D {s.get('deletions', '-')}, "
                             f"I {s.get('insertions', '-')}, N {s.get('n_ref_tokens', '-')})")
    summary = "\n".join(summary_lines)
    (args.output_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
