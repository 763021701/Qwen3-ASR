#!/usr/bin/env python3
# coding=utf-8
"""Evaluate one model on the clean/robust multilingual pilot sets."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

SETS = {
    "gigaspeech_en": {
        "language": "English",
        "script": "evaluation/english_medical/eval_english_medical_asr_jsonl.py",
        "metric": "wer",
        "extra": [],
    },
    "wenetspeech_zh": {
        "language": "Chinese",
        "script": "evaluation/cantonese/eval_cantonese_asr_jsonl.py",
        "metric": "cer",
        "extra": ["--hanzi_script_norm", "off"],
    },
    "magicdata_zh": {
        "language": "Chinese",
        "script": "evaluation/cantonese/eval_cantonese_asr_jsonl.py",
        "metric": "cer",
        "extra": ["--hanzi_script_norm", "off"],
    },
    "kespeech_zh": {
        "language": "Chinese",
        "script": "evaluation/cantonese/eval_cantonese_asr_jsonl.py",
        "metric": "cer",
        "extra": ["--hanzi_script_norm", "off"],
    },
    "chuan_zh": {
        "language": "Chinese",
        "script": "evaluation/cantonese/eval_cantonese_asr_jsonl.py",
        "metric": "cer",
        "extra": ["--hanzi_script_norm", "off"],
    },
    "yue_yue": {
        "language": "Cantonese",
        "script": "evaluation/cantonese/eval_cantonese_asr_jsonl.py",
        "metric": "cer",
        "extra": ["--hanzi_script_norm", "to_traditional"],
    },
}

_WER_RE = re.compile(r"Corpus WER:\s+([0-9.]+)%")
_CER_RE = re.compile(r"CER:\s+([0-9.]+)%")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--clean_root", default="outputs/eval_clean")
    p.add_argument("--robust_root", default="outputs/eval_robust")
    p.add_argument("--sets", default=",".join(SETS))
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--device_map", default="cuda:0")
    return p.parse_args()


def manifest(root: Path, name: str, robust: bool) -> Path:
    if robust:
        return root / name / "eval.jsonl"
    return root / name / f"{name}.jsonl"


def parse_metric(text: str, metric: str) -> float:
    pattern = _WER_RE if metric == "wer" else _CER_RE
    match = pattern.search(text)
    if match is None:
        raise ValueError("metric not found in eval output")
    return float(match.group(1))


def main() -> None:
    args = parse_args()
    names = [name.strip() for name in args.sets.split(",") if name.strip()]
    unknown = set(names) - set(SETS)
    if unknown:
        raise ValueError(f"Unknown eval sets: {sorted(unknown)}")
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    results = []
    for name in names:
        spec = SETS[name]
        for robust in (False, True):
            path = manifest(
                Path(args.robust_root if robust else args.clean_root), name, robust
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            tag = "robust" if robust else "clean"
            eval_dir = out_root / f"{name}_{tag}"
            eval_dir.mkdir(parents=True, exist_ok=True)
            predictions = eval_dir / "predictions.jsonl"
            summary = eval_dir / "summary.txt"
            cmd = [
                sys.executable,
                str(_REPO_ROOT / spec["script"]),
                "--jsonl",
                str(path),
                "--model",
                args.model,
                "--language",
                spec["language"],
                "--batch_size",
                str(1 if name == "yue_yue" else args.batch_size),
                "--device_map",
                args.device_map,
                "--output_predictions",
                str(predictions),
                *spec["extra"],
            ]
            english_script = "english_medical" in spec["script"]
            if english_script:
                cmd.extend(["--output_summary", str(summary)])
            if args.max_samples > 0:
                cmd.extend(["--max_samples", str(args.max_samples)])
            print(f"[eval] {name} {tag}: {' '.join(cmd)}", flush=True)
            completed = subprocess.run(
                cmd, cwd=str(_REPO_ROOT), capture_output=True, text=True
            )
            text = completed.stdout + "\n" + completed.stderr
            if completed.returncode != 0:
                print(text[-4000:], file=sys.stderr)
                raise RuntimeError(f"{name} {tag} failed with {completed.returncode}")
            value = parse_metric(text, spec["metric"])
            results.append(
                {
                    "set": name,
                    "condition": tag,
                    "metric": spec["metric"],
                    "value": value,
                    "manifest": str(path),
                    "predictions": str(predictions),
                }
            )
            print(f"[eval] {name} {tag} {spec['metric']}={value:.2f}%", flush=True)
    result_path = out_root / "summary.json"
    result_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[eval] wrote {result_path}")


if __name__ == "__main__":
    main()
