#!/usr/bin/env python3
"""Evaluate every numeric checkpoint in an ASR experiment on one JSONL set."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


WER_RE = re.compile(r"^Corpus WER:\s+([0-9.]+)%$", re.MULTILINE)
EXACT_RE = re.compile(r"^Sentence exact acc:\s+([0-9.]+)%", re.MULTILINE)
CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument(
        "--eval-script",
        default=str(repo_root / "evaluation/english_medical/eval_english_medical_asr_jsonl.py"),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--language", default="English")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument(
        "--wait-for-tmux-session",
        default="",
        help="Wait until this tmux session exits before starting evaluation.",
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    return parser.parse_args()


def wait_for_session(session: str, poll_seconds: float) -> None:
    if not session:
        return
    print(f"[wait] tmux session {session!r}", flush=True)
    while subprocess.run(
        ["tmux", "has-session", "-t", session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0:
        time.sleep(poll_seconds)


def checkpoints(experiment_dir: Path) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for path in experiment_dir.glob("checkpoint-*"):
        match = CHECKPOINT_RE.fullmatch(path.name)
        if match and path.is_dir():
            found.append((int(match.group(1)), path))
    return sorted(found)


def metric(path: Path, pattern: re.Pattern[str]) -> float:
    text = path.read_text(encoding="utf-8")
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"Missing metric in {path}")
    return float(match.group(1))


def main() -> None:
    args = parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")

    experiment_dir = Path(args.experiment_dir).resolve()
    eval_dir = experiment_dir / "eval"
    wait_for_session(args.wait_for_tmux_session, args.poll_seconds)

    items = checkpoints(experiment_dir)
    if not items:
        raise FileNotFoundError(f"No numeric checkpoints under {experiment_dir}")

    results = []
    for step, checkpoint in items:
        predictions = eval_dir / f"checkpoint-{step}_poc_test_predictions.jsonl"
        summary = eval_dir / f"checkpoint-{step}_poc_test_summary.txt"
        command = [
            args.python,
            args.eval_script,
            "--jsonl",
            args.jsonl,
            "--model",
            str(checkpoint),
            "--language",
            args.language,
            "--batch_size",
            str(args.batch_size),
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--device_map",
            args.device_map,
            "--output_predictions",
            str(predictions),
            "--output_summary",
            str(summary),
        ]
        print(f"[eval] checkpoint-{step}", flush=True)
        subprocess.run(command, check=True)
        results.append(
            {
                "step": step,
                "checkpoint": str(checkpoint),
                "wer_percent": metric(summary, WER_RE),
                "sentence_exact_accuracy_percent": metric(summary, EXACT_RE),
                "predictions": str(predictions),
                "summary": str(summary),
            }
        )

    result_path = eval_dir / "poc_test_checkpoint_wer.json"
    result_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wrote {result_path}")


if __name__ == "__main__":
    main()
