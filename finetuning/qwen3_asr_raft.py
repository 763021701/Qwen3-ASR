#!/usr/bin/env python3
# coding=utf-8
"""Reachable-hard RAFT post-training for Qwen3-ASR.

Iterated best-of-n distillation targeting the sampling-vs-greedy gap: clips
the policy solves only sometimes (best-of-G close to the reference but not
all samples correct) are distilled with their best sampled completion as the
target, so continued SFT transfers rarely-greedy success into greedy
decoding. Clips the policy never solves (best CER above --max_cer) are
excluded — reference-target oversampling of unreachable clips measurably
degrades the model — and a reference-target stabilizer sample from the same
manifest prevents drift.

Per round:
  1. roll out G samples per training clip with the current policy (on-policy);
  2. keep clips with 0 < best_CER <= --max_cer; target = that best sample's
     raw completion (its own language prefix preserved), repeated
     --raft_weight times with the source row's aug/noise_aug flags inherited;
  3. append --stabilizer_rows reference-target rows sampled from the manifest;
  4. write a per-round pipeline config (model_path = previous best, train_file
     = the RAFT set) and invoke tools/qwen3_asr_pipeline.py --stage train;
  5. read the round's best checkpoint (best_cer_checkpoints.json rank 1) and
     use it as the next round's policy.

Pure helper functions (classification/selection) are importable for tests;
see tests/test_qwen3_asr_raft.py.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import GenerationConfig

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.english_medical.text_normalization import normalize_english
from finetuning.qwen3_asr_grpo import (
    JsonlDataset,
    _prepare_prefix_inputs,
    rollout_groups,
)
from finetuning.qwen3_asr_sft import find_latest_checkpoint
from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import parse_asr_output


def char_cer(reference: str, hypothesis: str) -> float:
    """Character error rate on the shared evaluation normalization."""
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


def classify_clip(best_cer: float, worst_cer: float, max_cer: float) -> str:
    """reachable_hard: solved sometimes but not always; easy: all-perfect;
    unreachable: even the best sample misses --max_cer."""
    if best_cer > max_cer:
        return "unreachable"
    if worst_cer == 0.0:
        return "easy"
    return "reachable_hard"


def select_raft_rows(
    clip_reports: Sequence[Dict[str, Any]],
    max_cer: float,
    raft_weight: int,
    stabilizer_rows: int,
    rng: random.Random,
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Turn per-clip rollouts into training rows (pure, no I/O).

    clip_reports: dicts with keys audio, text (source label with prefix),
    completions (list of raw completions incl. language prefix), and optional
    aug/noise_aug flags. RAFT targets inherit the source aug/noise_aug flags
    (mixed-schema manifests would otherwise surface as None after HF datasets
    column unification and crash the collator).
    """
    rows: List[Dict[str, Any]] = []
    counts = {"reachable_hard": 0, "easy": 0, "unreachable": 0}
    cer_sum = 0.0
    for clip in clip_reports:
        reference = str(clip.get("reference") or "")
        best_raw, best_cer, worst_cer = None, 1e9, 0.0
        for raw in clip.get("completions") or []:
            try:
                _, hyp = parse_asr_output(raw, user_language=None)
            except Exception:
                continue
            cer = char_cer(reference, hyp)
            worst_cer = max(worst_cer, cer)
            if cer < best_cer:
                best_raw, best_cer = raw, cer
        category = classify_clip(
            best_cer if best_raw is not None else 1e9, worst_cer, max_cer
        )
        counts[category] += 1
        if category != "reachable_hard":
            continue
        cer_sum += best_cer
        aug = clip.get("aug")
        noise_aug = clip.get("noise_aug")
        aug = 1 if aug is None else int(aug)
        noise_aug = 1 if noise_aug is None else int(noise_aug)
        base_row = {
            "audio": clip["audio"],
            "text": best_raw,
            "aug": aug,
            "noise_aug": noise_aug,
        }
        for _ in range(raft_weight):
            rows.append(dict(base_row))
    if stabilizer_rows:
        sources = list(clip_reports)
        rng.shuffle(sources)
        for clip in sources[:stabilizer_rows]:
            aug = clip.get("aug")
            noise_aug = clip.get("noise_aug")
            rows.append(
                {
                    "audio": clip["audio"],
                    "text": clip["text"],
                    "aug": 1 if aug is None else int(aug),
                    "noise_aug": 1 if noise_aug is None else int(noise_aug),
                }
            )
    report = dict(counts)
    report["kept_clips"] = counts["reachable_hard"]
    report["mean_best_cer"] = (
        round(cer_sum / counts["reachable_hard"], 6) if counts["reachable_hard"] else None
    )
    report["rows"] = len(rows)
    return rows, report


def roll_out_clips(
    model: Any,
    processor: Any,
    dataset: JsonlDataset,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Dict[str, Any]]:
    """Sample G completions per clip; return per-clip report dicts."""
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=lambda rows: rows
    )
    eos = model.generation_config.eos_token_id or [151645, 151643]
    if isinstance(eos, int):
        eos = [eos]
    gen_cfg = GenerationConfig(
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=eos,
        pad_token_id=model.generation_config.pad_token_id or eos[-1],
    )
    reports: List[Dict[str, Any]] = []
    was_training = model.training
    model.eval()
    for rows in loader:
        row = rows[0]
        prefix_inputs = _prepare_prefix_inputs(
            [row], processor, device, dtype, args.sr
        )
        rollout = rollout_groups(
            model.thinker, processor, prefix_inputs, args.num_generations, gen_cfg
        )
        reports.append(
            {
                "audio": row["audio"],
                "text": row["text"],
                "aug": row.get("aug", 1),
                "noise_aug": row.get("noise_aug", 1),
                "completions": list(rollout.raw_completions),
            }
        )
    if was_training:
        model.train()
    return reports


def best_checkpoint_of(round_dir: Path) -> Optional[str]:
    marker = round_dir / "best_cer_checkpoints.json"
    if marker.is_file():
        data = json.loads(marker.read_text(encoding="utf-8"))
        checkpoints = data.get("checkpoints") or []
        if checkpoints:
            return checkpoints[0]["path"]
    # Tiny rounds may finish without a best-cer record; fall back to the
    # latest fully-written checkpoint in the round dir.
    latest = find_latest_checkpoint(str(round_dir))
    if latest and (Path(latest) / "model.safetensors").is_file():
        return latest
    return None


def write_round_config(
    base_config: Dict[str, Any],
    round_dir: Path,
    model_path: str,
    train_jsonl: Path,
    lr: float,
    epochs: int,
) -> Path:
    config = json.loads(json.dumps(base_config))  # deep copy
    config.setdefault("training", {})
    config["training"]["model_path"] = model_path
    config["training"]["output_dir"] = str(round_dir)
    config["training"]["lr"] = lr
    config["training"]["epochs"] = epochs
    config["training"].pop("resume_from", None)
    config.setdefault("dataset", {})
    config["dataset"]["train_jsonl"] = str(train_jsonl)
    config.setdefault("evaluation", {})
    config["evaluation"]["model"] = str(round_dir / "checkpoint-0")
    config["evaluation"]["output_dir"] = str(round_dir / "eval")
    round_config = round_dir / "config.yaml"
    with round_config.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    return round_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="Pipeline yaml (dataset/training blocks).")
    parser.add_argument("--output_dir", required=True, help="Root dir for rounds and reports.")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_cer", type=float, default=0.15)
    parser.add_argument("--raft_weight", type=int, default=3)
    parser.add_argument("--stabilizer_rows", type=int, default=1500)
    parser.add_argument("--max_samples", type=int, default=0, help="Cap clips per round (smoke tests).")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-6, help="Override training.lr for RAFT rounds.")
    parser.add_argument("--epochs", type=int, default=3, help="Override training.epochs for RAFT rounds.")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", type=int, default=0, choices=(0, 1))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    with open(args.config, "r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)
    train_file = str(base_config["dataset"]["train_jsonl"])
    eval_file = str(base_config["dataset"]["dev_jsonl"])
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[raft] config={args.config} train={train_file} rounds={args.rounds}")
    summary = {"rounds": [], "args": vars(args)}

    model_path = str(base_config["training"]["model_path"])
    for round_idx in range(1, args.rounds + 1):
        round_dir = output_root / f"round_{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)
        print(f"[raft] round {round_idx}: policy={model_path}")

        dataset = JsonlDataset(train_file, max_samples=args.max_samples)
        if args.dry_run:
            print(f"[dry-run] round {round_idx}: would roll out {len(dataset)} clips, "
                  f"train from {model_path} (lr={args.lr}, epochs={args.epochs})")
            summary["rounds"].append({"round": round_idx, "dry_run": True})
            model_path = f"{round_dir}/dry_run_checkpoint"
            continue

        device = torch.device("cuda:0")
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
        wrapper = Qwen3ASRModel.from_pretrained(model_path, dtype=dtype, device_map=None)
        model = wrapper.model.to(device)
        processor = wrapper.processor
        clip_reports = roll_out_clips(model, processor, dataset, args, device, dtype)
        del wrapper, model
        torch.cuda.empty_cache()

        rows, report = select_raft_rows(
            clip_reports, args.max_cer, args.raft_weight, args.stabilizer_rows, rng
        )
        train_jsonl = round_dir / "raft_train.jsonl"
        with train_jsonl.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        report_path = round_dir / "raft_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[raft] round {round_idx}: {report}")

        round_config = write_round_config(
            base_config, round_dir, model_path, train_jsonl, args.lr, args.epochs
        )
        cmd = [
            sys.executable,
            str(_REPO_ROOT / "tools" / "qwen3_asr_pipeline.py"),
            "--config", str(round_config),
            "--stage", "train",
        ]
        subprocess.run(cmd, check=True)

        best = best_checkpoint_of(round_dir)
        if best is None:
            print(f"[raft] round {round_idx}: no checkpoint found; stopping.")
            break
        summary["rounds"].append(
            {"round": round_idx, "policy_in": model_path, "best_checkpoint": best, **report}
        )
        model_path = best
        (output_root / "raft_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    (output_root / "raft_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[raft] done; final policy={model_path}; summary={output_root / 'raft_summary.json'}")


if __name__ == "__main__":
    main()
