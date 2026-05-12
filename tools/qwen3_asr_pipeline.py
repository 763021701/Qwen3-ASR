#!/usr/bin/env python3
# coding=utf-8

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shlex
import subprocess
import sys
from typing import Any, Dict, List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.validate_qwen3_asr_jsonl import report_to_dict, validate_jsonl


_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")
_STAGES = ("prepare", "validate", "train", "eval", "all")


def repo_root() -> str:
    return _REPO_ROOT


def abspath(path: str, root: Optional[str] = None) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(root or repo_root(), path))


def _parse_scalar(value: str) -> Any:
    raw = value.strip()
    if raw in ("true", "True"):
        return True
    if raw in ("false", "False"):
        return False
    if raw in ("null", "None", "~"):
        return None
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Parse the small YAML subset used by repo example configs."""
    root: Dict[str, Any] = {}
    stack: List[tuple[int, Dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            raise ValueError(f"Unsupported YAML line: {raw_line}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if path.endswith(".json"):
        return json.loads(text)
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        return data or {}
    except ImportError:
        return _parse_simple_yaml(text)


def save_resolved_config(config: Dict[str, Any], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "pipeline_config.resolved.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2, sort_keys=True)
    return path


def run_command(cmd: List[str], dry_run: bool = False) -> None:
    printable = " ".join(shlex.quote(x) for x in cmd)
    print(f"[cmd] {printable}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    best_step = -1
    best_path: Optional[str] = None
    for name in os.listdir(output_dir):
        match = _CKPT_RE.match(name)
        if not match:
            continue
        path = os.path.join(output_dir, name)
        if not os.path.isdir(path):
            continue
        step = int(match.group(1))
        if step > best_step:
            best_step = step
            best_path = path
    return best_path


def split_jsonl(
    source: str,
    train_out: str,
    dev_out: str,
    test_out: str,
    train_ratio: float,
    dev_ratio: float,
    seed: int,
) -> None:
    with open(source, "r", encoding="utf-8") as f:
        rows = [line for line in f if line.strip()]
    random.Random(seed).shuffle(rows)
    n = len(rows)
    train_n = int(n * train_ratio)
    dev_n = int(n * dev_ratio)
    splits = {
        train_out: rows[:train_n],
        dev_out: rows[train_n : train_n + dev_n],
        test_out: rows[train_n + dev_n :],
    }
    for path, items in splits.items():
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(items)


def manifest_paths(config: Dict[str, Any]) -> Dict[str, str]:
    dataset = config.get("dataset", {})
    output_dir = abspath(str(dataset.get("output_dir", "data/qwen3_asr_pipeline")))
    return {
        "train": abspath(str(dataset.get("train_jsonl") or os.path.join(output_dir, "train.jsonl"))),
        "dev": abspath(str(dataset.get("dev_jsonl") or os.path.join(output_dir, "dev.jsonl"))),
        "test": abspath(str(dataset.get("test_jsonl") or os.path.join(output_dir, "test.jsonl"))),
    }


def _convert_command(
    source_type: str,
    source_path: str,
    output_file: str,
    dataset: Dict[str, Any],
    language: str,
) -> List[str]:
    script = abspath("evaluation/tools/convert_to_qwen3_asr_jsonl.py")
    cmd = [sys.executable, script, "--output_file", output_file, "--language", language]
    if source_type == "common_voice":
        clips_dir = str(dataset.get("clips_dir") or "")
        if not clips_dir:
            clips_dir = os.path.join(os.path.dirname(abspath(source_path)), "clips")
        cmd.extend(["--cv_tsv", abspath(source_path), "--cv_clips_dir", abspath(clips_dir)])
    elif source_type == "kaldi":
        wav_scp = source_path
        text_file = str(dataset.get("text_file") or "")
        if not text_file:
            raise ValueError("Kaldi conversion requires dataset.text_file for single-source mode.")
        cmd.extend(["--wav_scp", abspath(wav_scp), "--text_file", abspath(text_file)])
    elif source_type == "funasr_jsonl":
        cmd.extend(["--funasr_jsonl", abspath(source_path)])
    else:
        raise ValueError(f"Unsupported dataset.source_type: {source_type!r}")
    return cmd


def _split_convert_commands(dataset: Dict[str, Any], language: str, paths: Dict[str, str]) -> List[List[str]]:
    source_type = str(dataset.get("source_type", "")).strip()
    commands: List[List[str]] = []
    for split in ("train", "dev", "test"):
        source = str(dataset.get(f"{split}_source") or "")
        if not source:
            continue
        split_dataset = dict(dataset)
        if source_type == "kaldi":
            text_file = str(dataset.get(f"{split}_text_file") or dataset.get("text_file") or "")
            split_dataset["text_file"] = text_file
        commands.append(_convert_command(source_type, source, paths[split], split_dataset, language))
    return commands


def stage_prepare(config: Dict[str, Any], dry_run: bool) -> Dict[str, str]:
    dataset = config.get("dataset", {})
    language = str(dataset.get("language") or "").strip()
    if not language:
        raise ValueError("dataset.language is required.")
    paths = manifest_paths(config)
    output_dir = os.path.dirname(paths["train"])
    if output_dir and not dry_run:
        os.makedirs(output_dir, exist_ok=True)

    commands = _split_convert_commands(dataset, language, paths)
    if commands:
        for cmd in commands:
            run_command(cmd, dry_run=dry_run)
        return paths

    source = str(dataset.get("source") or "")
    if not source:
        raise ValueError("Provide train/dev/test sources or dataset.source for split mode.")
    all_jsonl = abspath(str(dataset.get("all_jsonl") or os.path.join(output_dir, "all.jsonl")))
    run_command(_convert_command(str(dataset.get("source_type")), source, all_jsonl, dataset, language), dry_run=dry_run)
    if not dry_run:
        train_ratio = float(dataset.get("train_ratio", 0.8))
        dev_ratio = float(dataset.get("dev_ratio", 0.1))
        seed = int(dataset.get("split_seed", 42))
        split_jsonl(all_jsonl, paths["train"], paths["dev"], paths["test"], train_ratio, dev_ratio, seed)
    return paths


def stage_validate(config: Dict[str, Any], dry_run: bool) -> None:
    dataset = config.get("dataset", {})
    language = str(dataset.get("language") or "").strip()
    check_audio = bool(int(dataset.get("check_audio", 1)))
    paths = manifest_paths(config)
    out_dir = abspath(str(config.get("training", {}).get("output_dir", "outputs/qwen3_asr_pipeline")))
    report_dir = os.path.join(out_dir, "validation")
    if dry_run:
        for split, path in paths.items():
            print(f"[dry-run] validate {split}: {path}")
        return
    os.makedirs(report_dir, exist_ok=True)
    failed = False
    for split, path in paths.items():
        report = validate_jsonl(path, expected_language=language, check_audio=check_audio)
        report_path = os.path.join(report_dir, f"{split}_manifest_validation.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_to_dict(report), f, ensure_ascii=False, indent=2)
        print(f"[validate] {split}: valid={report.valid} report={report_path}")
        failed = failed or not report.valid
    if failed:
        raise SystemExit("Manifest validation failed.")


def stage_train(config: Dict[str, Any], dry_run: bool) -> None:
    training = config.get("training", {})
    runtime = config.get("runtime", {})
    paths = manifest_paths(config)
    output_dir = abspath(str(training.get("output_dir", "outputs/qwen3_asr_pipeline")))
    if not dry_run:
        save_resolved_config(config, output_dir)

    train_script = abspath("finetuning/qwen3_asr_sft.py")
    launcher = str(runtime.get("launcher", "python")).strip()
    if launcher == "torchrun":
        nproc = str(runtime.get("nproc_per_node", 1))
        cmd = ["torchrun", f"--nproc_per_node={nproc}", train_script]
    else:
        cmd = [sys.executable, train_script]

    option_map = {
        "model_path": "--model_path",
        "batch_size": "--batch_size",
        "grad_acc": "--grad_acc",
        "lr": "--lr",
        "epochs": "--epochs",
        "save_steps": "--save_steps",
        "save_total_limit": "--save_total_limit",
        "log_steps": "--log_steps",
        "sr": "--sr",
        "num_workers": "--num_workers",
        "pin_memory": "--pin_memory",
        "persistent_workers": "--persistent_workers",
        "prefetch_factor": "--prefetch_factor",
        "resume_from": "--resume_from",
        "resume": "--resume",
    }
    cmd.extend(["--train_file", paths["train"], "--eval_file", paths["dev"], "--output_dir", output_dir])
    for key, flag in option_map.items():
        if key in training:
            cmd.extend([flag, str(training[key])])
    if runtime.get("resume") is True and "resume" not in training:
        cmd.extend(["--resume", "1"])
    run_command(cmd, dry_run=dry_run)


def stage_eval(config: Dict[str, Any], dry_run: bool) -> None:
    evaluation = config.get("evaluation", {})
    training = config.get("training", {})
    paths = manifest_paths(config)
    output_dir = abspath(str(training.get("output_dir", "outputs/qwen3_asr_pipeline")))
    model = str(evaluation.get("model") or "")
    if not model:
        model = find_latest_checkpoint(output_dir) or ""
    if dry_run and not model:
        model = os.path.join(output_dir, "checkpoint-<latest>")
    if not model:
        raise ValueError(f"No model provided and no checkpoint found under {output_dir}")

    eval_script = str(evaluation.get("eval_script") or "")
    if not eval_script:
        raise ValueError("evaluation.eval_script is required.")
    eval_out = abspath(str(evaluation.get("output_dir") or os.path.join(output_dir, "eval")))
    predictions = abspath(str(evaluation.get("output_predictions") or os.path.join(eval_out, "predictions.jsonl")))
    if not dry_run:
        os.makedirs(eval_out, exist_ok=True)

    cmd = [
        sys.executable,
        abspath(eval_script),
        "--jsonl",
        paths["test"],
        "--model",
        abspath(model),
        "--output_predictions",
        predictions,
    ]
    option_map = {
        "language": "--language",
        "max_samples": "--max_samples",
        "batch_size": "--batch_size",
        "max_new_tokens": "--max_new_tokens",
        "context": "--context",
        "device_map": "--device_map",
        "hanzi_script_norm": "--hanzi_script_norm",
    }
    for key, flag in option_map.items():
        if key in evaluation:
            cmd.extend([flag, str(evaluation[key])])
    run_command(cmd, dry_run=dry_run)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Qwen3-ASR finetuning pipeline.")
    p.add_argument("--config", required=True, help="Pipeline YAML/JSON config.")
    p.add_argument("--stage", default="all", choices=_STAGES, help="Pipeline stage to run.")
    p.add_argument("--dry_run", type=int, default=-1, choices=(-1, 0, 1), help="Override runtime.dry_run.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runtime = config.setdefault("runtime", {})
    dry_run = bool(runtime.get("dry_run", False)) if args.dry_run < 0 else bool(args.dry_run)
    stage = args.stage

    if stage in ("prepare", "all"):
        stage_prepare(config, dry_run)
    if stage in ("validate", "all"):
        stage_validate(config, dry_run)
    if stage in ("train", "all"):
        stage_train(config, dry_run)
    if stage in ("eval", "all"):
        stage_eval(config, dry_run)


if __name__ == "__main__":
    main()
