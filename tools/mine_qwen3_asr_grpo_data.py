#!/usr/bin/env python3
# coding=utf-8
"""Mine ASR examples for GRPO from group-sampled WER behavior.

The miner generates G candidates per audio under one frozen audio encoding,
records per-candidate WER, assigns a GRPO usefulness category, and writes a
weighted JSONL manifest for the later RL stage. Reports are checkpointed per
batch and may be resumed after an interruption.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import GenerationConfig

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from finetuning.qwen3_asr_grpo import (
    JsonlDataset,
    _prepare_prefix_inputs,
    asr_rewards,
    extract_reference_text,
    rollout_groups,
)
from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import parse_asr_output


@dataclass(frozen=True)
class MiningThresholds:
    easy_mean_max: float
    easy_std_max: float
    capable_best_max: float
    hard_std_min: float
    suspect_worst_min: float
    suspect_best_min: float


def error_statistics(values: Sequence[float], label: str = "cer") -> Dict[str, float]:
    if not values:
        raise ValueError("Need at least one error-rate value.")
    return {
        f"mean_{label}": statistics.fmean(values),
        f"best_{label}": min(values),
        f"worst_{label}": max(values),
        f"std_{label}": statistics.pstdev(values),
    }


def wer_statistics(wers: Sequence[float]) -> Dict[str, float]:
    """Backwards-compatible wrapper; classification now uses CER stats."""
    return error_statistics(wers, label="wer")


def classify_error_rates(
    cers: Sequence[float], thresholds: MiningThresholds, label: str = "cer"
) -> Tuple[str, Dict[str, float]]:
    """Classify an audio by its within-group error-rate distribution (4 classes).

    Uses character-level CER: word-level WER systematically overstates
    severity on segmentation-free text (Cantonese, specimen-label digit
    strings), where a single character error marks a whole "word" wrong and
    trivially crosses the catastrophic WER threshold — 79% of the round-1
    catastrophic labels were such false alarms (user audio verification,
    2026-09-08).

    4-class design (2026-09-09, user-confirmed), one action per class:
      easy    mean <= easy_mean_max and std <= easy_std_max
              No within-group signal; excluded from GRPO (zero-variance group).
      suspect best > capable_best_max and (worst >= suspect_worst_min or
              best >= suspect_best_min)
              Even the best rollout is badly wrong -> bad label/audio suspect
              for human verification; never trained on.
      hard    best <= capable_best_max and std >= hard_std_min
              The model demonstrably solves the clip (best) AND the group
              carries reward variance -> the GRPO training mass. Covers both
              occasional full hallucinations ([0 x7, 3.0]) and the classic
              reachable-hard profile.
      weak    everything else (uniform-mediocre or unreachable); excluded.
    """
    stats = error_statistics(cers, label=label)
    mean = stats[f"mean_{label}"]
    best = stats[f"best_{label}"]
    worst = stats[f"worst_{label}"]
    std = stats[f"std_{label}"]

    if mean <= thresholds.easy_mean_max and std <= thresholds.easy_std_max:
        return "easy", stats
    if best > thresholds.capable_best_max and (
        worst >= thresholds.suspect_worst_min or best >= thresholds.suspect_best_min
    ):
        return "suspect", stats
    if best <= thresholds.capable_best_max and std >= thresholds.hard_std_min:
        return "hard", stats
    return "weak", stats


def category_weight(category: str, args: argparse.Namespace) -> int:
    return int(getattr(args, f"weight_{category}"))


def oversample_records(
    mined: Iterable[Tuple[Dict[str, Any], str, Dict[str, float]]], args: argparse.Namespace
) -> List[Dict[str, Any]]:
    """Repeat original manifest rows according to their deterministic category weight."""
    sampled: List[Dict[str, Any]] = []
    seen_audio = set()
    dedupe_by_audio = bool(getattr(args, "dedupe_by_audio", False))
    for source, category, stats in mined:
        weight = category_weight(category, args)
        if weight <= 0:
            continue
        if dedupe_by_audio:
            audio = str(source["audio"])
            if audio in seen_audio:
                continue
            seen_audio.add(audio)
        for _ in range(weight):
            row = dict(source)
            row["grpo_category"] = category
            for key, value in stats.items():
                row[f"grpo_mining_{key}"] = round(value, 6)
            sampled.append(row)
    return sampled


def _clean_hypothesis(raw_completion: str) -> str:
    try:
        _, text = parse_asr_output(raw_completion, user_language=None)
        return text
    except Exception:
        return ""


def _validate_output_paths(args: argparse.Namespace) -> None:
    source = os.path.abspath(args.input_jsonl)
    for output in (args.output_report, args.output_train_jsonl):
        if os.path.abspath(output) == source:
            raise ValueError("Output paths must not overwrite --input_jsonl.")
    if os.path.abspath(args.output_report) == os.path.abspath(args.output_train_jsonl):
        raise ValueError("--output_report and --output_train_jsonl must be different files.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine Qwen3-ASR data for GRPO from G-sample error-rate statistics.")
    parser.add_argument(
        "--classification_metric",
        choices=("cer", "wer"),
        default="cer",
        help="Error metric used for category classification. Default cer: "
        "word-level WER overstates severity on segmentation-free text "
        "(Cantonese / specimen-label digit strings). Both WER and CER "
        "statistics are always reported either way.",
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_jsonl", required=True, help="Source RL candidates with audio/text and optional prompt.")
    parser.add_argument("--output_report", required=True, help="Per-audio group rollout report JSONL.")
    parser.add_argument("--output_train_jsonl", required=True, help="Oversampled GRPO manifest JSONL.")
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1, help="Audio prompts per rollout batch.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--easy_mean_max", type=float, default=0.15)
    parser.add_argument("--easy_std_max", type=float, default=0.05)
    parser.add_argument(
        "--capable_best_max",
        type=float,
        default=0.30,
        help="Best-of-group error at or below this means the model can produce a "
        "near-correct transcript (hard); above it the clip is a bad-data suspect "
        "when it also hallucinates or fails deterministically.",
    )
    parser.add_argument(
        "--hard_std_min",
        type=float,
        default=0.05,
        help="Within-group std at or above which the group carries GRPO reward variance.",
    )
    parser.add_argument("--suspect_worst_min", type=float, default=1.00)
    parser.add_argument("--suspect_best_min", type=float, default=0.60)
    parser.add_argument("--weight_easy", type=int, default=0)
    parser.add_argument("--weight_hard", type=int, default=5)
    parser.add_argument("--weight_suspect", type=int, default=0)
    parser.add_argument("--weight_weak", type=int, default=0)
    parser.add_argument(
        "--dedupe_by_audio",
        type=int,
        default=0,
        choices=(0, 1),
        help="Keep only the first selected row for each exact audio path after category filtering.",
    )
    parser.add_argument(
        "--resume",
        type=int,
        default=1,
        choices=(0, 1),
        help="Resume from durable per-batch records already present in --output_report.",
    )
    parser.add_argument("--dry_run", type=int, default=0, choices=(0, 1))
    return parser.parse_args()


def _thresholds_from_args(args: argparse.Namespace) -> MiningThresholds:
    return MiningThresholds(
        easy_mean_max=args.easy_mean_max,
        easy_std_max=args.easy_std_max,
        capable_best_max=args.capable_best_max,
        hard_std_min=args.hard_std_min,
        suspect_worst_min=args.suspect_worst_min,
        suspect_best_min=args.suspect_best_min,
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_generations < 2:
        raise ValueError("--num_generations must be at least 2.")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be positive.")
    if args.max_new_tokens < 1:
        raise ValueError("--max_new_tokens must be positive.")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive.")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top_p must be in (0, 1].")
    if args.top_k < 0:
        raise ValueError("--top_k must be non-negative.")
    for category in ("easy", "hard", "suspect", "weak"):
        if category_weight(category, args) < 0:
            raise ValueError(f"weight for {category} must be non-negative.")


def _write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_completed_reports(path: str, dataset_size: int) -> Dict[int, Dict[str, Any]]:
    """Load durable reports keyed by their source-manifest row index."""
    if not os.path.isfile(path):
        return {}
    completed: Dict[int, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            source_index = record.get("miner_source_index", record.get("index"))
            if not isinstance(source_index, int) or not 0 <= source_index < dataset_size:
                raise ValueError(
                    f"{path}:{line_number} has invalid miner source index: {source_index!r}"
                )
            if source_index in completed:
                raise ValueError(f"{path}:{line_number} duplicates miner source index {source_index}.")
            completed[source_index] = record
    return completed


def _report_stats(record: Dict[str, Any]) -> Dict[str, float]:
    return {
        "mean_wer": float(record["mean_wer"]),
        "best_wer": float(record["best_wer"]),
        "std_wer": float(record["std_wer"]),
    }


def _build_train_manifest(
    dataset: JsonlDataset,
    reports: Dict[int, Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    if len(reports) != len(dataset):
        raise ValueError(
            "Cannot build the final GRPO manifest from an incomplete mining report: "
            f"{len(reports)}/{len(dataset)} rows."
        )
    mined = []
    for source_index, source in enumerate(dataset.rows):
        report = reports[source_index]
        if report.get("audio") != source["audio"]:
            raise ValueError(
                f"Report/source audio mismatch at source index {source_index}: "
                f"{report.get('audio')!r} != {source['audio']!r}."
            )
        mined.append((source, str(report["category"]), _report_stats(report)))
    return oversample_records(mined, args)


def main() -> None:
    args = parse_args()
    _validate_args(args)
    _validate_output_paths(args)
    thresholds = _thresholds_from_args(args)
    dataset = JsonlDataset(args.input_jsonl, max_samples=args.max_samples)
    print(f"[data] rows={len(dataset)} input={args.input_jsonl}")
    if args.dry_run:
        print(f"[dry-run] G={args.num_generations} batch_size={args.batch_size} model={args.model_path}")
        print(f"[dry-run] thresholds={thresholds}")
        print(
            "[dry-run] weights="
            + str({
                category: category_weight(category, args)
                for category in ("easy", "hard", "suspect", "weak")
            })
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-ASR data mining requires CUDA.")
    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    wrapper = Qwen3ASRModel.from_pretrained(args.model_path, dtype=dtype, device_map=None)
    model = wrapper.model.to(device).eval()
    processor = wrapper.processor
    eos_token_ids = model.generation_config.eos_token_id or [151645, 151643]
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=eos_token_ids,
        pad_token_id=model.generation_config.pad_token_id or eos_token_ids[-1],
        return_dict_in_generate=False,
    )

    output_parent = os.path.dirname(os.path.abspath(args.output_report))
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    if not args.resume and os.path.isfile(args.output_report):
        os.remove(args.output_report)
    reports = _load_completed_reports(args.output_report, len(dataset))
    category_counts: Counter[str] = Counter(
        str(record["category"]) for record in reports.values()
    )
    pending = [
        (source_index, source)
        for source_index, source in enumerate(dataset.rows)
        if source_index not in reports
    ]
    print(f"[resume] completed={len(reports)} remaining={len(pending)}", flush=True)
    loader = DataLoader(pending, batch_size=args.batch_size, shuffle=False, collate_fn=list)
    with open(args.output_report, "a", encoding="utf-8") as report_file:
        for batch_index, indexed_rows in enumerate(loader, start=1):
            rows = [source for _, source in indexed_rows]
            prefix_inputs = _prepare_prefix_inputs(rows, processor, device, dtype, args.sr)
            rollout = rollout_groups(
                model.thinker, processor, prefix_inputs, args.num_generations, generation_config
            )
            references = [
                extract_reference_text(str(row["text"]))
                for row in rows
                for _ in range(args.num_generations)
            ]
            wers = (-asr_rewards(references, rollout.raw_completions, reward_mode="wer")).tolist()
            cers = (-asr_rewards(references, rollout.raw_completions, reward_mode="cer")).tolist()
            batch_reports = []
            for row_index, (source_index, source) in enumerate(indexed_rows):
                start = row_index * args.num_generations
                end = start + args.num_generations
                group_raw = rollout.raw_completions[start:end]
                group_wers = wers[start:end]
                group_cers = cers[start:end]
                # Classification uses CER: WER overstates severity on
                # segmentation-free text (Cantonese / digit strings).
                classification_values = group_cers if args.classification_metric == "cer" else group_wers
                category, stats = classify_error_rates(
                    classification_values, thresholds, label=args.classification_metric
                )
                stats = {**wer_statistics(group_wers), **stats}
                reference = extract_reference_text(str(source["text"]))
                report = {
                    "index": source.get("index", source_index),
                    "miner_source_index": source_index,
                    "audio": source["audio"],
                    "reference": reference,
                    "category": category,
                    **{key: round(value, 6) for key, value in stats.items()},
                    "wers": [round(value, 6) for value in group_wers],
                    "cers": [round(value, 6) for value in group_cers],
                    "hypotheses": [_clean_hypothesis(raw) for raw in group_raw],
                    "raw_completions": group_raw,
                }
                batch_reports.append(report)
                category_counts[category] += 1
            for report in batch_reports:
                report_file.write(json.dumps(report, ensure_ascii=False) + "\n")
                reports[int(report["miner_source_index"])] = report
            report_file.flush()
            os.fsync(report_file.fileno())
            print(
                f"[batch {batch_index}] mined={len(reports)}/{len(dataset)} "
                f"categories={dict(sorted(category_counts.items()))}",
                flush=True,
            )
            del prefix_inputs, rollout

    sampled = _build_train_manifest(dataset, reports, args)
    _write_jsonl(args.output_train_jsonl, sampled)
    print(
        "[done] "
        f"report_rows={len(reports)} sampled_rows={len(sampled)} "
        f"sampled_unique_audio={len({row['audio'] for row in sampled})} "
        f"categories={dict(sorted(category_counts.items()))}"
    )


if __name__ == "__main__":
    main()
