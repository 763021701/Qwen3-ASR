#!/usr/bin/env python3
# coding=utf-8
"""
Download ASLP-lab/WSC-Eval and convert WSC-Eval-ASR into Qwen3-ASR jsonl.

WSC-Eval-ASR is Sichuan Mandarin (四川话) speech, not Cantonese (粤语).
Transcripts use the Qwen3 convention ``language Chinese<asr_text>...`` (default --language).

Output format per line:
  {"audio": "/abs/path/to/file.wav", "text": "language Chinese<asr_text>..."}

Dataset note:
  WSC-Eval-ASR provides two different partition schemes:
    1) Duration-based: Short / Long
    2) Difficulty-based: Easy / Hard
  These schemes overlap, so they must not be merged blindly.

Default behavior:
  - subset=all means Short + Long (full duration-based evaluation set, no duplicates)

Example:
  python evaluation/chinese/wsc/prepare_wsc_eval_qwen3.py \
    --output_jsonl data/chinese/wsc/wsc_eval_qwen3.jsonl \
    --dataset_dir /root/autodl-tmp/datasets/WSC-Eval
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple


DEFAULT_REPO_ID = "ASLP-lab/WSC-Eval"
DEFAULT_ASR_ROOT = "WSC-Eval-ASR"
_TEXT_PREFIX = "language {language}<asr_text>{text}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare WSC-Eval-ASR (Sichuan Mandarin / 四川话) as Qwen3-ASR jsonl for evaluation."
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help="Output Qwen3-style jsonl path.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=os.path.join(os.getcwd(), "downloads", "WSC-Eval"),
        help="Local directory for the downloaded WSC-Eval dataset repo.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        choices=("all", "short", "long", "easy", "hard", "easy_hard"),
        help="Subset to convert. all = Short + Long (duration-based full set); "
        "easy_hard = Easy + Hard (difficulty-based full set, matches the paper).",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="Chinese",
        help="Language label to inject into Qwen3-ASR text.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=DEFAULT_REPO_ID,
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Use an already-downloaded dataset_dir and skip Hugging Face download.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="If >0, write at most N samples after conversion.",
    )
    return parser.parse_args()


def subset_dirs(subset: str) -> List[str]:
    key = subset.strip().lower()
    if key == "all":
        return ["Short", "Long"]
    if key == "easy_hard":
        return ["Easy", "Hard"]
    return [key.capitalize()]


def allow_patterns_for_subset(subset: str) -> List[str]:
    pats = ["README.md", f"{DEFAULT_ASR_ROOT}/readme.md"]
    for d in subset_dirs(subset):
        pats.append(f"{DEFAULT_ASR_ROOT}/{d}/*")
    return pats


def download_repo(repo_id: str, dataset_dir: str, subset: str) -> None:
    try:
        from huggingface_hub import snapshot_download  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub first: pip install huggingface_hub") from exc

    print(f"Downloading {repo_id} to {dataset_dir} ...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=dataset_dir,
        allow_patterns=allow_patterns_for_subset(subset),
        ignore_patterns=["**/.DS_Store", "**/.cache/**", "WSC-Eval-TTS/**"],
    )


def ensure_required_files(dataset_dir: str, subset: str) -> None:
    needed: List[str] = []
    for d in subset_dirs(subset):
        needed.extend(
            [
                os.path.join(DEFAULT_ASR_ROOT, d, "text"),
                os.path.join(DEFAULT_ASR_ROOT, d, "wav.scp"),
            ]
        )
    missing = [rel for rel in needed if not os.path.exists(os.path.join(dataset_dir, rel))]
    if missing:
        raise SystemExit(f"Missing required dataset files under {dataset_dir}: {missing}")


def read_kaldi_mapping(path: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Invalid line in {path}:{line_no}: {raw_line.rstrip()}")
            utt_id, value = parts
            mapping[utt_id] = value.strip()
    return mapping


def normalize_target_text(text: str, language: str) -> str:
    txt = (text or "").strip()
    if not txt:
        raise ValueError("Empty transcript is not allowed.")
    return _TEXT_PREFIX.format(language=language, text=txt)


def resolve_audio_path(dataset_dir: str, rel_path: str, subset_dir: str) -> str:
    rel = rel_path.strip().replace("\\", "/")
    candidates = [
        os.path.join(dataset_dir, rel),
        os.path.join(dataset_dir, DEFAULT_ASR_ROOT, rel),
        os.path.join(dataset_dir, rel.replace("Eval/", f"{DEFAULT_ASR_ROOT}/", 1)) if rel.startswith("Eval/") else "",
        os.path.join(dataset_dir, DEFAULT_ASR_ROOT, subset_dir, "wav", os.path.basename(rel)),
    ]
    for cand in candidates:
        if cand and os.path.isfile(cand):
            return os.path.abspath(cand)
    raise ValueError(
        f"Cannot resolve audio path for subset {subset_dir!r}: {rel_path!r}. Tried: "
        + ", ".join(repr(c) for c in candidates if c)
    )


def iter_subset_records(dataset_dir: str, subset_dir: str, language: str) -> Iterable[dict]:
    text_path = os.path.join(dataset_dir, DEFAULT_ASR_ROOT, subset_dir, "text")
    wav_scp_path = os.path.join(dataset_dir, DEFAULT_ASR_ROOT, subset_dir, "wav.scp")

    text_map = read_kaldi_mapping(text_path)
    wav_map = read_kaldi_mapping(wav_scp_path)

    text_keys = set(text_map)
    wav_keys = set(wav_map)
    if text_keys != wav_keys:
        missing_in_wav = sorted(text_keys - wav_keys)[:5]
        missing_in_text = sorted(wav_keys - text_keys)[:5]
        msg = []
        if missing_in_wav:
            msg.append(f"missing in wav.scp: {missing_in_wav}")
        if missing_in_text:
            msg.append(f"missing in text: {missing_in_text}")
        raise ValueError(
            f"Utterance ids do not match for subset {subset_dir!r}: " + "; ".join(msg)
        )

    for utt_id in sorted(text_map):
        yield {
            "audio": resolve_audio_path(dataset_dir, wav_map[utt_id], subset_dir),
            "text": normalize_target_text(text_map[utt_id], language),
        }


def take_limit(records: Iterable[dict], max_samples: int) -> Iterator[dict]:
    if max_samples <= 0:
        yield from records
        return
    for i, rec in enumerate(records):
        if i >= max_samples:
            return
        yield rec


def detect_duplicates(records: Sequence[dict]) -> List[str]:
    seen: Dict[str, int] = {}
    dup: List[str] = []
    for rec in records:
        ap = rec["audio"]
        seen[ap] = seen.get(ap, 0) + 1
        if seen[ap] == 2:
            dup.append(ap)
    return dup


def main() -> None:
    args = parse_args()
    dataset_dir = os.path.abspath(args.dataset_dir)
    output_jsonl = os.path.abspath(args.output_jsonl)

    if not args.skip_download:
        os.makedirs(dataset_dir, exist_ok=True)
        download_repo(args.repo_id, dataset_dir, args.subset)

    ensure_required_files(dataset_dir, args.subset)

    records: List[dict] = []
    chosen_dirs = subset_dirs(args.subset)
    for d in chosen_dirs:
        records.extend(iter_subset_records(dataset_dir, d, args.language))

    dups = detect_duplicates(records)
    if dups:
        raise SystemExit(
            f"Detected duplicate audio entries after merging subset={args.subset!r}. "
            f"Example duplicates: {dups[:5]}"
        )

    output_dir = os.path.dirname(output_jsonl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    count = 0
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for rec in take_limit(records, args.max_samples):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} samples to {output_jsonl}")
    print(f"Dataset directory: {dataset_dir}")
    print(f"ASR subsets: {chosen_dirs}")


if __name__ == "__main__":
    main()
