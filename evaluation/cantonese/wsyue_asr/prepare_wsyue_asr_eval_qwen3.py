#!/usr/bin/env python3
# coding=utf-8
"""
Download ASLP-lab/WSYue-ASR-eval and convert it to Qwen3-ASR jsonl.

Output format per line:
  {"audio": "/abs/path/to/file.wav", "text": "language Cantonese<asr_text>..."}

By default this script:
  1. Downloads the dataset repo from Hugging Face
  2. Extracts the audio archives
  3. Parses:
       - Short/content.txt
       - Long/TextGrid.tar.gz
  4. Writes one combined jsonl manifest

Example:
  python evaluation/cantonese/wsyue_asr/prepare_wsyue_asr_eval_qwen3.py \
    --output_jsonl data/cantonese/wsyue_asr/wsyue_asr_eval_qwen3.jsonl \
    --dataset_dir /root/autodl-tmp/datasets/WSYue-ASR-eval
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tarfile
from typing import Dict, Iterable, Iterator, List, Tuple


DEFAULT_REPO_ID = "ASLP-lab/WSYue-ASR-eval"
_TEXT_PREFIX = "language {language}<asr_text>{text}"
_TEXTGRID_TEXT_RE = re.compile(r'^text = "(.*)"$')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare WSYue-ASR-eval as Qwen3-ASR jsonl for Cantonese evaluation."
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
        default=os.path.join(os.getcwd(), "downloads", "WSYue-ASR-eval"),
        help="Local directory for the downloaded WSYue-ASR-eval dataset repo.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        choices=("all", "short", "long"),
        help="Which subset(s) to convert.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="Cantonese",
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
        "--skip_extract",
        action="store_true",
        help="Assume audio archives were already extracted under dataset_dir/extracted.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="If >0, write at most N samples after conversion.",
    )
    return parser.parse_args()


def _allow_patterns_for_subset(subset: str) -> List[str]:
    if subset == "short":
        return ["README.md", "Short/*"]
    if subset == "long":
        return ["README.md", "Long/*"]
    return ["README.md", "Short/*", "Long/*"]


def download_repo(repo_id: str, dataset_dir: str, subset: str) -> None:
    try:
        from huggingface_hub import snapshot_download  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub first: pip install huggingface_hub") from exc

    allow_patterns = _allow_patterns_for_subset(subset)
    print(f"Downloading {repo_id} to {dataset_dir} ...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=dataset_dir,
        allow_patterns=allow_patterns,
    )


def safe_extract_tar(archive_path: str, dest_dir: str) -> None:
    print(f"Extracting {archive_path} -> {dest_dir}")
    os.makedirs(dest_dir, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tf:
        members = tf.getmembers()
        abs_dest = os.path.abspath(dest_dir)
        for member in members:
            member_path = os.path.abspath(os.path.join(dest_dir, member.name))
            if os.path.commonpath([abs_dest, member_path]) != abs_dest:
                raise ValueError(f"Unsafe path in archive {archive_path}: {member.name}")
        tf.extractall(dest_dir)


def ensure_required_files(dataset_dir: str, subset: str) -> None:
    needed: List[str] = []
    if subset in ("all", "short"):
        needed.extend(["Short/content.txt", "Short/wav.tar.gz"])
    if subset in ("all", "long"):
        needed.extend(["Long/TextGrid.tar.gz", "Long/wav.tar.gz"])
    missing = [rel for rel in needed if not os.path.exists(os.path.join(dataset_dir, rel))]
    if missing:
        raise SystemExit(f"Missing required dataset files under {dataset_dir}: {missing}")


def extract_archives_if_needed(dataset_dir: str, subset: str) -> str:
    extract_root = os.path.join(dataset_dir, "extracted")
    os.makedirs(extract_root, exist_ok=True)
    if subset in ("all", "short"):
        short_dest = os.path.join(extract_root, "Short")
        if not os.path.isdir(short_dest) or not any(True for _ in iter_audio_files(short_dest)):
            safe_extract_tar(os.path.join(dataset_dir, "Short", "wav.tar.gz"), short_dest)
    if subset in ("all", "long"):
        long_dest = os.path.join(extract_root, "Long")
        if not os.path.isdir(long_dest) or not any(True for _ in iter_audio_files(long_dest)):
            safe_extract_tar(os.path.join(dataset_dir, "Long", "wav.tar.gz"), long_dest)
    return extract_root


def iter_audio_files(root: str) -> Iterator[str]:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".wav"):
                yield os.path.join(dirpath, name)


def build_audio_index(*roots: str) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for root in roots:
        if not os.path.isdir(root):
            continue
        for path in iter_audio_files(root):
            name = os.path.basename(path)
            prev = index.get(name)
            if prev is not None and os.path.abspath(prev) != os.path.abspath(path):
                raise ValueError(f"Duplicate wav basename detected: {name}")
            index[name] = os.path.abspath(path)
    return index


def normalize_target_text(text: str, language: str) -> str:
    txt = (text or "").strip()
    if not txt:
        raise ValueError("Empty transcript is not allowed.")
    return _TEXT_PREFIX.format(language=language, text=txt)


def iter_short_records(content_path: str, audio_index: Dict[str, str], language: str) -> Iterable[dict]:
    with open(content_path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 4)
            if len(parts) != 5:
                raise ValueError(f"Invalid Short/content.txt line {line_no}: {line}")
            wav_name, _age, _emotion, _gender, transcript = parts
            audio_path = audio_index.get(wav_name)
            if not audio_path:
                raise ValueError(f"Cannot find extracted audio for Short sample {wav_name}")
            yield {
                "audio": audio_path,
                "text": normalize_target_text(transcript, language),
            }


def praat_unescape(s: str) -> str:
    return s.replace('""', '"')


def extract_text_from_textgrid_content(content: str) -> str:
    lines = content.splitlines()
    in_text_tier = False
    pieces: List[str] = []

    for raw in lines:
        line = raw.strip()
        if line.startswith("name = "):
            tier_name = line.split("=", 1)[1].strip().strip('"')
            if tier_name == "文本":
                in_text_tier = True
                continue
            if in_text_tier:
                break

        if not in_text_tier:
            continue

        match = _TEXTGRID_TEXT_RE.match(line)
        if not match:
            continue
        value = praat_unescape(match.group(1)).strip()
        if value:
            pieces.append(value)

    text = "".join(pieces).strip()
    if not text:
        raise ValueError("No transcript extracted from TextGrid 文本 tier.")
    return text


def iter_long_records(textgrid_tar_path: str, audio_index: Dict[str, str], language: str) -> Iterable[dict]:
    with tarfile.open(textgrid_tar_path, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile() or not member.name.endswith(".TextGrid"):
                continue
            stem = os.path.splitext(os.path.basename(member.name))[0]
            wav_name = f"{stem}.wav"
            audio_path = audio_index.get(wav_name)
            if not audio_path:
                raise ValueError(f"Cannot find extracted audio for Long sample {wav_name}")
            data = tf.extractfile(member)
            if data is None:
                raise ValueError(f"Cannot read TextGrid member: {member.name}")
            content = data.read().decode("utf-8", errors="replace")
            transcript = extract_text_from_textgrid_content(content)
            yield {
                "audio": audio_path,
                "text": normalize_target_text(transcript, language),
            }


def take_limit(records: Iterable[dict], max_samples: int) -> Iterator[dict]:
    if max_samples <= 0:
        yield from records
        return
    for i, rec in enumerate(records):
        if i >= max_samples:
            return
        yield rec


def main() -> None:
    args = parse_args()
    dataset_dir = os.path.abspath(args.dataset_dir)
    output_jsonl = os.path.abspath(args.output_jsonl)

    if not args.skip_download:
        os.makedirs(dataset_dir, exist_ok=True)
        download_repo(args.repo_id, dataset_dir, args.subset)

    ensure_required_files(dataset_dir, args.subset)

    extract_root = os.path.join(dataset_dir, "extracted")
    if not args.skip_extract:
        extract_root = extract_archives_if_needed(dataset_dir, args.subset)

    short_root = os.path.join(extract_root, "Short")
    long_root = os.path.join(extract_root, "Long")
    audio_index = build_audio_index(short_root, long_root)
    if not audio_index:
        raise SystemExit(f"No extracted wav files found under {extract_root}")

    records: List[dict] = []
    if args.subset in ("all", "short"):
        records.extend(
            iter_short_records(
                os.path.join(dataset_dir, "Short", "content.txt"),
                audio_index=audio_index,
                language=args.language,
            )
        )
    if args.subset in ("all", "long"):
        records.extend(
            iter_long_records(
                os.path.join(dataset_dir, "Long", "TextGrid.tar.gz"),
                audio_index=audio_index,
                language=args.language,
            )
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
    print(f"Audio root: {extract_root}")


if __name__ == "__main__":
    main()
