#!/usr/bin/env python3
"""
Verify Qwen3-ASR text tokenizer round-trip on Common Voice Uyghur test.tsv (column: sentence).

Usage:
  python finetuning/verify_tokenizer_cv_ug.py \\
    --tsv /path/to/ug/test.tsv \\
    --model_path Qwen/Qwen3-ASR-1.7B

Or with a local checkpoint that contains tokenizer files:
  python finetuning/verify_tokenizer_cv_ug.py --tsv ... --model_path outputs/.../checkpoint-69
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple


def load_sentences(tsv_path: Path) -> List[str]:
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "sentence" not in reader.fieldnames:
            raise ValueError(
                f"Expected column 'sentence' in TSV header, got: {reader.fieldnames}"
            )
        out: List[str] = []
        for row in reader:
            s = row.get("sentence", "")
            if s is None:
                s = ""
            out.append(s)
    return out


def verify_roundtrip(
    sentences: List[str],
    tokenizer,
) -> Tuple[int, int, List[Tuple[int, str, str]]]:
    """
    Returns: (ok_count, fail_count, failures as list of (line_index_1based, original, decoded))
    """
    unk_id = tokenizer.unk_token_id
    ok = 0
    failures: List[Tuple[int, str, str]] = []

    for i, s in enumerate(sentences, start=1):
        ids = tokenizer.encode(s, add_special_tokens=False)
        if unk_id is not None:
            n_unk = sum(1 for t in ids if t == unk_id)
            if n_unk > 0:
                failures.append((i, s, f"<{n_unk} UNK tokens>"))
                continue
        back = tokenizer.decode(ids, skip_special_tokens=True)
        if back != s:
            failures.append((i, s, back))
        else:
            ok += 1

    return ok, len(failures), failures


def main() -> None:
    p = argparse.ArgumentParser(description="Verify Qwen2 tokenizer on CV ug test.tsv sentences.")
    p.add_argument(
        "--tsv",
        type=Path,
        default=Path("/root/autodl-tmp/datasets/cv-corpus-25.0-2026-03-09/ug/test.tsv"),
        help="Path to Common Voice ug/test.tsv",
    )
    p.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen3-ASR-1.7B",
        help="HF model id or local dir with tokenizer (same as Qwen3ASR from_pretrained)",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="If >0, only check the first N rows (for quick smoke test)",
    )
    p.add_argument(
        "--list_preview",
        type=int,
        default=0,
        help="If >0, print that many example sentences from the TSV and exit",
    )
    args = p.parse_args()

    if not args.tsv.is_file():
        print(f"TSV not found: {args.tsv}", file=sys.stderr)
        sys.exit(1)

    sentences = load_sentences(args.tsv)
    if args.list_preview > 0:
        for j, s in enumerate(sentences[: args.list_preview], start=1):
            print(f"{j}\t{s}")
        return

    if args.max_samples > 0:
        sentences = sentences[: args.max_samples]

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    ok, n_fail, failures = verify_roundtrip(sentences, tok)
    total = len(sentences)
    print(f"model_path: {args.model_path}")
    print(f"tsv: {args.tsv}")
    print(f"rows_checked: {total}")
    print(f"roundtrip_ok: {ok}")
    print(f"roundtrip_fail_or_unk: {n_fail}")

    if failures:
        print("\nFirst 20 failures (line_in_tsv, original, decoded_or_note):")
        for item in failures[:20]:
            line, orig, dec = item
            print(f"--- line {line} ---")
            print(f"  orig: {orig!r}")
            print(f"  got:  {dec!r}")
        if len(failures) > 20:
            print(f"... and {len(failures) - 20} more")
        sys.exit(1)
    print("All sentences passed encode/decode round-trip (and no UNK when unk_token_id is set).")


if __name__ == "__main__":
    main()
