#!/usr/bin/env python3
"""Convert predictions.jsonl (reference_raw / hypothesis_raw) to a two-column CSV."""

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_jsonl",
        type=Path,
        nargs="?",
        default=Path("outputs/qwen3_asr_sft_ug2/predictions.jsonl"),
        help="Path to predictions.jsonl",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: same stem as input with .csv)",
    )
    args = parser.parse_args()
    in_path: Path = args.input_jsonl
    out_path: Path = args.output or in_path.with_suffix(".csv")

    rows_written = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        writer = csv.writer(fout)
        writer.writerow(["Reference", "Hypothesis"])
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ref = obj.get("reference_raw", "")
            hyp = obj.get("hypothesis_raw", "")
            writer.writerow([ref, hyp])
            rows_written += 1

    print(f"Wrote {rows_written} rows to {out_path}")


if __name__ == "__main__":
    main()
