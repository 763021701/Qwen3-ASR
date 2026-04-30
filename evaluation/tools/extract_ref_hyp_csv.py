#!/usr/bin/env python3
# coding=utf-8

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract reference_raw and hypothesis_for_scoring_raw from jsonl into a CSV file."
    )
    parser.add_argument("--input", required=True, help="Input predictions jsonl path.")
    parser.add_argument("--output", required=True, help="Output csv path.")
    parser.add_argument(
        "--reference_key",
        default="reference_raw",
        help="JSON key used for the CSV reference column.",
    )
    parser.add_argument(
        "--hypothesis_key",
        default="hypothesis_for_scoring_raw",
        help="JSON key used for the CSV hypothesis column.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        writer = csv.writer(fout)
        writer.writerow(["reference", "hypothesis"])

        for line_no, line in enumerate(fin, start=1):
            text = line.strip()
            if not text:
                continue

            record = json.loads(text)
            if args.reference_key not in record:
                raise KeyError(f"Missing key {args.reference_key!r} at line {line_no}")
            if args.hypothesis_key not in record:
                raise KeyError(f"Missing key {args.hypothesis_key!r} at line {line_no}")

            writer.writerow([record[args.reference_key], record[args.hypothesis_key]])
            row_count += 1

    print(f"Wrote {row_count} rows to {output_path}")


if __name__ == "__main__":
    main()
