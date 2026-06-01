#!/usr/bin/env python3
# coding=utf-8
"""Convert SwitchLingua manifest audio from m4a to 16 kHz mono wav."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert SwitchLingua m4a audio to 16 kHz mono wav.")
    parser.add_argument("--input_jsonl", required=True, help="Input SwitchLingua JSONL.")
    parser.add_argument("--output_jsonl", required=True, help="Output JSONL with wav paths.")
    parser.add_argument("--output_dir", required=True, help="Directory for converted wav files.")
    parser.add_argument("--overwrite", type=int, default=0, choices=(0, 1), help="Overwrite existing wav files.")
    parser.add_argument("--progress_every", type=int, default=250, help="Progress logging interval.")
    return parser.parse_args()


def load_records(path: Path) -> list[tuple[int, dict, Path, Path]]:
    rows: list[tuple[int, dict, Path, Path]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            item = json.loads(line)
            src = Path(str(item.get("audio") or ""))
            if not src.is_file():
                raise FileNotFoundError(f"missing audio at line {line_no}: {src}")
            rows.append((line_no, item, src, Path(src.stem + ".wav")))
    return rows


def convert_one(src: Path, dst: Path, overwrite: bool) -> str:
    if dst.is_file() and dst.stat().st_size > 44 and not overwrite:
        return "skipped_existing"
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    return "converted"


def main() -> None:
    args = parse_args()
    input_jsonl = Path(args.input_jsonl)
    output_jsonl = Path(args.output_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_records(input_jsonl)
    converted = 0
    skipped = 0
    failed: list[dict[str, object]] = []
    rewritten: list[dict] = []

    for idx, (line_no, item, src, rel_dst) in enumerate(rows, start=1):
        dst = output_dir / rel_dst
        try:
            status = convert_one(src, dst, overwrite=bool(args.overwrite))
        except subprocess.CalledProcessError as exc:
            failed.append({"line": line_no, "audio": str(src), "returncode": exc.returncode})
            if len(failed) >= 20:
                break
        else:
            if status == "converted":
                converted += 1
            else:
                skipped += 1
            new_item = dict(item)
            new_item["audio"] = str(dst.resolve())
            rewritten.append(new_item)

        if idx % args.progress_every == 0 or idx == len(rows):
            print(
                json.dumps(
                    {
                        "processed": idx,
                        "total": len(rows),
                        "converted": converted,
                        "skipped_existing": skipped,
                        "failed": len(failed),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    if failed:
        print(json.dumps({"failed_examples": failed[:20]}, ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(1)

    tmp_path = output_jsonl.with_suffix(output_jsonl.suffix + ".tmp")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("w", encoding="utf-8") as f:
        for item in rewritten:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.replace(tmp_path, output_jsonl)

    print(
        json.dumps(
            {
                "done": True,
                "input_records": len(rows),
                "converted": converted,
                "skipped_existing": skipped,
                "output_jsonl": str(output_jsonl.resolve()),
                "output_dir": str(output_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
