#!/usr/bin/env python3
"""Map saved checkpoints to their training epoch.

Scans ``--output_dir`` for ``checkpoint-*`` directories, reads each
``trainer_state.json`` (written by HF Trainer on every save), and emits a
``epoch -> checkpoint_dir`` mapping. Used to confirm that a 3-epoch run
produced one checkpoint at the end of each epoch.

Usage:
    python tools/map_epoch_checkpoints.py --output_dir outputs/test_samples_seg_sft
"""

from __future__ import annotations

import argparse
import json
import os
import re

_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.isdir(args.output_dir):
        raise SystemExit(f"output_dir not found: {args.output_dir}")

    rows = []
    for name in sorted(os.listdir(args.output_dir)):
        m = _CKPT_RE.match(name)
        if not m:
            continue
        step = int(m.group(1))
        ckpt_dir = os.path.join(args.output_dir, name)
        state_path = os.path.join(ckpt_dir, "trainer_state.json")
        epoch = None
        if os.path.isfile(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            epoch = state.get("epoch")
        rows.append({"step": step, "epoch": epoch, "checkpoint": ckpt_dir})

    rows.sort(key=lambda r: r["step"])

    by_epoch = {}
    for r in rows:
        ep = r["epoch"]
        ep_int = int(round(ep)) if isinstance(ep, (int, float)) else None
        by_epoch[ep_int] = r["checkpoint"]

    report = {
        "output_dir": os.path.abspath(args.output_dir),
        "num_checkpoints": len(rows),
        "checkpoints": rows,
        "epoch_to_checkpoint": by_epoch,
    }
    out_path = os.path.join(args.output_dir, "epoch_checkpoints.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
