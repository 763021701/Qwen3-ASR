#!/usr/bin/env python3
"""Write a resume variant of a pipeline YAML config.

The training driver needs model_path to point at the same checkpoint it resumes
from (the Trainer checkpoint omits the tied lm_head), so both fields are set to
the checkpoint. An explicit resume_from is used instead of "latest numeric
checkpoint" so a hardlinked checkpoint-latest can be selected.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--init-from", default="")
    args = ap.parse_args()

    text = Path(args.base).read_text(encoding="utf-8")
    text = re.sub(r"(?m)^  resume_from: .*\n?", "", text)
    if args.checkpoint:
        replacement = f"  model_path: {args.checkpoint}\n  resume_from: {args.checkpoint}"
    elif args.init_from:
        replacement = f"  model_path: {args.init_from}"
    else:
        replacement = None
    if replacement is not None:
        text, n = re.subn(r"(?m)^  model_path: .*$", replacement, text, count=1)
        if n != 1:
            raise SystemExit("model_path line not found")
    text, _ = re.subn(r"(?m)^  resume: (?:true|false)$", "  resume: false", text, count=1)
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"[resume-config] {args.out} checkpoint={args.checkpoint or '-'} init={args.init_from or '-'}")


if __name__ == "__main__":
    main()
