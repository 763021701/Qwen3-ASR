#!/usr/bin/env python3
"""Average full-finetune checkpoints into a new inferable checkpoint (simple mean).

All inputs must share the same architecture/key set (same base lineage). Weights are
averaged in float32 and written back in each tensor's dtype. HF/processor files are
copied from the first checkpoint.

    python tools/soup_checkpoints.py --out outputs/soup_x \
        outputs/run_a/checkpoint-564 outputs/run_b/checkpoint-188
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

COPY = [
    "config.json", "generation_config.json", "preprocessor_config.json",
    "processor_config.json", "tokenizer_config.json", "tokenizer.json",
    "special_tokens_map.json", "chat_template.json", "chat_template.jinja",
    "merges.txt", "vocab.json", "added_tokens.json",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("ckpts", nargs="+")
    args = ap.parse_args()
    srcs = [Path(c) for c in args.ckpts]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    acc: dict[str, torch.Tensor] = {}
    for i, src in enumerate(srcs):
        state = load_file(str(src / "model.safetensors"))
        for k, v in state.items():
            t = v.to(torch.float32)
            if k not in acc:
                acc[k] = t.clone()
            else:
                if acc[k].shape != t.shape:
                    raise ValueError(f"shape mismatch for {k}: {acc[k].shape} vs {t.shape}")
                acc[k] += t
        print(f"loaded {src} ({i + 1}/{len(srcs)})")

    n = float(len(srcs))
    ref = load_file(str(srcs[0] / "model.safetensors"))
    merged = {k: (acc[k] / n).to(ref[k].dtype) for k in acc}
    save_file(merged, str(out / "model.safetensors"), metadata={"format": "pt"})
    for name in COPY:
        for src in srcs:
            p = src / name
            if p.is_file():
                shutil.copy2(p, out / name)
                break
    print(f"soup of {len(srcs)} checkpoints -> {out}")


if __name__ == "__main__":
    main()