#!/usr/bin/env python3
"""Append the pathology_0821 real/denoised pairs to the yt_fromscratch600 manifests.

Experiment (2026-09-10): train on raw/POC_test/pathology_0821.jsonl clips (moved
into training via metadata_0821_raw.jsonl + denoised metadata_0821.jsonl) with an
otherwise-identical yt_fromscratch600 setup. Dev/test (pathology.jsonl, 53 clips)
are untouched and share 0 segments AND 0 cases with 0821, so dev CER stays valid.

Rows are built with the SAME loader the pipeline uses (load_rows) so the format
is byte-identical to other real rows. sampling_source is tagged real_0821_raw /
real_0821_denoised for auditability (no training code keys on it). The yt
train/dev/test jsonls are copied verbatim.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO / "tools"))
from prepare_real_raw_denoised_sft import load_rows  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base_dir", default="data/poc_train_fullvoice_clean_v2synth_yt")
    ap.add_argument("--out_dir", default="data/poc_train_fullvoice_clean_v2synth_yt_plus0821")
    ap.add_argument("--raw_jsonl", default="raw/POC_train/real_target_domain/metadata_0821_raw.jsonl")
    ap.add_argument("--denoised_jsonl", default="raw/POC_train/real_target_domain/metadata_0821.jsonl")
    args = ap.parse_args()

    base, out = Path(args.base_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_rows = load_rows(Path(args.raw_jsonl).resolve(), sampling_source="real_0821_raw", noise_aug=0, check_audio=True)
    den_rows = load_rows(Path(args.denoised_jsonl).resolve(), sampling_source="real_0821_denoised", noise_aug=1, check_audio=True)
    assert len(raw_rows) == len(den_rows), f"pair misaligned: {len(raw_rows)} vs {len(den_rows)}"

    train_lines = (base / "train.jsonl").read_text(encoding="utf-8").rstrip("\n").split("\n")
    existing_audio = {json.loads(l)["audio"] for l in train_lines}
    dup = [r["audio"] for r in raw_rows + den_rows if r["audio"] in existing_audio]
    assert not dup, f"0821 audio already in train: {dup[:3]}"

    with (out / "train.jsonl").open("w", encoding="utf-8") as h:
        h.write("\n".join(train_lines) + "\n")
        for row in raw_rows + den_rows:
            h.write(json.dumps(row, ensure_ascii=False) + "\n")

    for name in ("dev.jsonl", "test.jsonl"):
        shutil.copyfile(base / name, out / name)

    total = len(train_lines) + len(raw_rows) + len(den_rows)
    print(f"train: {len(train_lines)} base + {len(raw_rows)} raw + {len(den_rows)} denoised = {total}")
    print(f"dev/test copied from {base}")


if __name__ == "__main__":
    main()
