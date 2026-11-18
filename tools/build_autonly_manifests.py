#!/usr/bin/env python3
"""Build manifests for the AuT-only then full-model two-stage run.

Stage 1 trains on real_target_domain (raw/denoised + 0821 raw/denoised) and
real_target_domain_vc (radiology + SwitchLingua voice clones).

Stage 2 trains on the existing unified-v2 pool (every current train source,
no hour subsampling) plus those voice-clone rows.

Labels stay on the unified-v2 contract: ``language None<asr_text>`` plus the
source transcript (NFKC and whitespace collapse for VC). Scoring-time
normalization stays in the eval scripts; this builder does not strip
punctuation, so stage 2 stays consistent with data/poc_train_unified_v2/all.jsonl.
"""
from __future__ import annotations

import json
import random
import sys
import unicodedata
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from prepare_real_raw_denoised_sft import load_rows  # noqa: E402

PREFIX = "language None<asr_text>"
VC_ROOT = Path("/root/autodl-tmp/workspace/project/CosyVoice/examples/voice_clone")
POOL = REPO / "data/poc_train_unified_v2/all.jsonl"
DEV = REPO / "data/poc_train_unified_v2/dev.jsonl"
SEED = 20261116


def label_body(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or "").strip())
    value = " ".join(value.split())
    if not value:
        raise ValueError("empty transcript")
    return value


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def hours(rows: list[dict]) -> float:
    return sum(float(row.get("duration_sec") or 0.0) for row in rows) / 3600.0


def load_real() -> list[dict]:
    root = REPO / "raw/POC_train/real_target_domain"
    specs = (
        ("metadata_raw.jsonl", "real_raw", 0),
        ("metadata_denoised.jsonl", "real_denoised", 1),
        ("metadata_0821_raw.jsonl", "real_0821_raw", 0),
        ("metadata_0821.jsonl", "real_0821_denoised", 1),
    )
    rows = []
    for name, source, noise_aug in specs:
        loaded = load_rows(root / name, sampling_source=source, noise_aug=noise_aug, check_audio=True)
        for row in loaded:
            row["segment_id"] = f"{source}/{row['segment_id']}"
        rows.extend(loaded)
        print(f"[real] {source:20s} rows={len(loaded):5d} hours={hours(loaded):.3f}")
    return rows


def load_vc(dev_audio: set[str]) -> tuple[list[dict], dict]:
    root = REPO / "raw/POC_train/real_target_domain_vc"
    dev_stems = {Path(audio).stem for audio in dev_audio}
    specs = (
        ("vc_radiology.manifest.json", "vc_radiology"),
        ("vc_sl_globe.manifest.json", "vc_sl_globe"),
    )
    rows = []
    report = {"missing_audio": 0, "dropped_poc53_source": 0, "empty_text": 0}
    for name, source in specs:
        items = json.loads((root / name).read_text(encoding="utf-8"))
        kept = 0
        for item in items:
            source_wav = str(item.get("source_wav") or "")
            if source_wav in dev_audio or Path(source_wav).stem in dev_stems:
                raise RuntimeError(
                    f"{name} still contains a POC53 source clone: {item.get('utt')}"
                )
            audio = (VC_ROOT / item["synth_wav"]).resolve()
            if not audio.is_file():
                report["missing_audio"] += 1
                continue
            try:
                text = label_body(item.get("text"))
            except ValueError:
                report["empty_text"] += 1
                continue
            rows.append({
                "audio": str(audio),
                "text": f"{PREFIX}{text}",
                "aug": 1,
                "noise_aug": 1,
                "sampling_source": source,
                "source_group": source,
                "segment_id": f"{source}/{item.get('utt') or audio.stem}",
                "duration_sec": round(float(item["synth_sec"]), 6),
                "voice": str(item.get("prompt_speaker") or ""),
            })
            kept += 1
        print(f"[vc]   {source:20s} rows={kept:5d} hours={hours(rows[-kept:]):.3f}")
    if report["missing_audio"] or report["empty_text"] or report["dropped_poc53_source"]:
        print(f"[vc]   filtered {report}")
    if report["missing_audio"] > 0.01 * (len(rows) + report["missing_audio"]):
        raise RuntimeError(f"too many missing VC wavs: {report}")
    return rows, report


def by_source(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["sampling_source"], []).append(row)
    return {
        source: {"rows": len(group), "hours": round(hours(group), 3)}
        for source, group in sorted(grouped.items())
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default="data/poc_autonly")
    args = parser.parse_args()
    out_root = (REPO / args.out_root).resolve()
    if not POOL.is_file() or not DEV.is_file():
        raise FileNotFoundError("unified-v2 pool/dev missing; build data/poc_train_unified_v2 first")
    dev = read_jsonl(DEV)
    if len(dev) != 53:
        raise ValueError(f"expected 53 POC dev rows, got {len(dev)}")
    dev_audio = {row["audio"] for row in dev}

    real = load_real()
    vc, vc_report = load_vc(dev_audio)
    stage1 = real + vc
    overlap = dev_audio & {row["audio"] for row in stage1}
    if overlap:
        raise RuntimeError(f"stage1 audio overlaps POC53: {list(overlap)[:3]}")
    dup = [audio for audio, count in Counter(row["audio"] for row in stage1).items() if count > 1]
    if dup:
        raise RuntimeError(f"duplicate stage1 audio: {dup[:3]}")

    pool = read_jsonl(POOL)
    pool_audio = {row["audio"] for row in pool}
    vc_new = [row for row in vc if row["audio"] not in pool_audio]
    stage2 = pool + vc_new
    if dev_audio & {row["audio"] for row in stage2}:
        raise RuntimeError("stage2 audio overlaps POC53")

    rng = random.Random(SEED)
    rng.shuffle(stage1)
    rng.shuffle(stage2)

    out1 = out_root / "stage1_real_vc"
    out2 = out_root / "stage2_all"
    write_jsonl(out1 / "train.jsonl", stage1)
    write_jsonl(out1 / "dev.jsonl", dev)
    write_jsonl(out1 / "test.jsonl", dev)
    write_jsonl(out2 / "train.jsonl", stage2)
    write_jsonl(out2 / "dev.jsonl", dev)
    write_jsonl(out2 / "test.jsonl", dev)

    stats = {
        "seed": SEED,
        "vc_filter": vc_report,
        "stage1": {
            "rows": len(stage1),
            "hours": round(hours(stage1), 3),
            "by_source": by_source(stage1),
        },
        "stage2": {
            "rows": len(stage2),
            "hours": round(hours(stage2), 3),
            "pool_rows": len(pool),
            "vc_rows_added": len(vc_new),
            "by_source": by_source(stage2),
        },
    }
    (out1 / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out2 / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[stage1] rows={stats['stage1']['rows']} hours={stats['stage1']['hours']}")
    print(f"[stage2] rows={stats['stage2']['rows']} hours={stats['stage2']['hours']} "
          f"vc_added={len(vc_new)}")


if __name__ == "__main__":
    main()
