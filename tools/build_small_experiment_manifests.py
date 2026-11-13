#!/usr/bin/env python3
"""Build manifests for the three small-data SFT probes.

    realraw_noaug      real_target_domain raw channels only, noise_aug=0
    realdenoise_aug    real_target_domain denoised channels only, noise_aug=1
    synth_ab_aug       CosyVoice A_slref (llm_text_syn) + B_pathref (tts_pathology)

Dev/test are the new-label POC53 manifests from data/poc_train_unified_v2.
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))
from prepare_real_raw_denoised_sft import load_rows  # noqa: E402
from evaluation.english_medical.text_normalization import normalize_english  # noqa: E402

PREFIX = "language None<asr_text>"
COSY_ROOT = Path("/root/autodl-tmp/workspace/project/CosyVoice/examples/switchlingua-yue")


def body(text: str) -> str:
    return (text or "").split("<asr_text>", 1)[-1].strip()


def label_body(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or "").strip())
    value = " ".join(value.split())
    if not value:
        raise ValueError("empty transcript")
    return value


def cosy_rows(manifest: Path, source: str, variant: str) -> list[dict]:
    rows = []
    for item in json.loads(manifest.read_text(encoding="utf-8")):
        audio = (COSY_ROOT / item["synth_wav"]).resolve()
        if not audio.is_file():
            raise FileNotFoundError(f"{manifest.name}: missing {audio}")
        rows.append({
            "audio": str(audio),
            "text": f"{PREFIX}{label_body(item['text'])}",
            "aug": 1,
            "noise_aug": 1,
            "sampling_source": source,
            "source_group": source,
            "segment_id": f"synth_ab/{variant}_{item['utt']}",
            "duration_sec": round(float(item["synth_sec"]), 6),
        })
    return rows


def grams(text: str, n: int = 8) -> set[str]:
    tokens = text.split()
    return {" ".join(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1))}


def report(rows: list[dict], dev: list[dict]) -> dict:
    dev_norm = {normalize_english(body(row["text"])) for row in dev}
    dev_grams = set().union(*(grams(text) for text in dev_norm))
    dev_audio = {row["audio"] for row in dev}
    by_source = Counter()
    hours = Counter()
    for row in rows:
        by_source[row["sampling_source"]] += 1
        hours[row["sampling_source"]] += float(row["duration_sec"]) / 3600.0
    exact = sum(1 for row in rows if normalize_english(body(row["text"])) in dev_norm)
    gram8 = sum(1 for row in rows if grams(normalize_english(body(row["text"]))) & dev_grams)
    return {
        "rows": len(rows),
        "hours": round(sum(hours.values()), 3),
        "by_source": {k: {"rows": by_source[k], "hours": round(hours[k], 3)} for k in sorted(by_source)},
        "unique_texts": len({normalize_english(body(row["text"])) for row in rows}),
        "leakage_exact": exact,
        "leakage_8gram": gram8,
        "dev_audio_overlap": sum(1 for row in rows if row["audio"] in dev_audio),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default="data/poc_small_experiments")
    ap.add_argument("--dev-dir", default="data/poc_train_unified_v2")
    ap.add_argument("--check-audio", type=int, default=1)
    args = ap.parse_args()
    out = Path(args.out_root).resolve()
    dev = [json.loads(line) for line in Path(args.dev_dir, "dev.jsonl").read_text(encoding="utf-8").splitlines()
           if line.strip()]
    real = REPO / "raw/POC_train/real_target_domain"

    exp1 = (
        load_rows(real / "metadata_raw.jsonl", sampling_source="real_raw", noise_aug=0,
                  check_audio=bool(args.check_audio))
        + load_rows(real / "metadata_0821_raw.jsonl", sampling_source="real_0821_raw", noise_aug=0,
                    check_audio=bool(args.check_audio))
    )
    exp2 = (
        load_rows(real / "metadata_denoised.jsonl", sampling_source="real_denoised", noise_aug=1,
                  check_audio=bool(args.check_audio))
        + load_rows(real / "metadata_0821.jsonl", sampling_source="real_0821_denoised", noise_aug=1,
                    check_audio=bool(args.check_audio))
    )
    exp3 = (
        cosy_rows(REPO / "raw/POC_train/llm_text_syn/A_slref.manifest.json", "llm_text_syn_a_slref", "a_slref")
        + cosy_rows(REPO / "raw/POC_train/tts_pathology/B_pathref.manifest.json", "tts_pathology_b_pathref", "b_pathref")
    )

    stats = {}
    for name, rows in (("realraw_noaug", exp1), ("realdenoise_aug", exp2), ("synth_ab_aug", exp3)):
        seg_ids = [row["segment_id"] for row in rows]
        if len(seg_ids) != len(set(seg_ids)):
            raise ValueError(f"{name}: duplicate segment ids")
        directory = out / name
        write_jsonl(directory / "train.jsonl", rows)
        for split in ("dev", "test"):
            (directory / f"{split}.jsonl").write_bytes(Path(args.dev_dir, f"{split}.jsonl").read_bytes())
        stats[name] = report(rows, dev)
        print(f"[{name}] {json.dumps(stats[name], ensure_ascii=False)}")

    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
