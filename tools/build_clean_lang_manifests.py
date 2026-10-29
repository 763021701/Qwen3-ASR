#!/usr/bin/env python3
"""Build cleaned radiology manifests with a controlled language prior.

Steps:
  * Decontaminate TTS texts: drop any sentence sharing an 8-gram with the
    radiology95 references (train76 + holdout19).  177/4001 ep1 texts are
    contaminated (57 of them Cantonese); RADX has zero overlap.
  * TTS base: ep1 (all clean texts) + RADX_slref (16,032 clean texts).
  * Cantonese rows keep three voices: ep1 + RAD_slref + RADP_pathref, selected
    by the Cantonese-particle text set so English texts are not duplicated.
  * YouTube: keep only radiology-findings rows (no second-person/teaching
    markers, terminal punctuation, a radiology term, unique text, sane
    chars/sec).

Writes JSONL inputs for mega-asr-sim plus the filtered YouTube manifest:

    python tools/build_clean_lang_manifests.py --out-dir data/radiology_clean_lang
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "finetuning"))

from evaluation.english_medical.text_normalization import normalize_english  # noqa: E402

TTS_DIR = REPO / "raw/POC_train/tts_radiology"
YOUTUBE = REPO / "raw/POC_train/youtube_med_en/radiology.jsonl"
EVAL_MANIFESTS = [REPO / "data/poc_radiology_oracle/train.jsonl",
                  REPO / "data/poc_radiology_oracle/holdout.jsonl"]
CANTON = re.compile(r"[嘅嗰唔係咁啦啊誒冇咗喺]")
TEACH = re.compile(
    r"\b(you|your|yours|we|us|our|let s|let us|please|remember|video|subscribe|"
    r"channel|today|question|questions|presentation|lecture|case number|thank)\b", re.I)
RAD = re.compile(
    r"\b(nodule|nodules|lesion|lesions|lobe|lobes|mass|masses|opacity|opacities|"
    r"effusion|consolidation|atelecta\w*|lymph|mediastin\w*|pleura\w*|parenchym\w*|"
    r"hyperdens\w*|hypodens\w*|calcif\w*|granuloma|cyst|cysts|thickening|stenosis|"
    r"hernia\w*|fracture\w*|tumor|tumour|metasta\w*|carcinoma|adenopathy|bronch\w*|"
    r"pulmonary|hepatic|renal|pancrea\w*|spleen|thyroid|disc|vertebra\w*|spine|"
    r"joint|tendon|ligament|artery|vein|aneurysm)\b", re.I)


def ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    words = text.split()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def eval_grams() -> set[tuple[str, ...]]:
    grams: set[tuple[str, ...]] = set()
    for path in EVAL_MANIFESTS:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            text = json.loads(line)["text"].split("<asr_text>")[-1]
            grams |= ngrams(normalize_english(text))
    return grams


def load_tts(name: str) -> tuple[list[dict], Path]:
    manifest = TTS_DIR / f"{name}.manifest.json"
    target = Path(os.path.realpath(manifest))
    root = target.parents[2]  # .../examples/switchlingua-yue
    rows = json.loads(target.read_text(encoding="utf-8"))
    out = []
    for item in rows:
        audio = root / item["synth_wav"]
        out.append({"audio": str(audio), "text": str(item["text"]).strip(),
                    "utt": item.get("utt", audio.stem)})
    return out, root


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="data/radiology_clean_lang")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    bad_grams = eval_grams()
    stats: dict[str, dict] = {}

    def write_jsonl(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def clean_tts(name: str, keep_texts: set[str] | None = None) -> list[dict]:
        rows, _ = load_tts(name)
        kept, seen = [], set()
        contam = missing = short = dup = 0
        for row in rows:
            text = row["text"]
            key = normalize_english(text)
            if keep_texts is not None and text not in keep_texts:
                continue
            if not os.path.isfile(row["audio"]):
                missing += 1
                continue
            if len(key.split()) < 3:
                short += 1
                continue
            if key in seen:
                dup += 1
                continue
            if ngrams(key) & bad_grams:
                contam += 1
                continue
            seen.add(key)
            kept.append(row)
        stats[name] = {"rows": len(rows), "kept": len(kept), "contaminated": contam,
                       "missing_audio": missing, "short": short, "dup": dup}
        return kept

    ep1 = clean_tts("ep1")
    # Cantonese text set after decontamination: same texts across the three voices.
    canto_texts = {row["text"] for row in ep1 if CANTON.search(row["text"])}
    rad_slref = clean_tts("RAD_slref", canto_texts)
    radp = clean_tts("RADP_pathref", canto_texts)
    radx = clean_tts("RADX_slref")

    for name, rows in [("ep1", ep1), ("canto_RAD_slref", rad_slref),
                       ("canto_RADP_pathref", radp), ("radx", radx)]:
        write_jsonl(out / f"clean_{name}.jsonl", rows)

    # YouTube: radiology-findings subset only.
    yt_rows = [json.loads(line) for line in YOUTUBE.read_text(encoding="utf-8").splitlines() if line.strip()]
    yt_kept, seen = [], set()
    drop = Counter()
    for row in yt_rows:
        text = str(row.get("text", "")).strip()
        key = normalize_english(text)
        dur = float(row.get("duration_sec") or 0)
        if not os.path.isfile(row["audio"]):
            drop["missing_audio"] += 1
        elif not text or len(text) < 20 or dur <= 0:
            drop["short"] += 1
        elif TEACH.search(text):
            drop["teaching"] += 1
        elif text[-1] not in ".!?\"'":
            drop["no_terminal_punct"] += 1
        elif not RAD.search(text):
            drop["no_radiology_term"] += 1
        elif not (5.0 <= len(text) / dur <= 25.0):
            drop["chars_per_sec"] += 1
        elif key in seen:
            drop["dup_text"] += 1
        elif ngrams(key) & bad_grams:
            drop["eval_overlap"] += 1
        else:
            seen.add(key)
            yt_kept.append({"audio": row["audio"], "text": text,
                            "duration_sec": dur, "video_id": row.get("video_id"),
                            "segment_id": row.get("segment_id")})
    write_jsonl(out / "youtube_strict.jsonl", yt_kept)
    stats["youtube"] = {"rows": len(yt_rows), "kept": len(yt_kept),
                        "hours_kept": round(sum(r["duration_sec"] for r in yt_kept) / 3600, 2),
                        "dropped": dict(drop)}

    stats["totals"] = {
        "ep1_rows": len(ep1), "ep1_canto_rows": len(canto_texts),
        "canto_slref_rows": len(rad_slref), "canto_pathref_rows": len(radp),
        "radx_rows": len(radx), "youtube_rows": len(yt_kept),
        "sim_input_rows": len(ep1) + len(rad_slref) + len(radp) + len(radx),
    }
    (out / "clean_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
