#!/usr/bin/env python3
"""Mix real target-domain clips into the synthetic radiology training manifest.

Real sources live under raw/POC_train/real_target_domain:
    raw       metadata_raw.jsonl       1,263 real segments (2.65 h)
    denoised  metadata_denoised.jsonl  same utterances, denoised
    0821      metadata_0821.jsonl       188 real pathology segments (0.52 h)

Each real clip is repeated --multiplier times (oversampling); --sim-multiplier 0
builds a real-only manifest. Rows are shuffled with --seed.

An optional real radiology source can be added with --youtube-manifest (rows use
keys audio/text/duration_sec, e.g. raw/POC_train/youtube_med_en/radiology.jsonl);
it is repeated --youtube-multiplier times.

    python tools/build_radiology_real_mix.py --multiplier 1 \
        --out-dir data/poc_train_radiology_mix_m1
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
REAL_DIR = REPO / "raw/POC_train/real_target_domain"
SIM_MANIFEST = REPO / "data/poc_train_tts_radiology_simA3x/train.jsonl"
SOURCES = {
    "raw": ("metadata_raw.jsonl", "from_lid_raw", "real_raw"),
    "denoised": ("metadata_denoised.jsonl", "from_lid_denoised", "real_denoised"),
    "0821": ("metadata_0821.jsonl", "pathology_0821", "real_0821"),
}
PREFIX = "language None<asr_text>"


def group_of(name: str) -> str:
    stem = Path(name).stem
    return stem.split("_seg_")[0].split("_")[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--multiplier", type=int, default=1, help="Real-clip repeat count.")
    ap.add_argument("--sim-multiplier", type=int, default=1, help="0 = real only.")
    ap.add_argument("--sources", default="raw,denoised,0821")
    ap.add_argument("--sim-manifest", default=str(SIM_MANIFEST))
    ap.add_argument("--youtube-manifest", default="", help="Optional real radiology jsonl (audio/text/duration_sec).")
    ap.add_argument("--youtube-multiplier", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20261028)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    if args.sim_multiplier:
        sim = [json.loads(l) for l in Path(args.sim_manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
        rows.extend(sim * args.sim_multiplier)
        sim_rows = len(sim) * args.sim_multiplier
    else:
        sim_rows = 0

    real_rows = 0
    by_source = Counter()
    real_dur = 0.0
    for tag in [s for s in args.sources.split(",") if s]:
        fname, source_group, sampling_source = SOURCES[tag]
        items = [json.loads(l) for l in (REAL_DIR / fname).read_text(encoding="utf-8").splitlines() if l.strip()]
        for item in items:
            audio = Path(item["audio_path"])
            if not audio.is_file():
                raise FileNotFoundError(audio)
            dur = float(item.get("duration") or 0.0)
            if dur <= 0:
                info = sf.info(str(audio))
                dur = info.frames / float(info.samplerate)
            for k in range(args.multiplier):
                rows.append(
                    {
                        "audio": str(audio),
                        "text": PREFIX + str(item["text"]).strip(),
                        "aug": 1,
                        "noise_aug": 0,
                        "sampling_source": sampling_source,
                        "source_group": group_of(audio.name),
                        "segment_id": f"{source_group}/{audio.stem}_r{k}",
                        "duration_sec": round(dur, 6),
                    }
                )
                real_rows += 1
                by_source[tag] += 1
                real_dur += dur

    youtube_rows = 0
    youtube_dur = 0.0
    if args.youtube_manifest:
        items = [json.loads(l) for l in Path(args.youtube_manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
        for item in items:
            audio = Path(item["audio"])
            if not audio.is_file():
                raise FileNotFoundError(audio)
            dur = float(item.get("duration_sec") or 0.0)
            if dur <= 0:
                info = sf.info(str(audio))
                dur = info.frames / float(info.samplerate)
            video = str(item.get("video_id") or audio.parent.name)
            seg = str(item.get("segment_id") or audio.stem)
            for k in range(args.youtube_multiplier):
                rows.append(
                    {
                        "audio": str(audio),
                        "text": PREFIX + str(item["text"]).strip(),
                        "aug": 1,
                        "noise_aug": 0,
                        "sampling_source": "youtube_radiology",
                        "source_group": video,
                        "segment_id": f"youtube_rad/{video}/{seg}_r{k}",
                        "duration_sec": round(dur, 6),
                    }
                )
                youtube_rows += 1
                youtube_dur += dur

    random.Random(args.seed).shuffle(rows)
    with (out / "train.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "sim_rows": sim_rows,
        "real_rows": real_rows,
        "real_share_by_rows": round(real_rows / max(len(rows), 1), 4),
        "real_hours_after_repeat": round(real_dur / 3600, 3),
        "multiplier": args.multiplier,
        "youtube_rows": youtube_rows,
        "youtube_share_by_rows": round(youtube_rows / max(len(rows), 1), 4),
        "youtube_hours_after_repeat": round(youtube_dur / 3600, 3),
        "youtube_multiplier": args.youtube_multiplier,
        "by_source": dict(by_source),
        "seed": args.seed,
    }
    (out / "mix_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"wrote {len(rows)} rows -> {out / 'train.jsonl'}")


if __name__ == "__main__":
    main()
