#!/usr/bin/env python3
"""Build a training manifest from mega-asr-sim output directories.

Usage:
    python tools/build_sim_radiology_manifest.py --out_dir data/poc_train_tts_radiology_simA \
        ep1=data/sim_radiology/simA_ep1 rad_slref=data/sim_radiology/simA_rad_slref \
        radp_pathref=data/sim_radiology/simA_radp_pathref

Each simulator out-dir must contain manifest.jsonl ({audio(rel), text, scenario, severity}).
Rows are converted to the project contract: "language None<asr_text>{text}", aug=1,
noise_aug=0 (audio is already degraded), sampling_source=sim_<tag>_<scenario>.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import soundfile as sf

ASR_PREFIX = "language None<asr_text>"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_severity", type=float, default=-0.01)
    ap.add_argument("--max_severity", type=float, default=1.01,
                    help="Drop simulated rows outside [min,max] severity (Mega-ASR learnability idea).")
    ap.add_argument("specs", nargs="+", help="TAG=SIM_OUT_DIR")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    by_scenario = Counter()
    dur = 0.0
    skipped_severity = 0
    for spec in args.specs:
        tag, _, path = spec.partition("=")
        sim_dir = Path(path)
        for line in (sim_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            sev = float(item.get("severity", 0.0))
            if not (args.min_severity <= sev <= args.max_severity):
                skipped_severity += 1
                continue
            audio = (sim_dir / item["audio"]).resolve()
            if not audio.is_file():
                raise FileNotFoundError(audio)
            info = sf.info(str(audio))
            d = info.frames / float(info.samplerate)
            rows.append(
                {
                    "audio": str(audio),
                    "text": ASR_PREFIX + str(item["text"]).strip(),
                    "aug": "1",
                    "noise_aug": "0",
                    "sampling_source": f"sim_{tag}_{item['scenario']}",
                    "source_group": f"sim_{tag}",
                    "segment_id": f"{tag}/{item['scenario']}/{audio.stem}",
                    "duration_sec": f"{round(d, 6)}",
                }
            )
            by_scenario[f"{tag}:{item['scenario']}"] += 1
            dur += d

    with (out / "train.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"train rows: {len(rows)} | {dur / 3600:.2f}h")
    if skipped_severity:
        print(f"  dropped by severity filter: {skipped_severity}")
    for k, v in sorted(by_scenario.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()