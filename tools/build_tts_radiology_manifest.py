#!/usr/bin/env python3
"""Append tts_radiology CosyVoice synth rows to the plus0821 champion mix (experiment B).

Mix = data/poc_train_fullvoice_clean_v2synth_yt_plus0821/train.jsonl rows whose audio still
        exists on disk (23,383/23,813 youtube_med_en wavs were deleted after the champion
        run and drop out; 44,894 rows kept)
    + raw/POC_train/tts_radiology/RAD_slref.manifest.json   (4,001) -> tts_radiology_slref
    + raw/POC_train/tts_radiology/RADP_pathref.manifest.json(4,001) -> tts_radiology_pathref
= 76,279 rows (radiology synth share 10.5%). dev/test: the plus0821 53-clip manifests copied.

Labels are the CosyVoice text AS-IS (CJK numerals such as T十二/L五至S一, 跟住/啦 discourse
markers and punctuation kept; only NFKC + whitespace collapse) — same convention as the
B/A pathref-slref rows: the voices read CJK digits in Chinese and the eval normalizer maps
them back. noise_aug=1 (synthetic audio).

Leakage (2026-10-23): 0 of 4,001 normalized texts match the radiology oracle holdout
recordings DocB/1866S / DocC/1435R (19 segments); only discourse-marker fragments
("next slide", "impression") occur inside their long transcripts.

Also writes the report-only eval CSV for the holdout (extra_eval_set consumers do NOT
strip the language prefix from jsonl refs, so the eval manifest must be plain CSV).
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COSY_ROOT = Path("/root/autodl-tmp/workspace/project/CosyVoice/examples/switchlingua-yue")
ASR_PREFIX = "language None<asr_text>"
VOICES = {
    "tts_radiology_slref": ("RAD_slref", "raw/POC_train/tts_radiology/RAD_slref.manifest.json"),
    "tts_radiology_pathref": ("RADP_pathref", "raw/POC_train/tts_radiology/RADP_pathref.manifest.json"),
}


def label_body(text: str) -> str:
    s = unicodedata.normalize("NFKC", str(text or "").strip())
    s = " ".join(s.split())
    if not s:
        raise ValueError("empty transcript body")
    return s


def synth_rows(group: str, manifest: Path) -> list[dict]:
    out = []
    for item in json.loads(manifest.read_text(encoding="utf-8")):
        audio = (COSY_ROOT / item["synth_wav"]).resolve()
        if not audio.is_file():
            raise FileNotFoundError(f"{manifest.name}: missing wav {audio}")
        out.append(
            {
                "audio": str(audio),
                "text": ASR_PREFIX + label_body(item["text"]),
                "aug": "1",
                "noise_aug": "1",
                "sampling_source": group,
                "source_group": group,
                "segment_id": f"{group}/{item['utt']}",
                "duration_sec": f"{round(float(item['synth_sec']), 6)}",
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base_dir", default="data/poc_train_fullvoice_clean_v2synth_yt_plus0821")
    ap.add_argument("--out_dir", default="data/poc_train_plus0821_ttsradiology")
    ap.add_argument("--holdout_csv", default="data/poc_radiology_oracle_eval/holdout19.csv")
    ap.add_argument("--synth_only", type=int, default=0, choices=(0, 1),
                    help="train.jsonl = tts_radiology rows only (synth oracle from base); no mainline replay.")
    ap.add_argument("--voices", default="",
                    help="Optional comma list NAME=MANIFEST to override the default RAD/RADP voices, "
                         "e.g. ep1=raw/POC_train/tts_radiology/ep1.manifest.json")
    args = ap.parse_args()

    base, out = Path(args.base_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not args.synth_only:
        for src in ("dev", "test"):
            (out / f"{src}.jsonl").write_text(
                "".join(ln + "\n" for ln in (base / f"{src}.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()),
                encoding="utf-8",
            )

    new_rows = []
    counts = {}
    voices = dict(VOICES)
    if args.voices.strip():
        voices = {}
        for spec in args.voices.split(","):
            name, _, rel = spec.partition("=")
            if not name.strip() or not rel.strip():
                raise ValueError(f"bad --voices entry {spec!r}; expected NAME=MANIFEST")
            voices[name.strip()] = ("", rel.strip())
    for group, (_variant, rel) in voices.items():
        rows = synth_rows(group, REPO / rel)
        counts[group] = len(rows)
        new_rows.extend(rows)

    train_lines = []
    dropped = {}
    if not args.synth_only:
        for ln in (base / "train.jsonl").read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            row = json.loads(ln)
            if not Path(row["audio"]).is_file():
                dropped[row.get("sampling_source", "?")] = dropped.get(row.get("sampling_source", "?"), 0) + 1
                continue
            train_lines.append(ln)

    with (out / "train.jsonl").open("w", encoding="utf-8") as handle:
        for ln in train_lines:
            handle.write(ln + "\n")
        for row in new_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Plain-text CSV of the oracle holdout for report-only extra eval.
    holdout_src = REPO / "data/poc_radiology_oracle/holdout.jsonl"
    csv_path = Path(args.holdout_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write("audio_path,text\n")
        n_csv = 0
        for line in holdout_src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            body = item["text"].split("<asr_text>", 1)[1].strip()
            handle.write(f'{item["audio"]},"{body.replace(chr(34), chr(34) * 2)}"\n')
            n_csv += 1

    total = len(train_lines) + len(new_rows)
    print(f"train rows: {len(train_lines)} base + {len(new_rows)} new = {total}")
    if dropped:
        print(f"dropped base rows with missing audio: {sum(dropped.values())} -> {dropped}")
    print("new by source:", counts)
    print(f"radiology synth share: {len(new_rows) / total * 100:.1f}%")
    print(f"wrote {out / 'train.jsonl'} and {csv_path} ({n_csv} rows)")


if __name__ == "__main__":
    main()