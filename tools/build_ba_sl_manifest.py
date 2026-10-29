#!/usr/bin/env python3
"""Rebuild the POC train manifest: CosyVoice B/A synth + SwitchLingua in, YouTube out.

Changes vs data/poc_train_fullvoice_clean_v2synth_yt_plus0821/train.jsonl (ckpt-3900 run):
- REMOVE sampling_source tts_pathology (30,976), llm_text_syn (3,718), youtube_med_en (23,813)
- ADD raw/POC_train/tts_pathology/B_{pathref,slref}.manifest.json (13,899 x2 voices) -> tts_pathology_cosy_b
- ADD raw/POC_train/llm_text_syn/A_{pathref,slref}.manifest.json (1,884 x2 voices)  -> llm_text_syn_cosy_a
- ADD raw/POC_train/switchlingua/switchlingua_cantonese_english_wav16k.jsonl (7,272) -> switchlingua_cant_en
- KEEP verbatim (raw line passthrough): real_raw/real_denoised (1,263 x2), real_0821_raw/denoised
  (188 x2), cantonese_measure (3,812), cantonese_specimen_number (3,056)
- dev/test (53, pathology.jsonl) copied unchanged.

Label convention (user decision 2026-10-16): B/A text is taken AS-IS from the CosyVoice
manifests (CJK digits, 乘, capitalised "Period." kept; only NFKC + whitespace collapse),
because pathref/slref voices read the digits in Chinese — same "label what the audio says"
precedent as cantonese_measure. The eval normalizer (round-2) already maps Chinese digits
-> Arabic and 乘/x -> x, so this stays WER-neutral. SwitchLingua rows already carry
language None<asr_text> and are used as-is. noise_aug=1 for all new rows (user decision:
SwitchLingua is clean read speech, treat like synthetic for online noising).

Leakage: B/A normalized texts vs the 53 dev/test refs measured 0 overlap (exact-normalized
and digit-free-core) before this script was written (see session 2026-10-16).
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
COSY_ROOT = Path("/root/autodl-tmp/workspace/project/CosyVoice/examples/switchlingua-yue")
ASR_PREFIX = "language None<asr_text>"
KEEP_SOURCES = {
    "real_raw",
    "real_denoised",
    "real_0821_raw",
    "real_0821_denoised",
    "cantonese_measure",
    "cantonese_specimen_number",
}


def label_body(text: str) -> str:
    s = unicodedata.normalize("NFKC", str(text or "").strip())
    s = " ".join(s.split())
    if not s:
        raise ValueError("empty transcript body")
    return s


def synth_rows(manifest: Path, group: str, variant: str) -> list[dict]:
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
                "segment_id": f"{group}/{variant}_{item['utt']}",
                "duration_sec": f"{round(float(item['synth_sec']), 6)}",
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base_dir", default="data/poc_train_fullvoice_clean_v2synth_yt_plus0821")
    ap.add_argument("--out_dir", default="data/poc_train_yue_ba_sl_no_yt_plus0821")
    args = ap.parse_args()

    base, out = Path(args.base_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    kept_lines: list[str] = []
    kept_sources: dict[str, int] = {}
    existing_audio: set[str] = set()
    for raw in (base / "train.jsonl").read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        src = json.loads(raw)["sampling_source"]
        if src in KEEP_SOURCES:
            kept_lines.append(raw)
            kept_sources[src] = kept_sources.get(src, 0) + 1
            existing_audio.add(json.loads(raw)["audio"])
        elif src not in {"tts_pathology", "llm_text_syn", "youtube_med_en"}:
            raise ValueError(f"unexpected sampling_source in base: {src}")

    new_rows: list[dict] = []
    new_rows += synth_rows(REPO / "raw/POC_train/tts_pathology/B_pathref.manifest.json", "tts_pathology_cosy_b", "pathref")
    new_rows += synth_rows(REPO / "raw/POC_train/tts_pathology/B_slref.manifest.json", "tts_pathology_cosy_b", "slref")
    new_rows += synth_rows(REPO / "raw/POC_train/llm_text_syn/A_pathref.manifest.json", "llm_text_syn_cosy_a", "pathref")
    new_rows += synth_rows(REPO / "raw/POC_train/llm_text_syn/A_slref.manifest.json", "llm_text_syn_cosy_a", "slref")

    sl_group = "switchlingua_cant_en"
    for line in (REPO / "raw/POC_train/switchlingua/switchlingua_cantonese_english_wav16k.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        audio = Path(item["audio"])
        if not audio.is_file():
            raise FileNotFoundError(f"switchlingua: missing wav {audio}")
        text = str(item["text"]).strip()
        if not text.startswith("language ") or "<asr_text>" not in text:
            raise ValueError(f"switchlingua row missing prefix: {text[:60]}")
        if not text.split("<asr_text>", 1)[1].strip():
            raise ValueError(f"switchlingua row empty transcript: {audio}")
        info = sf.info(str(audio))
        dur = info.frames / float(info.samplerate)
        new_rows.append(
            {
                "audio": str(audio),
                "text": text,
                "aug": "1",
                "noise_aug": "1",
                "sampling_source": sl_group,
                "source_group": sl_group,
                "segment_id": f"{sl_group}/{audio.stem}",
                "duration_sec": f"{round(dur, 6)}",
            }
        )

    dup = [r["audio"] for r in new_rows if r["audio"] in existing_audio]
    assert not dup, f"new audio already in kept rows: {dup[:3]}"
    segs = [r["segment_id"] for r in new_rows]
    assert len(segs) == len(set(segs)), "duplicate segment_id among new rows"

    with (out / "train.jsonl").open("w", encoding="utf-8") as h:
        for raw in kept_lines:
            h.write(raw + "\n")
        for row in new_rows:
            h.write(json.dumps(row, ensure_ascii=False) + "\n")

    for name in ("dev.jsonl", "test.jsonl"):
        (out / name).write_bytes((base / name).read_bytes())

    groups = dict(kept_sources)
    for r in new_rows:
        groups[r["sampling_source"]] = groups.get(r["sampling_source"], 0) + 1
    print(f"train: {len(kept_lines)} kept + {len(new_rows)} new = {len(kept_lines) + len(new_rows)}")
    for k, v in sorted(groups.items()):
        print(f"  {v:7d}  {k}")
    print(f"dev/test copied from {base}")


if __name__ == "__main__":
    main()
