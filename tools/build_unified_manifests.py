#!/usr/bin/env python3
"""Build the unified POC candidate pool from raw/ (v2).

Sources (raw/POC_train/):
    tts_pathology/metadata_v2.csv                VoxCPM pathology (8 voices)
    tts_pathology/supp_{pathref,slref}.manifest   fused specimen-no/measure/slab templates
    tts_radiology/*.manifest.json                RAD/RADP/RADX/ep1 CosyVoice
    llm_text_syn/metadata_voxcpm2var.csv          VoxCPM free sentences (2 voices)
    real_target_domain/*.jsonl                   real raw/denoised + 0821 raw/denoised
    youtube_med_en/{pathology,radiology,hk_en_yue}.jsonl
    wenetspeech_yue/*.jsonl                      real Cantonese (5 overlapping subsets)

Writes:
    data/poc_train_unified_v2/all.jsonl          train-eligible pool (unfiltered)
    data/poc_train_unified_v2/{dev,test}.jsonl   POC53 rebuilt from raw (new labels)
    data/poc_train_unified_v2/manifest_stats.json
    data/poc_train_unified_v2/label_sync.json    old-manifest vs raw label diff

All labels are "language None<asr_text>" + NFKC-collapsed text. raw/POC_train is the
only source of truth; no old data/ manifests are read except for the label diff report.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))
from prepare_real_raw_denoised_sft import load_rows  # noqa: E402
from evaluation.english_medical.text_normalization import normalize_english  # noqa: E402

PREFIX = "language None<asr_text>"
COSY_ROOT = Path("/root/autodl-tmp/workspace/project/CosyVoice/examples/switchlingua-yue")
SYNTH_SOURCES = {
    "tts_pathology_vox",
    "tts_pathology_supp",
    "tts_radiology",
    "llm_text_syn",
}


def label_body(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or "").strip())
    value = " ".join(value.split())
    if not value:
        raise ValueError("empty transcript")
    return value


def body(text: str) -> str:
    return (text or "").split("<asr_text>", 1)[-1].strip()


def make_row(audio: Path, text: str, source: str, group: str, segment: str,
             duration: float, noise_aug: int, voice: str = "", **extra) -> dict:
    row = {
        "audio": str(audio),
        "text": f"{PREFIX}{label_body(text)}",
        "aug": 1,
        "noise_aug": int(noise_aug),
        "sampling_source": source,
        "source_group": group,
        "segment_id": segment,
        "duration_sec": round(float(duration), 6),
    }
    if voice:
        row["voice"] = voice
    row.update({k: v for k, v in extra.items() if v not in (None, "", [])})
    return row


def load_vox_csv(path: Path, source: str) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for item in csv.DictReader(handle):
            audio = Path(item["audio_path"]).resolve()
            if not audio.is_file():
                raise FileNotFoundError(audio)
            voice = str(item.get("source") or "")
            rows.append(make_row(
                audio, item["text"], source, source,
                f"{source}/{voice}_{audio.parent.name}_{audio.stem}",
                float(item["duration"]), 1, voice=voice,
            ))
    return rows


def load_cosy(manifest: Path, source: str, variant: str) -> list[dict]:
    rows = []
    for item in json.loads(manifest.read_text(encoding="utf-8")):
        audio = (COSY_ROOT / item["synth_wav"]).resolve()
        if not audio.is_file():
            raise FileNotFoundError(f"{manifest.name}: missing {audio}")
        rows.append(make_row(
            audio, item["text"], source, source,
            f"{source}/{variant}_{item['utt']}",
            float(item["synth_sec"]), 1, voice=variant,
        ))
    return rows


def load_youtube(path: Path, subset: str) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            audio = Path(item["audio"])
            if not audio.is_file():
                raise FileNotFoundError(audio)
            video = str(item.get("video_id") or audio.parent.name)
            rows.append(make_row(
                audio, item["text"], f"youtube_{subset}", video,
                f"youtube_{subset}/{video}/{item.get('segment_id') or audio.stem}",
                float(item["duration_sec"]), 0,
                voice=str(item.get("source") or ""),
                language=item.get("language"), quality=item.get("quality_bucket"),
                sampling_rate=item.get("sampling_rate"),
            ))
    return rows


def load_wenetspeech(root: Path) -> list[dict]:
    """Merge the overlapping subset exports; one row per utt, union of flags."""
    merged: dict[str, dict] = {}
    for path in sorted(root.glob("*.jsonl")):
        audio_dir = path.resolve().parent
        subset = path.stem
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                utt = str(item["utt"])
                audio = (audio_dir / item["audio"]).resolve()
                if not audio.is_file():
                    raise FileNotFoundError(f"{path.name}: missing {audio}")
                entry = merged.get(utt)
                if entry is None:
                    parts = utt.split("_")
                    merged[utt] = {
                        "audio": audio,
                        "text": item["text"],
                        "duration": float(item["dur_ms"]) / 1000.0,
                        "video": parts[0],
                        "session": "_".join(parts[:2]),
                        "flags": set(item.get("flags") or []),
                        "subsets": {subset},
                        "n_num": int(item.get("n_num") or 0),
                        "n_marker": int(item.get("n_marker") or 0),
                    }
                else:
                    if entry["text"] != item["text"]:
                        raise ValueError(f"wenetspeech text mismatch for {utt}")
                    entry["flags"] |= set(item.get("flags") or [])
                    entry["subsets"].add(subset)
                    entry["n_num"] = max(entry["n_num"], int(item.get("n_num") or 0))
                    entry["n_marker"] = max(entry["n_marker"], int(item.get("n_marker") or 0))
    rows = []
    for utt, item in sorted(merged.items()):
        rows.append(make_row(
            item["audio"], item["text"], "wenetspeech_yue", item["session"],
            f"wenetspeech_yue/{utt}", item["duration"], 0,
            voice=item["session"], flags=sorted(item["flags"]),
            subsets=sorted(item["subsets"]), n_num=item["n_num"], n_marker=item["n_marker"],
        ))
    return rows


def validate_pairs(named_pairs: list[tuple[str, list[dict], list[dict]]]) -> dict:
    report = {}
    for name, left, right in named_pairs:
        left_map = {row["segment_id"]: body(row["text"]) for row in left}
        right_map = {row["segment_id"]: body(row["text"]) for row in right}
        if set(left_map) != set(right_map):
            raise ValueError(f"{name}: segment sets differ")
        mismatch = [key for key in left_map if left_map[key] != right_map[key]]
        if mismatch:
            raise ValueError(f"{name}: transcript mismatch for {mismatch[:5]}")
        report[name] = {"rows": len(left_map), "text_mismatch": 0}
    return report


def leakage(train_rows: list[dict], dev_rows: list[dict], n: int = 8) -> dict:
    def grams(text: str) -> set[str]:
        tokens = text.split()
        return {" ".join(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1))}

    dev_text = {body(row["text"]) for row in dev_rows}
    dev_grams = set().union(*(grams(text) for text in dev_text))
    audio_overlap = sum(1 for row in train_rows if row["audio"] in {r["audio"] for r in dev_rows})
    exact = Counter()
    gram8 = Counter()
    for row in train_rows:
        text = body(row["text"])
        if text in dev_text:
            exact[row["sampling_source"]] += 1
        if grams(text) & dev_grams:
            gram8[row["sampling_source"]] += 1
    return {
        "dev_audio_overlap": audio_overlap,
        "exact_text_overlap": dict(sorted(exact.items())),
        "eight_gram_overlap": dict(sorted(gram8.items())),
    }


def stats(rows: list[dict]) -> dict:
    by_source = defaultdict(lambda: {"rows": 0, "hours": 0.0, "unique_texts": set(), "voices": set()})
    durations = defaultdict(list)
    for row in rows:
        key = row["sampling_source"]
        item = by_source[key]
        item["rows"] += 1
        item["hours"] += float(row["duration_sec"]) / 3600.0
        item["unique_texts"].add(normalize_english(body(row["text"])))
        if row.get("voice"):
            item["voices"].add(str(row["voice"]))
        durations[key].append(float(row["duration_sec"]))
    detail = {}
    synth_h = real_h = 0.0
    for key, item in sorted(by_source.items()):
        dur = sorted(durations[key])
        hours = round(item["hours"], 3)
        detail[key] = {
            "rows": item["rows"],
            "hours": hours,
            "unique_texts": len(item["unique_texts"]),
            "rows_per_text": round(item["rows"] / max(len(item["unique_texts"]), 1), 2),
            "voices": len(item["voices"]),
            "duration_p50": round(statistics.median(dur), 2),
            "duration_max": round(dur[-1], 2),
        }
        if key in SYNTH_SOURCES:
            synth_h += hours
        else:
            real_h += hours
    total = synth_h + real_h
    return {
        "rows": len(rows),
        "hours": round(total, 3),
        "synth_hours": round(synth_h, 3),
        "real_hours": round(real_h, 3),
        "synth_share_hours": round(synth_h / total, 4) if total else 0.0,
        "by_source": detail,
    }


def label_diff(old_dev: Path, new_dev: list[dict]) -> dict:
    if not old_dev.is_file():
        return {"available": False}
    old = [json.loads(line) for line in old_dev.read_text(encoding="utf-8").splitlines() if line.strip()]
    old_map = {row["audio"]: body(row["text"]) for row in old}
    new_map = {row["audio"]: body(row["text"]) for row in new_dev}
    changed = [{"audio": audio, "old": old_map[audio], "new": new_map[audio]}
               for audio in sorted(set(old_map) & set(new_map)) if old_map[audio] != new_map[audio]]
    return {
        "available": True,
        "old_rows": len(old),
        "new_rows": len(new_map),
        "changed": len(changed),
        "examples": changed[:10],
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default="data/poc_train_unified_v2")
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--check-audio", type=int, default=1)
    ap.add_argument("--old-dev", default="data/poc_train_unified7/direct_mix/dev.jsonl")
    args = ap.parse_args()
    out_root = Path(args.out_root).resolve()
    raw = REPO / "raw/POC_train"
    check = bool(args.check_audio)

    print("[load] synth")
    synth = []
    synth += load_vox_csv(raw / "tts_pathology/metadata_v2.csv", "tts_pathology_vox")
    synth += load_cosy(raw / "tts_pathology/supp_pathref.manifest.json", "tts_pathology_supp", "supp_pathref")
    synth += load_cosy(raw / "tts_pathology/supp_slref.manifest.json", "tts_pathology_supp", "supp_slref")
    for name, variant in (("RAD_slref", "rad_slref"), ("RADP_pathref", "radp_pathref"),
                          ("RADX_slref", "radx_slref"), ("ep1", "ep1")):
        synth += load_cosy(raw / f"tts_radiology/{name}.manifest.json", "tts_radiology", variant)
    synth += load_vox_csv(raw / "llm_text_syn/metadata_voxcpm2var.csv", "llm_text_syn")

    print("[load] real target")
    rt = raw / "real_target_domain"
    real_main_raw = load_rows(rt / "metadata_raw.jsonl", sampling_source="real_raw", noise_aug=0, check_audio=check)
    real_main_den = load_rows(rt / "metadata_denoised.jsonl", sampling_source="real_denoised", noise_aug=1, check_audio=check)
    real_0821_raw = load_rows(rt / "metadata_0821_raw.jsonl", sampling_source="real_0821_raw", noise_aug=0, check_audio=check)
    real_0821_den = load_rows(rt / "metadata_0821.jsonl", sampling_source="real_0821_denoised", noise_aug=1, check_audio=check)
    pairs = validate_pairs([("real_main", real_main_raw, real_main_den),
                            ("real_0821", real_0821_raw, real_0821_den)])
    for row in real_main_raw + real_main_den + real_0821_raw + real_0821_den:
        row["segment_id"] = f"{row['sampling_source']}/{row['segment_id']}"

    print("[load] youtube")
    youtube = []
    for subset in ("pathology", "radiology", "hk_en_yue"):
        rows = load_youtube(raw / f"youtube_med_en/{subset}.jsonl", subset)
        print(f"  {subset:10s} rows={len(rows)}")
        youtube += rows

    print("[load] wenetspeech_yue")
    yue = load_wenetspeech(raw / "wenetspeech_yue")
    print(f"  unique utts={len(yue)}")

    pool = synth + real_main_raw + real_main_den + real_0821_raw + real_0821_den + youtube + yue
    audio = [row["audio"] for row in pool]
    if len(audio) != len(set(audio)):
        dup = [a for a, c in Counter(audio).items() if c > 1][:3]
        raise RuntimeError(f"duplicate audio across sources: {dup}")

    print("[load] POC53 dev/test")
    dev = load_rows(REPO / "raw/POC_test/pathology.jsonl", sampling_source="poc_test",
                    noise_aug=0, check_audio=check)
    dev = [dict(row, aug=0, noise_aug=0) for row in dev]
    if len(dev) != 53:
        raise ValueError(f"expected 53 POC clips, got {len(dev)}")

    pool = list(pool)
    random.Random(args.seed).shuffle(pool)
    write_jsonl(out_root / "all.jsonl", pool)
    write_jsonl(out_root / "dev.jsonl", dev)
    write_jsonl(out_root / "test.jsonl", dev)

    report = {
        "seed": args.seed,
        "raw_root": str(raw),
        "pair_validation": pairs,
        "pool": stats(pool),
        "dev_test": stats(dev),
        "leakage_vs_poc53": leakage(pool, dev),
        "label_sync": label_diff(Path(args.old_dev), dev),
    }
    (out_root / "manifest_stats.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_root / "label_sync.json").write_text(
        json.dumps(report["label_sync"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[write] pool rows={len(pool)} hours={report['pool']['hours']:.2f}")
    for key, item in report["pool"]["by_source"].items():
        print(f"  {key:24s} rows={item['rows']:7d} hours={item['hours']:8.2f} "
              f"uniq={item['unique_texts']:6d} voices={item['voices']}")
    print(f"[leak] {report['leakage_vs_poc53']}")
    print(f"[label] {report['label_sync'].get('changed')} POC53 label changes")
    print(f"[done] {out_root}")


if __name__ == "__main__":
    main()
