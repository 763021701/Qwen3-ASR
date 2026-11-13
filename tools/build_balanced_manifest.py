#!/usr/bin/env python3
"""Build a text/acoustically distribution-balanced POC SFT manifest.

Input is data/poc_train_unified_v2/all.jsonl (unfiltered pool built from raw/).

Text policy
  - POC53 leakage: drop any row sharing an 8-gram (or exact normalized text) with dev.
  - Synth voice cap: <=3 voices per normalized text, <=5 for rare phenomena.
  - VoxCPM generic rows: keep every phenomenon-bearing row, sample the rest to
    --vox-generic-hours (default 26 h) so generic templates do not dominate.
  - supp CosyVoice: repeat by phenomenon tier (rare 4x, medium 2x, common 1x);
    this is the only source covering paraffin/SN/coronal-full-slab templates.
  - tts_radiology: uniform sample to --radiology-hours.
  - llm_text_syn: keep all.
  - YouTube: drop reference loops (5-gram repeated >=3x) and very short/long texts;
    sample with per-video caps to increase video/speaker diversity.
  - WenetSpeech-Yue: weighted toward number-dense/medical clips, 10 min/session cap.
  - real_target_domain: keep all (raw+denoised pair), domain-priority data.

Acoustic policy
  - duration 1.0-30.0 s.
  - probe first 3 s from the middle third: reject silence, clipping >2%, active
    frame ratio <10%, sample rate <8 kHz.

Usage:
    python tools/build_balanced_manifest.py
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as futures
import json
import math
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from evaluation.english_medical.text_normalization import normalize_english  # noqa: E402

PREFIX = "language None<asr_text>"
SYNTH = {"tts_pathology_vox", "tts_pathology_supp", "tts_radiology", "llm_text_syn"}
PATTERNS = {
    "specimen_id": re.compile(r"\b\d{4,}\b"),
    "sn": re.compile(r"\bsn\s*\d|\bsn\b"),
    "block_letter": re.compile(r"\bblock\s*[a-z]\b|\bblocks?\s+[a-z]\s+to\s+[a-z]\b"),
    "frozen": re.compile(r"\bfrozen\b"),
    "paraffin": re.compile(r"\bparaffin\b"),
    "coronal": re.compile(r"\bcoronal\b|full slab"),
    "margins": re.compile(r"\bmargin|perpendicular"),
    "measurement": re.compile(r"\d+(?:\.\d+)?\s*(?:cm|mm|gram|g)\b"),
    "specimen_type": re.compile(
        r"hemithyroidectomy|lumpectomy|hysterectomy|sentinel node|mastectomy|nephrectomy"),
    "orientation": re.compile(r"orientation|stitch"),
}
RARE = {"paraffin", "sn", "coronal"}
MEDIUM = {"frozen", "specimen_type", "margins", "orientation"}
COMMON = {"block_letter", "measurement", "specimen_id"}


def body(text: str) -> str:
    return (text or "").split("<asr_text>", 1)[-1].strip()


def tokens(text: str) -> list[str]:
    return normalize_english(text).split()


def ngrams(items: list[str], n: int) -> list[str]:
    return [" ".join(items[i:i + n]) for i in range(max(0, len(items) - n + 1))]


def phenomena(norm: str) -> set[str]:
    return {name for name, pattern in PATTERNS.items() if pattern.search(norm)}


def tier_of(phen: set[str]) -> str:
    if phen & RARE:
        return "rare"
    if phen & MEDIUM:
        return "medium"
    if phen & COMMON:
        return "common"
    return "none"


def has_loop(toks: list[str], n: int = 5, repeat: int = 3) -> bool:
    return any(count >= repeat for count in collections.Counter(ngrams(toks, n)).values())


def has_loop_cjk(norm: str, n: int = 10, repeat: int = 3) -> bool:
    chars = norm.replace(" ", "")
    grams = [chars[i:i + n] for i in range(max(0, len(chars) - n + 1))]
    return any(count >= repeat for count in collections.Counter(grams).values())


def text_units(norm: str) -> tuple[int, bool]:
    """Return (length, is_cjk). CJK transcripts are not word-segmented."""
    cjk = sum(1 for char in norm if "\u4e00" <= char <= "\u9fff")
    if cjk >= 3:
        return len(norm.replace(" ", "")), True
    return len(norm.split()), False


def hours(items: list[dict]) -> float:
    return sum(float(item["row"]["duration_sec"]) for item in items) / 3600.0


def weighted_order(items: list[dict], seed: int, weight) -> list[dict]:
    """Efraimidis-Spirakis weighted sampling order (larger key first)."""
    rng = random.Random(seed)
    ordered = list(items)
    for item in ordered:
        item["_key"] = rng.random() ** (1.0 / max(float(weight(item)), 1e-6))
    ordered.sort(key=lambda item: item["_key"], reverse=True)
    return ordered


def sample_capped(items: list[dict], target_hours: float, seed: int,
                  cap_group_hours: float, weight=None) -> list[dict]:
    ordered = weighted_order(items, seed, weight or (lambda _item: 1.0))
    picked, group_hours, total = [], collections.defaultdict(float), 0.0
    for item in ordered:
        if total >= target_hours:
            break
        duration = float(item["row"]["duration_sec"]) / 3600.0
        group = item["row"].get("source_group", "")
        if group_hours[group] + duration > cap_group_hours:
            continue
        picked.append(item)
        group_hours[group] += duration
        total += duration
    return picked


def cap_synth_voices(items: list[dict]) -> tuple[list[dict], dict]:
    groups = collections.defaultdict(list)
    for item in items:
        groups[(item["row"]["sampling_source"], item["norm"])].append(item)
    voice_hours = collections.defaultdict(float)
    kept, dropped = [], 0
    for key in sorted(groups):
        group = groups[key]
        cap = 5 if group[0]["tier"] == "rare" else 3
        by_voice = collections.defaultdict(list)
        for item in group:
            by_voice[item["row"].get("voice") or item["row"]["sampling_source"]].append(item)
        voices = sorted(by_voice, key=lambda voice: (voice_hours[(key[0], voice)], voice))
        for voice in voices[:cap]:
            chosen = sorted(by_voice[voice], key=lambda item: item["row"]["audio"])[0]
            kept.append(chosen)
            voice_hours[(key[0], voice)] += float(chosen["row"]["duration_sec"]) / 3600.0
        dropped += sum(len(rows) for voice, rows in by_voice.items() if voice not in voices[:cap])
    return kept, {"dropped": dropped}


def probe_one(path: str) -> dict:
    try:
        info = sf.info(path)
        if info.samplerate <= 0 or info.frames <= 0:
            return {"error": "empty"}
        window = int(min(3.0 * info.samplerate, info.frames))
        with sf.SoundFile(path) as handle:
            handle.seek(max(0, info.frames // 3))
            data = handle.read(frames=window, dtype="float32", always_2d=True)
        if data.size == 0:
            return {"error": "empty"}
        mono = data.mean(axis=1)
        peak = float(np.max(np.abs(mono)))
        rms = float(np.sqrt(np.mean(mono ** 2)))
        clip_ratio = float(np.mean(np.abs(mono) >= 0.999))
        frame = max(1, int(0.025 * info.samplerate))
        trimmed = mono[: len(mono) - len(mono) % frame]
        frames = trimmed.reshape(-1, frame) if trimmed.size else np.zeros((0, frame), dtype=np.float32)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1)) if frames.size else np.zeros(1)
        reference = float(np.percentile(frame_rms, 95))
        active = float(np.mean(frame_rms > max(1e-5, reference / 20.0)))
        return {
            "sr": int(info.samplerate),
            "peak": round(peak, 6),
            "rms_db": round(20.0 * math.log10(max(rms, 1e-9)), 2),
            "clip_ratio": round(clip_ratio, 6),
            "active_ratio": round(active, 4),
        }
    except Exception as exc:  # noqa: BLE001 - report per-file decode failures
        return {"error": str(exc)}


def reject_reason(stats: dict) -> str:
    if "error" in stats:
        return "decode_error"
    if stats["sr"] < 8000:
        return "low_sr"
    if stats["peak"] < 0.005 or stats["rms_db"] < -55.0:
        return "silence"
    if stats["clip_ratio"] > 0.02:
        return "clipping"
    if stats["active_ratio"] < 0.10:
        return "low_active"
    return ""


def probe_selected(items: list[dict], workers: int, cache_path: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if cache_path.is_file():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                cache[row["audio"]] = row
    paths = sorted({item["row"]["audio"] for item in items} - set(cache))
    print(f"[probe] unique={len({item['row']['audio'] for item in items})} cached={len(cache)} new={len(paths)}")
    if paths:
        with futures.ProcessPoolExecutor(max_workers=workers) as pool:
            for path, stats in zip(paths, pool.map(probe_one, paths, chunksize=64)):
                cache[path] = {"audio": path, **stats}
        with cache_path.open("w", encoding="utf-8") as handle:
            for path in sorted(cache):
                handle.write(json.dumps(cache[path], ensure_ascii=False) + "\n")
    return cache


def entropy(counts: collections.Counter) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum((value / total) * math.log2(value / total) for value in counts.values() if value)


def text_report(items: list[dict], dev_norms: list[str]) -> dict:
    dev_grams = set().union(*(set(ngrams(text.split(), 8)) for text in dev_norms))
    dev_set = set(dev_norms)
    by_source = collections.defaultdict(list)
    for item in items:
        by_source[item["row"]["sampling_source"]].append(item)
    report = {"poc53_phenomena": dict(collections.Counter(
        phen for text in dev_norms for phen in phenomena(text)))}
    for source, group in sorted(by_source.items()):
        unique, phen_rows, phen_unique = {}, collections.Counter(), collections.Counter()
        five, uni, tri = collections.Counter(), collections.Counter(), collections.Counter()
        covered_grams = set()
        exact = 0
        for item in group:
            unique.setdefault(item["norm"], item)
            for name in item["phen"]:
                phen_rows[name] += 1
            toks = item["norm"].split()
            uni.update(toks)
            tri.update(ngrams(toks, 3))
            five.update(ngrams(toks, 5))
            if set(ngrams(toks, 8)) & dev_grams:
                covered_grams |= set(ngrams(toks, 8)) & dev_grams
            if item["norm"] in dev_set:
                exact += 1
        for norm in unique:
            for name in phenomena(norm):
                phen_unique[name] += 1
        top5 = sum(count for _, count in five.most_common(5)) / max(sum(five.values()), 1)
        report[source] = {
            "rows": len(group),
            "hours": round(hours(group), 3),
            "unique_texts": len(unique),
            "rows_per_text": round(len(group) / max(len(unique), 1), 2),
            "unigram_entropy": round(entropy(uni), 2),
            "trigram_entropy": round(entropy(tri), 2),
            "top5_5gram_token_share": round(top5, 3),
            "poc53_8gram_hits": len(covered_grams),
            "poc53_exact_texts": exact,
            "phenomena_rows": dict(sorted(phen_rows.items())),
            "phenomena_unique": dict(sorted(phen_unique.items())),
        }

    total_rows = sum(item["rows"] for key, item in report.items() if key != "poc53_phenomena")
    all_phen = collections.Counter()
    for key, item in report.items():
        if key != "poc53_phenomena":
            all_phen.update(item["phenomena_rows"])
    report["total"] = {
        "rows": total_rows,
        "phenomena_rows": dict(sorted(all_phen.items())),
        "phenomena_row_share": {name: round(count / max(total_rows, 1), 4)
                                for name, count in sorted(all_phen.items())},
        "poc53_clip_share": {name: round(count / max(len(dev_norms), 1), 4)
                             for name, count in report["poc53_phenomena"].items()},
    }
    for group_name, sources in {
        "pathology_synth": ["tts_pathology_vox", "tts_pathology_supp", "llm_text_syn"],
        "real_domain": ["real_raw", "real_denoised", "real_0821_raw", "real_0821_denoised"],
    }.items():
        agg_rows = sum(report[source]["rows"] for source in sources if source in report)
        agg_hours = sum(report[source]["hours"] for source in sources if source in report)
        agg_phen = collections.Counter()
        for source in sources:
            if source in report:
                agg_phen.update(report[source]["phenomena_rows"])
        report[group_name] = {
            "rows": agg_rows,
            "hours": round(agg_hours, 3),
            "phenomena_row_share": {name: round(count / max(agg_rows, 1), 4)
                                    for name, count in sorted(agg_phen.items())},
        }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default="data/poc_train_unified_v2/all.jsonl")
    ap.add_argument("--dev", default="data/poc_train_unified_v2/dev.jsonl")
    ap.add_argument("--out-dir", default="data/poc_train_unified_v2/dist_balanced")
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--synth-share", type=float, default=0.5)
    ap.add_argument("--vox-generic-hours", type=float, default=26.0)
    ap.add_argument("--radiology-hours", type=float, default=50.0)
    ap.add_argument("--supp-repeats", default="rare:4,medium:2,common:1,none:1")
    ap.add_argument("--wenetspeech-share", type=float, default=0.14)
    ap.add_argument("--hk-hours", type=float, default=18.0)
    ap.add_argument("--youtube-cap-hours", default="pathology:0.15,radiology:0.2,hk_en_yue:0.5")
    ap.add_argument("--wenetspeech-session-cap-hours", type=float, default=0.167)
    ap.add_argument("--min-duration", type=float, default=1.0)
    ap.add_argument("--max-duration", type=float, default=30.0)
    ap.add_argument("--probe-workers", type=int, default=min(32, os.cpu_count() or 8))
    args = ap.parse_args()

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    dev_rows = [json.loads(line) for line in Path(args.dev).read_text(encoding="utf-8").splitlines() if line.strip()]
    dev_norms = [normalize_english(body(row["text"])) for row in dev_rows]

    print("[load] pool")
    pool = [json.loads(line) for line in Path(args.pool).read_text(encoding="utf-8").splitlines() if line.strip()]
    dev_grams = set().union(*(set(ngrams(text.split(), 8)) for text in dev_norms))
    dev_set = set(dev_norms)
    filters = collections.Counter()
    items, synth = [], []
    for row in pool:
        source = row["sampling_source"]
        duration = float(row["duration_sec"])
        if not (args.min_duration <= duration <= args.max_duration):
            filters[f"{source}:duration"] += 1
            continue
        norm = normalize_english(body(row["text"]))
        if not norm:
            filters[f"{source}:empty_text"] += 1
            continue
        units, is_cjk = text_units(norm)
        min_units = 1 if source.startswith("real_") else (5 if is_cjk else 3)
        if units < min_units:
            filters[f"{source}:short_text"] += 1
            continue
        if norm in dev_set or set(ngrams(norm.split(), 8)) & dev_grams:
            filters[f"{source}:poc53_leak"] += 1
            continue
        toks = norm.split()
        if source.startswith("youtube") and (
                (has_loop_cjk(norm) if is_cjk else has_loop(toks)) or units > 200):
            filters[f"{source}:text_loop_or_long"] += 1
            continue
        if source == "wenetspeech_yue" and (has_loop_cjk(norm) if is_cjk else has_loop(toks)):
            filters[f"{source}:text_loop"] += 1
            continue
        item = {"row": row, "norm": norm, "phen": phenomena(norm)}
        item["tier"] = tier_of(item["phen"])
        items.append(item)
        if source in SYNTH:
            synth.append(item)
    print(f"[filter] kept={len(items)} dropped={dict(filters)}")

    synth, voice_report = cap_synth_voices(synth)
    by_source = collections.defaultdict(list)
    for item in synth:
        by_source[item["row"]["sampling_source"]].append(item)
    real_by_source = collections.defaultdict(list)
    for item in items:
        if item["row"]["sampling_source"] not in SYNTH:
            real_by_source[item["row"]["sampling_source"]].append(item)

    selected: list[dict] = []
    plan = {}

    vox = by_source["tts_pathology_vox"]
    vox_phen = [item for item in vox if item["tier"] != "none"]
    vox_generic = [item for item in vox if item["tier"] == "none"]
    vox_keep = vox_phen + sample_capped(
        vox_generic, args.vox_generic_hours, args.seed + 1, cap_group_hours=1e9)
    selected += vox_keep
    plan["tts_pathology_vox"] = {
        "phenomena_rows": len(vox_phen), "phenomena_hours": round(hours(vox_phen), 3),
        "generic_rows": len(vox_keep) - len(vox_phen),
        "generic_hours": round(hours(vox_keep) - hours(vox_phen), 3),
    }

    radiology = sample_capped(by_source["tts_radiology"], args.radiology_hours,
                              args.seed + 2, cap_group_hours=1e9)
    selected += radiology
    plan["tts_radiology"] = {"rows": len(radiology), "hours": round(hours(radiology), 3)}

    llm = by_source["llm_text_syn"]
    llm_repeats = []
    for item in llm:
        reps = 2 if item["tier"] == "rare" else 1
        for repeat in range(reps):
            llm_repeats.append({"row": item["row"], "repeat": repeat,
                                **{k: item[k] for k in ("norm", "phen", "tier")}})
    selected += llm_repeats
    plan["llm_text_syn"] = {"rows": len(llm_repeats), "hours": round(hours(llm_repeats), 3)}

    repeats = {key: int(value) for key, value in
               (pair.split(":") for pair in args.supp_repeats.split(",") if pair)}
    supp = []
    for item in by_source["tts_pathology_supp"]:
        for repeat in range(repeats.get(item["tier"], 1)):
            supp.append({"row": item["row"], "repeat": repeat, **{k: item[k] for k in ("norm", "phen", "tier")}})
    selected += supp
    plan["tts_pathology_supp"] = {
        "rows": len(supp), "hours": round(hours(supp), 3),
        "repeats": repeats,
        "rows_by_tier": dict(collections.Counter(item["tier"] for item in supp)),
    }

    synth_hours = hours(selected)
    real_hours = synth_hours * args.synth_share / (1.0 - args.synth_share)
    print(f"[plan] synth={synth_hours:.2f}h real_target={real_hours:.2f}h")

    real_target = [item for source in ("real_raw", "real_denoised", "real_0821_raw", "real_0821_denoised")
                   for item in real_by_source[source]]
    selected += real_target
    plan["real_target"] = {"rows": len(real_target), "hours": round(hours(real_target), 3)}

    wenet_target = real_hours * args.wenetspeech_share
    wenet = sample_capped(
        real_by_source["wenetspeech_yue"], wenet_target, args.seed + 3,
        cap_group_hours=args.wenetspeech_session_cap_hours,
        weight=lambda item: 1.0 + 0.15 * min(int(item["row"].get("n_num") or 0), 10)
        + 0.5 * ("medical" in (item["row"].get("flags") or [])),
    )
    selected += wenet
    plan["wenetspeech_yue"] = {"rows": len(wenet), "hours": round(hours(wenet), 3)}

    caps = {key: float(value) for key, value in
            (pair.split(":") for pair in args.youtube_cap_hours.split(",") if pair)}
    hk_target = min(args.hk_hours, hours(real_by_source["youtube_hk_en_yue"]))
    hk = sample_capped(real_by_source["youtube_hk_en_yue"], hk_target, args.seed + 4,
                       cap_group_hours=caps["hk_en_yue"])
    selected += hk
    remaining = max(0.0, real_hours - hours(real_target) - hours(wenet) - hours(hk))
    path_pool = hours(real_by_source["youtube_pathology"])
    rad_pool = hours(real_by_source["youtube_radiology"])
    path_target = remaining * path_pool / max(path_pool + rad_pool, 1e-9)
    rad_target = remaining - path_target
    youtube_path = sample_capped(real_by_source["youtube_pathology"], path_target, args.seed + 5,
                                 cap_group_hours=caps["pathology"])
    youtube_rad = sample_capped(real_by_source["youtube_radiology"], rad_target, args.seed + 6,
                                cap_group_hours=caps["radiology"])
    selected += youtube_path + youtube_rad
    plan["youtube"] = {
        "hk_hours": round(hours(hk), 3),
        "pathology_hours": round(hours(youtube_path), 3),
        "radiology_hours": round(hours(youtube_rad), 3),
        "video_caps": caps,
    }

    print("[probe] acoustic quality")
    cache = probe_selected(selected, args.probe_workers, out / "acoustic_stats.jsonl")
    kept, rejected = [], collections.Counter()
    for item in selected:
        reason = reject_reason(cache[item["row"]["audio"]])
        if reason:
            rejected[f"{item['row']['sampling_source']}:{reason}"] += 1
        else:
            kept.append(item)
    print(f"[probe] kept={len(kept)} rejected={dict(rejected)}")

    rows = []
    final = []
    for item in kept:
        row = dict(item["row"])
        if "repeat" in item:
            row["segment_id"] = f"{row['segment_id']}_r{item['repeat']}"
            row["repeat"] = item["repeat"]
        rows.append(row)
        final.append({**item, "row": row})
    random.Random(args.seed + 7).shuffle(final)
    rows = [item["row"] for item in final]

    with (out / "train.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for split in ("dev", "test"):
        (out / f"{split}.jsonl").write_bytes(Path(args.dev).with_name(f"{split}.jsonl").read_bytes())

    acoustic_summary = {}
    for source in sorted({item["row"]["sampling_source"] for item in final}):
        stats = [cache[item["row"]["audio"]] for item in final
                 if item["row"]["sampling_source"] == source]
        rms = sorted(stat["rms_db"] for stat in stats if "rms_db" in stat)
        active = sorted(stat["active_ratio"] for stat in stats if "active_ratio" in stat)
        acoustic_summary[source] = {
            "rows": len(stats),
            "rms_db_p10_p50": [round(rms[len(rms) // 10], 1), round(rms[len(rms) // 2], 1)] if rms else [],
            "active_ratio_p10_p50": [round(active[len(active) // 10], 3),
                                     round(active[len(active) // 2], 3)] if active else [],
        }
    by_source = collections.defaultdict(lambda: {"rows": 0, "hours": 0.0})
    for item in final:
        entry = by_source[item["row"]["sampling_source"]]
        entry["rows"] += 1
        entry["hours"] += float(item["row"]["duration_sec"]) / 3600.0
    for entry in by_source.values():
        entry["hours"] = round(entry["hours"], 3)

    report = {
        "seed": args.seed,
        "policy": {
            "synth_share": args.synth_share,
            "vox_generic_hours": args.vox_generic_hours,
            "radiology_hours": args.radiology_hours,
            "supp_repeats": repeats,
            "wenetspeech_share_of_real": args.wenetspeech_share,
            "hk_hours": args.hk_hours,
            "youtube_caps": caps,
            "wenetspeech_session_cap_hours": args.wenetspeech_session_cap_hours,
            "duration_range": [args.min_duration, args.max_duration],
        },
        "plan": plan,
        "filters": dict(filters),
        "voice_cap": voice_report,
        "acoustic_rejects": dict(rejected),
        "acoustic_kept": acoustic_summary,
        "manifest": {
            "rows": len(final),
            "hours": round(hours(final), 3),
            "synth_hours": round(hours([item for item in final
                                        if item["row"]["sampling_source"] in SYNTH]), 3),
            "real_hours": round(hours([item for item in final
                                       if item["row"]["sampling_source"] not in SYNTH]), 3),
            "by_source": dict(sorted(by_source.items())),
        },
        "text": text_report(final, dev_norms),
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")

    print(f"[write] {out / 'train.jsonl'} rows={len(rows)} hours={report['manifest']['hours']:.2f} "
          f"synth={report['manifest']['synth_hours']:.2f} real={report['manifest']['real_hours']:.2f}")


if __name__ == "__main__":
    main()
