#!/usr/bin/env python3
"""Build and audit source-disjoint noise/RIR manifests without moving audio.

The six JSONL files are the runtime allowlists. inventory.jsonl preserves every
input version, including aliases. Fingerprints are cached by file signature.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import sqlite3
import sys
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SPLITS = ("train", "dev", "test")
RATIOS = (0.8, 0.1, 0.1)
PRIORITY = {name: index for index, name in enumerate(SPLITS)}
FINGERPRINT_VERSION = "pcm16-mono16k-window4s-hop1s-2k-int8-v1"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def walk_audio(root: Path):
    seen = set()
    for directory, dirs, files in os.walk(root, followlinks=True):
        resolved = Path(directory).resolve()
        if resolved in seen:
            dirs[:] = []
            continue
        seen.add(resolved)
        dirs.sort()
        for name in sorted(files):
            if Path(name).suffix.lower() in {".wav", ".flac"}:
                yield Path(directory) / name


def real_room(stem: str) -> tuple[str, bool]:
    """Conservatively merge microphone/angle/distance aliases of real rooms."""
    if stem.startswith("RVB2014_"):
        match = re.search(r"_rir_(.+?)_(?:far|near)_", stem)
        if match:
            return "slr28:reverb2014:" + match[1], False
    if stem.startswith("RWCP_"):
        match = re.search(r"_(?:circle|cirline)_([^_]+)", stem)
        if match:
            room = re.sub(r"^(e[12])[abc]$", r"\1", match[1])
            room = re.sub(r"^jr[12]$", "jr", room)
            return "slr28:rwcp:" + room, False
        if stem.startswith("RWCP_type4_"):
            # The list treats individual directions/ears as rooms. No physical
            # room identity is available, so keep this entire family in train.
            return "slr28:rwcp:unidentified_type4", True
    if stem.startswith("air_"):
        match = re.search(r"_air_(?:binaural|phone)_(.+)", stem)
        if match:
            label = match[1]
            if label.startswith("bt_"):
                label = label[3:]
            label = re.sub(r"(?:_\d.*|_hfrp$|_hhp$)", "", label)
            label = re.sub(r"\d+$", "", label)
            # Be conservative across the phone/binaural naming systems.
            if label == "aula_carolina":
                label = "lecture"
            return "slr28:air:" + label, False
    raise ValueError(f"Unrecognized real RIR room: {stem}")


def inventory(noise_root: Path, rir_root: Path, esc_meta: Path) -> list[dict]:
    esc = {r["filename"]: r for r in csv.DictReader(esc_meta.open())}
    artists = {}
    for annotations in (noise_root / "musan" / "music").rglob("ANNOTATIONS"):
        for line in annotations.read_text().splitlines():
            tokens = line.split()
            if len(tokens) >= 4:
                artist = re.sub(r"[^a-z0-9]", "", tokens[3].lower())
                artist = artist.replace("kevinmacload", "kevinmacleod")
                artists[tokens[0]] = artist
    rows = []
    for path in walk_audio(noise_root):
        relative = path.relative_to(noise_root)
        corpus = relative.parts[0]
        stem = path.stem
        pin = None
        if corpus == "ESC-50":
            meta = esc[path.name]
            origin = "freesound:" + meta["src_file"]
            category = meta["category"]
            source = "ESC-50"
            pin = {"1": "train", "2": "train", "3": "train",
                   "4": "dev", "5": "test"}[meta["fold"]]
        elif corpus == "DNS-Challenge":
            match = re.search(r"Freesound_validated_(\d+)(?:_|$)", stem, re.I)
            if match:
                origin = "freesound:" + match[1]
                category = re.split("_Freesound_validated_", stem, flags=re.I)[0]
                source = "DNS-Freesound"
            elif re.fullmatch(r"[A-Za-z0-9_-]{11}", stem):
                origin = "youtube:" + stem
                category = "audioset"
                source = "DNS-AudioSet"
            elif "Freesound_validated_" in stem:
                origin = "freesound-name:" + re.sub(
                    r"_\d+$", "", stem.split("Freesound_validated_", 1)[1])
                category = stem.split("_Freesound_validated_", 1)[0]
                source = "DNS-Freesound"
            else:
                raise ValueError(f"Unrecognized DNS noise source: {path}")
        elif corpus == "musan":
            origin = "musan:" + stem
            category = "/".join(relative.parts[1:-1])
            source = "MUSAN"
        else:
            raise ValueError(f"Unrecognized noise corpus: {path}")
        keys = ["origin:" + origin]
        if corpus == "musan" and stem in artists:
            keys.append("musan-artist:" + artists[stem])
        rows.append(dict(kind="noise", path=str(path.absolute()), corpus=corpus,
                         source=source, category=category, origin_ids=[origin],
                         group_keys=keys, pinned_split=pin))
    original_real = rir_root / "RIRS_NOISES" / "real_rirs_isotropic_noises"
    known_real = {Path(line.split()[-1]).stem
                  for line in (original_real / "rir_list").read_text().splitlines()
                  if line.strip() and not line.startswith("#")}
    for path in walk_audio(rir_root):
        relative = path.relative_to(rir_root)
        if "pointsource_noises" in relative.parts or "_noise_" in path.stem:
            continue
        if "simulated_rirs" in str(relative):
            size = next((s for s in ("smallroom", "mediumroom", "largeroom")
                         if s in relative.parts), None)
            room = path.parent.name
            if size is None or not re.fullmatch(r"Room\d+", room):
                raise ValueError(f"Unrecognized simulated RIR: {path}")
            room_id = f"slr26:{size}:{room}"
            origin = f"{room_id}:{path.stem}"
            source = "simulated"
            category = size.replace("room", "")
            pin = None
        else:
            if path.stem not in known_real:
                raise ValueError(f"RIR missing a verified SLR28 origin: {path}")
            room_id, train_only = real_room(path.stem)
            origin = "slr28:" + path.stem
            source = "real"
            category = next((size for size in ("small", "medium", "large")
                             if size + "room" in path.stem), "unknown")
            pin = "train" if train_only else None
        rows.append(dict(kind="rir", path=str(path.absolute()),
                         corpus=relative.parts[0], source=source,
                         category=category, room_id=room_id, origin_ids=[origin],
                         group_keys=["origin:" + origin, "room:" + room_id],
                         pinned_split=pin))
    if not rows or not any(r["kind"] == "noise" for r in rows):
        raise ValueError("No augmentation assets found")
    for row in rows:
        path = Path(row["path"])
        info = sf.info(str(path))
        stat = path.stat()
        if info.frames <= 0 or info.samplerate <= 0:
            raise ValueError(f"Empty/invalid audio: {path}")
        row.update(sample_rate=info.samplerate, channels=info.channels,
                   frames=info.frames, duration=info.duration,
                   size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    return sorted(rows, key=lambda r: (r["kind"], r["path"]))


def audio_fingerprint(job: tuple[str, bool]) -> dict:
    """Canonical decoded PCM plus gain-normalized overlapping audio shingles.

    Four-second windows catch shared extracts at a one-second alignment; they
    are not a claim to recognize arbitrary time stretching or arbitrary crops.
    """
    path, windows = job
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if not np.isfinite(audio).all():
        raise ValueError(f"Non-finite audio: {path}")
    mono = audio.mean(axis=1)
    if sr != 16000:
        divisor = math.gcd(sr, 16000)
        mono = resample_poly(mono, 16000 // divisor, sr // divisor)
    pcm = np.clip(np.rint(mono * 32768), -32768, 32767).astype("<i2")
    whole = hashlib.sha256(pcm.tobytes()).hexdigest()
    shingles = set()
    if windows:
        low = resample_poly(mono, 1, 8)
        for start in range(0, len(low) - 8000 + 1, 2000):
            chunk = low[start:start + 8000]
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            # Exclude silence and near-constant/tone-like windows from linkage.
            if rms < 1e-5 or float(np.std(np.diff(chunk))) < rms * 0.03:
                continue
            scale = max(float(np.max(np.abs(chunk))), 1e-12)
            quantized = np.rint(chunk / scale * 100).astype(np.int8)
            shingles.add(hashlib.sha256(quantized.tobytes()).hexdigest()[:32])
    return dict(pcm_sha256=whole, window_hashes=sorted(shingles),
                silent=bool(float(np.max(np.abs(mono))) < 1e-7))


def fingerprint_rows(rows: list[dict], cache: Path, workers: int, rehash=False):
    db = sqlite3.connect(cache)
    db.execute("CREATE TABLE IF NOT EXISTS fingerprints "
               "(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    missing = []
    for row in rows:
        key = digest(json.dumps([FINGERPRINT_VERSION, row["path"],
                                 row["size_bytes"], row["mtime_ns"]]))
        row["_cache_key"] = key
        hit = None if rehash else db.execute(
            "SELECT value FROM fingerprints WHERE key=?", (key,)).fetchone()
        if hit:
            row.update(json.loads(hit[0]))
        else:
            missing.append(row)
    print(f"[fingerprint] cached={len(rows)-len(missing)} pending={len(missing)}",
          flush=True)
    jobs = [(r["path"], r["kind"] == "noise") for r in missing]
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        for number, (row, result) in enumerate(
                zip(missing, pool.map(audio_fingerprint, jobs, chunksize=4)), 1):
            row.update(result)
            db.execute("INSERT OR REPLACE INTO fingerprints VALUES (?,?)",
                       (row["_cache_key"], json.dumps(result)))
            if number % 1000 == 0:
                db.commit()
                print(f"[fingerprint] {number}/{len(missing)}", flush=True)
    db.commit()
    db.close()
    for row in rows:
        row.pop("_cache_key", None)


class UnionFind:
    def __init__(self, n):
        self.parents = list(range(n))

    def find(self, value):
        while value != self.parents[value]:
            self.parents[value] = self.parents[self.parents[value]]
            value = self.parents[value]
        return value

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parents[max(a, b)] = min(a, b)


def canonical_assets(rows: list[dict]) -> list[dict]:
    """Bind sample-rate aliases first, then collapse identical decoded audio."""
    by_origin = defaultdict(list)
    for row in rows:
        key = (row["kind"], row["origin_ids"][0]) if row["kind"] == "rir" else (
            "noise", row["pcm_sha256"])
        by_origin[key].append(row)
    result = []
    for originals in by_origin.values():
        chosen = min(originals, key=lambda r: (
            0 if r["sample_rate"] == 16000 else 1,
            0 if r["corpus"] in ("RIRS_NOISES", "ESC-50") else 1, r["path"]))
        row = dict(chosen)
        row["aliases"] = [dict(r) for r in originals if r is not chosen]
        row["origin_ids"] = sorted({x for r in originals for x in r["origin_ids"]})
        row["group_keys"] = sorted({x for r in originals for x in r["group_keys"]})
        row["content_hashes"] = sorted({r["pcm_sha256"] for r in originals if r["pcm_sha256"]})
        row["window_hashes"] = sorted({x for r in originals for x in r["window_hashes"]})
        pins = [r["pinned_split"] for r in originals if r["pinned_split"]]
        row["pinned_split"] = max(pins, key=PRIORITY.get) if pins else None
        row["asset_id"] = row["kind"] + ":" + digest("|".join(row["origin_ids"]))[:24]
        result.append(row)
    return sorted(result, key=lambda r: r["asset_id"])


def assign_splits(rows: list[dict], seed: int):
    union = UnionFind(len(rows))
    seen = {}
    for index, row in enumerate(rows):
        keys = list(row["group_keys"])
        if not row["silent"]:
            keys += ["pcm:" + h for h in row["content_hashes"]]
            keys += ["window:" + h for h in row["window_hashes"]]
        for key in keys:
            key = row["kind"] + ":" + key
            if key in seen:
                union.union(index, seen[key])
            else:
                seen[key] = index
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[union.find(index)].append(row)

    totals = Counter()
    weights = {}
    pins = {}
    for group, items in groups.items():
        vector = Counter()
        for row in items:
            if row["kind"] == "noise" and row["source"] == "ESC-50":
                continue  # Official folds pin these components.
            stratum = (row["kind"], row["source"], row["category"])
            vector[stratum] += row["duration"] if row["kind"] == "noise" else 1
        weights[group] = vector
        totals.update(vector)
        values = [r["pinned_split"] for r in items if r["pinned_split"]]
        pins[group] = max(values, key=PRIORITY.get) if values else None

    used = {s: Counter() for s in SPLITS}
    assignments = {}
    for group in groups:
        if pins[group]:
            split = pins[group]
            assignments[group] = split
            used[split].update(weights[group])

    def order(group):
        impact = sum(v / max(totals[k], 1e-9) for k, v in weights[group].items())
        name = min(r["asset_id"] for r in groups[group])
        return -impact, digest(f"{seed}:{name}")

    for group in sorted((g for g in groups if g not in assignments), key=order):
        def cost(split):
            ratio = RATIOS[PRIORITY[split]]
            value = 0.0
            for key, amount in weights[group].items():
                target = totals[key] * ratio
                before = used[split][key] - target
                value += ((before + amount) ** 2 - before ** 2) / max(target, 1e-9)
            return value, digest(f"{seed}:{group}:{split}")
        split = min(SPLITS, key=cost)
        assignments[group] = split
        used[split].update(weights[group])

    for group, items in groups.items():
        group_keys = sorted({key for row in items for key in row["group_keys"]})
        group_id = "group:" + digest("\n".join(group_keys))[:24]
        for row in items:
            row["split"] = assignments[group]
            row["group_id"] = group_id
            row["schema_version"] = 1


def write_jsonl(path: Path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def relative_row(row: dict, output: Path) -> dict:
    row = dict(row)
    row["path"] = os.path.relpath(row["path"], output)
    row["aliases"] = [relative_row(alias, output) for alias in row.get("aliases", [])]
    return row


def manifest_rows(output: Path):
    for kind in ("noise", "rir"):
        for split in SPLITS:
            path = output / f"{kind}.{split}.jsonl"
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if row.get("kind") != kind or row.get("split") != split:
                        raise ValueError(f"Wrong kind/split in {path}")
                    yield row


def verify(output: Path, *, check_files=True) -> dict:
    """Independent cross-split audit of identities, aliases and fingerprints."""
    rows = list(manifest_rows(output))
    overlaps = Counter()
    seen = {}
    coverage = set()
    counts = defaultdict(Counter)
    groups = defaultdict(set)
    for row in rows:
        kind, split = row["kind"], row["split"]
        counts[kind][split] += 1
        groups[kind].add(row["group_id"])
        keys = ["group:" + row["group_id"]]
        keys += ["origin:" + value for value in row["origin_ids"]]
        keys += ["binding:" + value for value in row["group_keys"]]
        if not row["silent"]:
            keys += ["pcm:" + value for value in row["content_hashes"]]
            keys += ["window:" + value for value in row["window_hashes"]]
        for version in [row] + row.get("aliases", []):
            path = (output / version["path"]).resolve()
            keys.append("path:" + str(path))
            if path in coverage:
                raise ValueError(f"Input version occurs more than once: {path}")
            coverage.add(path)
            if check_files:
                stat = path.stat()
                if stat.st_size != version["size_bytes"] or stat.st_mtime_ns != version["mtime_ns"]:
                    raise ValueError(f"Asset changed after fingerprinting: {path}")
            if not set(version["origin_ids"]) <= set(row["origin_ids"]):
                raise ValueError(f"Alias origin missing from canonical asset: {path}")
            if not set(version["group_keys"]) <= set(row["group_keys"]):
                raise ValueError(f"Alias group missing from canonical asset: {path}")
        for key in keys:
            identity = (kind, key)
            previous = seen.setdefault(identity, split)
            if previous != split:
                overlaps[key.split(":", 1)[0]] += 1
    with (output / "inventory.jsonl").open() as handle:
        inputs = [json.loads(line) for line in handle]
    expected = {(output / row["path"]).resolve() for row in inputs}
    if coverage != expected:
        raise ValueError(f"Inventory coverage differs: missing={len(expected-coverage)}, "
                         f"extra={len(coverage-expected)}")
    if overlaps:
        raise ValueError(f"Cross-split leakage: {dict(overlaps)}")
    for kind in ("noise", "rir"):
        if any(counts[kind][split] == 0 for split in SPLITS):
            raise ValueError(f"Empty split for {kind}")
    return dict(passed=True, input_versions=len(inputs),
                covered_versions=len(coverage),
                cross_split_overlaps={name: 0 for name in
                    ("origin", "binding", "group", "pcm", "window", "path")},
                counts={kind: dict(counts[kind]) for kind in counts},
                group_counts={kind: len(value) for kind, value in groups.items()},
                fingerprint_version=FINGERPRINT_VERSION,
                scope="Known recording IDs, conservative room/artist groups, aliases, "
                      "decoded mono16k PCM, gain-normalized 4s windows on a 1s grid. "
                      "This does not prove arbitrary edited/unattributed audio unrelated.")


def build(args):
    output = Path(args.output_dir).absolute()
    if (output / "summary.json").exists():
        raise ValueError("Frozen split already exists; use --verify or a new output_dir")
    output.mkdir(parents=True, exist_ok=True)
    noise = Path(args.noise_dir).absolute()
    rir = Path(args.rir_dir).absolute()
    esc = Path(args.esc_metadata).absolute() if args.esc_metadata else (
        (noise / "ESC-50").resolve().parent / "meta" / "esc50.csv")
    rows = inventory(noise, rir, esc)
    print("[inventory]", dict(Counter(r["kind"] for r in rows)), flush=True)
    # RIR sample-rate variants already have an authoritative origin binding.
    # Hash each canonical RIR once; all noise files receive full fingerprints.
    rir_origins = defaultdict(list)
    for row in rows:
        if row["kind"] == "rir":
            rir_origins[row["origin_ids"][0]].append(row)
    chosen = {min(versions, key=lambda r: (
        r["sample_rate"] != 16000, r["corpus"] != "RIRS_NOISES", r["path"]))["path"]
        for versions in rir_origins.values()}
    fingerprinted = [r for r in rows if r["kind"] == "noise" or r["path"] in chosen]
    fingerprint_rows(fingerprinted, output / "fingerprints.sqlite", args.workers, args.rehash)
    by_origin = {r["origin_ids"][0]: r for r in fingerprinted if r["kind"] == "rir"}
    for row in rows:
        if "pcm_sha256" not in row:
            original = by_origin[row["origin_ids"][0]]
            row.update(pcm_sha256=None, window_hashes=[],
                       silent=original["silent"], alias_of=original["path"])
    assets = canonical_assets(rows)
    assign_splits(assets, args.seed)
    write_jsonl(output / "inventory.jsonl", (relative_row(r, output) for r in rows))
    for kind in ("noise", "rir"):
        for split in SPLITS:
            write_jsonl(output / f"{kind}.{split}.jsonl",
                        (relative_row(row, output) for row in assets
                         if row["kind"] == kind and row["split"] == split))
    report = verify(output)
    report.update(seed=args.seed, ratios=dict(zip(SPLITS, RATIOS)),
                  input_counts=dict(Counter(r["kind"] for r in rows)),
                  canonical_counts=dict(Counter(r["kind"] for r in assets)),
                  aliases=sum(len(r["aliases"]) for r in assets))
    statistics = {}
    for kind in ("noise", "rir"):
        statistics[kind] = {}
        for split in SPLITS:
            selected = [r for r in assets if r["kind"] == kind and r["split"] == split]
            statistics[kind][split] = dict(
                files=len(selected),
                hours=round(sum(r["duration"] for r in selected) / 3600, 6),
                groups=len({r["group_id"] for r in selected}),
                sources=dict(Counter(r["source"] for r in selected)),
                categories=dict(Counter(r["category"] for r in selected)))
    report["statistics"] = statistics
    report["manifest_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.glob("*.jsonl"))}
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise_dir", default="data/augmentation/noise")
    parser.add_argument("--rir_dir", default="data/augmentation/rir")
    parser.add_argument("--esc_metadata")
    parser.add_argument("--output_dir", default="data/augmentation/splits/v1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rehash", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        output = Path(args.output_dir).absolute()
        summary = json.loads((output / "summary.json").read_text())
        for name, expected in summary["manifest_sha256"].items():
            if hashlib.sha256((output / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Frozen manifest modified: {name}")
        print(json.dumps(verify(output), ensure_ascii=False, indent=2))
    else:
        build(args)


if __name__ == "__main__":
    main()
