#!/usr/bin/env python3
"""Build full HM-ASR test sets with frozen held-out noise and paired controls."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import io
import json
import multiprocessing
from pathlib import Path
import shutil
import sys
import tarfile

import numpy as np
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetuning.augmentation.assets import read_asset_manifest
from finetuning.augmentation.simulator.core import NoisePool
from tools.build_controlled_robust_eval import (
    NOISE_FAMILIES, mix_at_snr, noise_family, pcm_hash, rms,
    sha256_file, shared_peak_gain, stable_seed,
)

CONDITIONS = ("clean", "noise_snr20", "noise_snr10", "noise_snr0")
EXCLUDED_PREFIXES = ("alimeeting", "ami_", "renji")


def excluded(name):
    return name.lower().startswith(EXCLUDED_PREFIXES)


def family_assignments(ids, seed, dataset):
    ordered = sorted(ids, key=lambda key: stable_seed(seed, dataset, key, "noise-family"))
    if len(ordered) != len(set(ordered)):
        raise ValueError("Duplicate source speech IDs")
    return {key: NOISE_FAMILIES[i % len(NOISE_FAMILIES)] for i, key in enumerate(ordered)}


INVALID_SPEECH_POLICY = "Quarantine proven empty/nonfinite/all-zero source PCM; account for every source ID and evaluate every valid utterance."


def invalid_speech_reason(wav):
    wav = np.asarray(wav)
    if wav.size == 0:
        return "empty_pcm"
    if not np.isfinite(wav).all():
        return "nonfinite_pcm"
    if rms(wav) <= 1e-12:
        return "silent_pcm"
    return None


def render_variants(signal, noise):
    """Preserve complete duration; all conditions share normalization and gain."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 1 or not signal.size or not np.isfinite(signal).all() or rms(signal) <= 1e-12:
        raise ValueError("Speech must be non-empty, finite and non-silent")
    normalization_gain = .05 / rms(signal)
    clean = np.asarray(signal * normalization_gain, dtype=np.float32)
    variants = {"clean": clean}
    for snr in (20, 10, 0):
        variants[f"noise_snr{snr}"] = mix_at_snr(clean, noise, snr)[0]
    gain = shared_peak_gain(variants)
    encoded, decoded = {}, {}
    for name, wav in variants.items():
        buffer = io.BytesIO()
        sf.write(buffer, wav * gain, 16000, format="FLAC", subtype="PCM_16")
        encoded[name] = buffer.getvalue()
        actual, sr = sf.read(io.BytesIO(encoded[name]), dtype="float32")
        if sr != 16000 or actual.ndim != 1 or len(actual) != len(signal):
            raise ValueError("Encoded audio changed rate/channels/duration")
        if not np.isfinite(actual).all() or np.max(np.abs(actual)) > .981:
            raise ValueError("Invalid or clipped encoded audio")
        decoded[name] = actual
    measured = {}
    for snr in (20, 10, 0):
        name = f"noise_snr{snr}"
        value = float(20*np.log10(rms(decoded["clean"])/rms(decoded[name]-decoded["clean"])))
        if abs(value-snr) > .1:
            raise ValueError(f"PCM16 SNR calibration failed: target={snr}, measured={value}")
        measured[name] = value
    return encoded, normalization_gain, gain, measured


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")
    tmp.replace(path)


def digest_bytes(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


class ShardWriter:
    def __init__(self, root, name, max_samples, min_free_gb):
        self.name = name
        self.root = Path(root)/name
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples
        self.min_free = int(min_free_gb * 1024**3)
        self.manifest = (self.root/"manifest.jsonl").open("w")
        self.archive = None
        self.shards = []
        self.count = self.in_shard = 0
        self.seconds = 0.
        self.closed = False

    def close_shard(self):
        if self.archive is None:
            return
        self.archive.close()
        path = self.root/self.shard_name
        self.shards.append(dict(file=self.shard_name, bytes=path.stat().st_size,
                                sha256=sha256_file(path), samples=self.in_shard))
        self.archive = None

    def add(self, key, flac, meta):
        if self.archive is None or self.in_shard >= self.max_samples:
            self.close_shard()
            if shutil.disk_usage(self.root).free < self.min_free:
                raise RuntimeError("Free disk is below the configured reserve")
            self.shard_name = f"{self.name}-{len(self.shards):05d}.tar"
            self.archive = tarfile.open(self.root/self.shard_name, "w", format=tarfile.PAX_FORMAT)
            self.in_shard = 0
        item = dict(meta, id=key, dataset=self.name, shard=self.shard_name,
                    audio_sha256=digest_bytes(flac))
        content = [("flac", flac), ("txt", item["reference"].encode()),
                   ("json", json.dumps(item, ensure_ascii=False).encode())]
        for ext, data in content:
            info = tarfile.TarInfo(key+"."+ext)
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            self.archive.addfile(info, io.BytesIO(data))
        self.manifest.write(json.dumps(item, ensure_ascii=False)+"\n")
        self.count += 1
        self.in_shard += 1
        self.seconds += item["duration"]

    def finish(self):
        self.close_shard()
        self.manifest.close()
        self.closed = True
        stats = dict(dataset=self.name, n_samples=self.count, seconds=self.seconds,
                     hours=self.seconds/3600, n_shards=len(self.shards), shards=self.shards,
                     format="webdataset/flac-pcm16-16khz-mono",
                     manifest_sha256=sha256_file(self.root/"manifest.jsonl"))
        write_json(self.root/"STATS.json", stats)
        return stats

    def abort(self):
        if self.archive is not None:
            self.archive.close()
            self.archive = None
        if not self.closed:
            self.manifest.close()
            self.closed = True


def benchmark_api(root):
    root = Path(root)
    sys.path.insert(0, str(root/"src"))
    sys.path.insert(0, str(root/"third_party"))
    from asr_benchmark.data import load_config, load_dataset_rows, dataset_fingerprint
    from asr_benchmark.runner import _iter_webdataset_audio
    return load_config, load_dataset_rows, dataset_fingerprint, _iter_webdataset_audio


def verify_dataset(output, source, assets):
    out = Path(output)
    protocol = json.loads((out/"protocol.json").read_text())
    spec = protocol["sources"][source]
    allowed = {(r["asset_id"], str(r["path"])): r["group_id"]
               for r in read_asset_manifest(Path(assets)/"noise.test.jsonl", "noise", "test")}
    source_manifest = Path(spec["webdataset"])/"manifest.jsonl"
    assert sha256_file(source_manifest) == spec["manifest_sha256"]
    originals = [json.loads(line) for line in source_manifest.open()]
    if protocol["smoke_limit"]:
        originals = originals[:protocol["smoke_limit"]]
    assert len(originals) == spec["n_utts"]
    original_index = {row["id"]:row for row in originals}
    quarantine_path = out/"quarantine"/(source+".json")
    quarantined = json.loads(quarantine_path.read_text()) if quarantine_path.exists() else []
    invalid_ids = {row["id"] for row in quarantined}
    assert len(invalid_ids) == len(quarantined) and invalid_ids <= set(original_index)
    for row in quarantined:
        original = original_index[row["id"]]
        assert row["reference"] == original["reference"]
        with tarfile.open(Path(spec["webdataset"])/original["shard"], "r:") as archive:
            wav,sr = sf.read(io.BytesIO(archive.extractfile(row["id"]+".flac").read()),dtype="float32",always_2d=True)
        wav = wav.mean(axis=1)
        if sr != 16000:
            import librosa
            wav = librosa.resample(wav,orig_sr=sr,target_sr=16000,res_type="soxr_hq")
        assert invalid_speech_reason(wav) == row["reason"]
        assert digest_bytes(np.asarray(wav,dtype=np.float32).tobytes()) == row["source_float32_sha256"]
    expected = spec["n_utts"]-len(invalid_ids)
    if expected <= 0:
        raise ValueError("No valid speech remains in "+source)
    control = None
    noise_references = {}
    result = {}
    used_paths, used_groups = set(), set()
    for condition in CONDITIONS:
        name = source+"__"+condition
        folder = out/"webdataset"/name
        stats = json.loads((folder/"STATS.json").read_text())
        assert stats["manifest_sha256"] == sha256_file(folder/"manifest.jsonl")
        entries = [json.loads(line) for line in (folder/"manifest.jsonl").open()]
        indexed = {r["id"]:r for r in entries}
        assert len(indexed) == len(entries) == stats["n_samples"] == expected
        assert all(r["condition"] == condition and r["split"] == "test" for r in entries)
        if control is None:
            control = indexed
        assert set(indexed) == set(control)
        found = set()
        for shard in stats["shards"]:
            path = folder/shard["file"]
            assert path.stat().st_size == shard["bytes"]
            assert sha256_file(path) == shard["sha256"]
            pending = {}
            with tarfile.open(path, "r:") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    key, _, ext = member.name.rpartition(".")
                    assert ext in {"flac", "txt", "json"}
                    pending.setdefault(key, {})[ext] = archive.extractfile(member).read()
                    parts = pending[key]
                    if len(parts) != 3:
                        continue
                    row = indexed[key]
                    assert json.loads(parts["json"]) == row
                    assert parts["txt"].decode() == row["reference"]
                    assert digest_bytes(parts["flac"]) == row["audio_sha256"]
                    info = sf.info(io.BytesIO(parts["flac"]))
                    assert info.samplerate == 16000 and info.channels == 1
                    assert info.frames == row["frames"]
                    assert row["reference"] == control[key]["reference"]
                    assert row["shared_gain"] == control[key]["shared_gain"]
                    assert row["speech_origin"] == control[key]["speech_origin"]
                    assert row["source_id"] == control[key]["source_id"]
                    assert row["frames"] == control[key]["frames"]
                    assert row["noise_manifest_sha256"] == protocol["noise_manifest_sha256"]["test"]
                    if condition != "clean":
                        trace = row["trace"]["noise"]
                        if key in noise_references:
                            assert trace == noise_references[key]
                        else:
                            noise_references[key] = trace
                        assert allowed[(trace["asset_id"], trace["path"])] == trace["group_id"]
                        assert trace["source_family"] in NOISE_FAMILIES
                        assert trace["num_samples"] == row["frames"]
                        assert abs(row["trace"]["actual_encoded_snr_db"]-row["trace"]["snr_db"]) <= .1
                        used_paths.add(trace["path"])
                        used_groups.add(trace["group_id"])
                    assert key not in found
                    found.add(key)
                    del pending[key]
            assert not pending
        assert found == set(indexed)
        result[condition] = dict(n_utts=expected, manifest_sha256=stats["manifest_sha256"],
                                 shard_sha256={s["file"]:s["sha256"] for s in stats["shards"]},
                                 bytes=sum(s["bytes"] for s in stats["shards"]))
    assert set(control) == set(original_index)-invalid_ids
    assert set(control).isdisjoint(invalid_ids)
    assert all(control[key]["reference"] == original_index[key]["reference"] for key in control)
    report = dict(passed=True, source_dataset=source, source_speech_utterances=spec["n_utts"],
                  evaluated_speech_utterances=expected, invalid_source_utterances=len(invalid_ids),
                  invalid_source_ids=sorted(invalid_ids),
                  evaluated_seconds=sum(r["duration"] for r in control.values()),
                  rendered_utterances=expected*len(CONDITIONS), conditions=result,
                  all_encoded_audio_sha256_verified=True,
                  snr_verified_after_encoding_before_sharding=True,
                  test_noise_membership_verified=True, noise_assets=len(used_paths),
                  noise_groups=len(used_groups), unaccounted_source_utterances=0)
    return report


def build_dataset(job):
    output, source, benchmark_root, assets = map(Path, [job["output"], job["source_name"],
                                                       job["benchmark_root"], job["assets"]])
    source = str(source)
    protocol = json.loads((output/"protocol.json").read_text())
    ready = output/"ready"/(source+".json")
    if ready.exists():
        previous = json.loads(ready.read_text())
        report = verify_dataset(output, source, assets)
        return dict(previous, **report)
    load_config, load_rows, fingerprint, iterate = benchmark_api(benchmark_root)
    spec = load_config(output/"source_datasets.yaml")[source]
    rows = load_rows(spec, limit=protocol["smoke_limit"])
    original_fp = fingerprint(spec, rows, limit=protocol["smoke_limit"])
    if spec.webdataset is None:
        raise ValueError("This builder requires pre-segmented HM-ASR WebDataset inputs")
    assignment = family_assignments([r.id for r in rows], protocol["seed"], source)
    noise = NoisePool(manifest_path=str(assets/"noise.test.jsonl"), split="test", max_cached_files=16)
    if not noise.files:
        raise ValueError("The held-out test noise pool is empty")
    buckets = defaultdict(list)
    for path, row in noise._assets.items():
        buckets[noise_family(row)].append(path)
    assert set(buckets) == set(NOISE_FAMILIES)
    pools = {}
    for family, paths in buckets.items():
        pool = copy.copy(noise)
        pool.files = sorted(paths)
        pools[family] = pool
    writers = {}
    for condition in CONDITIONS:
        name = source+"__"+condition
        folder = output/"webdataset"/name
        if folder.exists():
            assert folder.parent == output/"webdataset"
            shutil.rmtree(folder)
        writers[condition] = ShardWriter(output/"webdataset", name,
                                         protocol["shard_samples"], protocol["min_free_gb"])
    maximum_error = 0.
    family_counts = Counter()
    processed = 0
    seen = 0
    quarantined = []
    try:
        for row, (wav, sr) in iterate({"spec":spec, "rows":rows}, {r.id for r in rows}):
            if sr != 16000:
                import librosa
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000, res_type="soxr_hq")
            wav = np.asarray(wav, dtype=np.float32)
            seen += 1
            invalid = invalid_speech_reason(wav)
            if invalid:
                quarantined.append(dict(id=row.id, reference=row.text, reason=invalid,
                                        frames=len(wav), duration=len(wav)/16000,
                                        source_float32_sha256=digest_bytes(wav.tobytes())))
                print(json.dumps(dict(quarantined=source,id=row.id,reason=invalid)),flush=True)
                continue
            family = assignment[row.id]
            pool = pools[family]
            rng = np.random.default_rng(stable_seed(protocol["seed"], source, row.id, "noise-crop"))
            for retry in range(50):
                noise_clip = pool.sample(len(wav), rng)
                if rms(noise_clip) > 1e-6:
                    break
            else:
                raise ValueError(f"{source}/{row.id}: no non-silent held-out noise window")
            trace = dict(pool.last_sample, source_family=family)
            native_samples = len(pool._load(Path(trace["path"])))
            trace.update(source_num_samples=native_samples, tiled=native_samples<len(wav))
            try:
                variants, normalization_gain, gain, measured = render_variants(wav, noise_clip)
            except Exception as exc:
                raise ValueError(f"{source}/{row.id}: {exc}") from exc
            origin = dict(dataset=source, utterance_id=row.id, source_id=row.source_id,
                          source_pcm_sha256=pcm_hash(wav), source_manifest_sha256=original_fp.get("manifest_sha256"))
            for condition, data in variants.items():
                corruption = {}
                if condition != "clean":
                    snr = int(condition.removeprefix("noise_snr"))
                    maximum_error = max(maximum_error, abs(measured[condition]-snr))
                    corruption = dict(noise=trace, snr_db=snr, actual_encoded_snr_db=measured[condition])
                meta = dict(reference=row.text, language=row.language, split="test",
                            source_id=f"test/{source}/{row.id}", speech_origin=origin,
                            duration=len(wav)/16000, frames=len(wav), sample_rate=16000, channels=1,
                            start=None, end=None, condition=condition, augmentation_split="test",
                            noise_manifest_sha256=noise.manifest_sha256, trace=corruption,
                            normalization_gain=normalization_gain, shared_gain=gain)
                writers[condition].add(row.id, data, meta)
            family_counts[family] += 1
            processed += 1
            if processed % 1000 == 0:
                print(json.dumps(dict(prepared=source, n=processed, total=len(rows))), flush=True)
        if seen != len(rows) or processed+len(quarantined) != len(rows):
            raise ValueError(f"{source}: source coverage mismatch")
        (output/"quarantine").mkdir(exist_ok=True)
        write_json(output/"quarantine"/(source+".json"),quarantined)
        for writer in writers.values():
            writer.finish()
    except BaseException:
        for writer in writers.values():
            writer.abort()
        raise
    report = verify_dataset(output, source, assets)
    report.update(source_fingerprint=original_fp, noise_family_counts=dict(family_counts),
                  max_encoded_snr_error_db=maximum_error)
    write_json(ready, report)
    print(json.dumps(dict(ready=source, source_utterances=seen, evaluated_utterances=processed,
                          quarantined_utterances=len(quarantined), audio_utterances=4*processed)), flush=True)
    return report


def build(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/"generation_summary.json").exists():
        raise FileExistsError("Frozen suite exists; use --verify_only or a new version")
    benchmark_root = Path(args.benchmark_root).resolve()
    assets = Path(args.assets_dir).resolve()
    load_config, load_rows, _, _ = benchmark_api(benchmark_root)
    source_config = Path(args.source_config).resolve() if args.source_config else benchmark_root/"configs/datasets.yaml"
    specs = load_config(source_config)
    selected = {n:s for n,s in specs.items() if s.enabled and not excluded(n)}
    excluded_names = [n for n,s in specs.items() if s.enabled and excluded(n)]
    if not selected:
        raise ValueError("No included source datasets")
    normalized = {}
    sources = {}
    for name, spec in selected.items():
        if spec.webdataset is None:
            raise ValueError(f"{name}: WebDataset source required")
        rows = load_rows(spec, limit=args.smoke_limit)
        manifest = spec.webdataset/"manifest.jsonl"
        hours = sum(r.duration or 0 for r in rows)/3600
        sources[name] = dict(n_utts=len(rows), hours=hours, webdataset=str(spec.webdataset),
                             manifest_sha256=sha256_file(manifest), language=spec.language,
                             metric=spec.metric, normalizer=spec.normalizer)
        normalized[name] = dict(enabled=True, tier=spec.tier, webdataset=str(spec.webdataset),
                                language=spec.language, metric=spec.metric,
                                normalizer=spec.normalizer, split=spec.split, source=spec.source)
    manifest_hashes = {s:sha256_file(assets/f"noise.{s}.jsonl") for s in ["train","dev","test"]}
    groups = {s:{r["group_id"] for r in read_asset_manifest(assets/f"noise.{s}.jsonl","noise",s)}
              for s in ["train","dev","test"]}
    for a,b in [("train","dev"),("train","test"),("dev","test")]:
        if groups[a]&groups[b]:
            raise ValueError("Noise source groups cross the frozen split")
    hours = sum(s["hours"] for s in sources.values())
    upper_bound = hours*3600*32000*len(CONDITIONS)*1.08
    completed_bytes = 0
    if args.resume and (output/"ready").exists():
        for path in (output/"ready").glob("*.json"):
            completed_bytes += sum(c["bytes"] for c in json.loads(path.read_text())["conditions"].values())
    remaining_bound = max(0,upper_bound-completed_bytes)
    if not args.smoke_limit and shutil.disk_usage(output).free < remaining_bound+args.min_free_gb*1024**3:
        raise RuntimeError("Insufficient free disk for remaining conservative PCM16 size plus reserve")
    protocol = dict(schema_version=1, sources=sources, excluded_datasets=excluded_names,
                    conditions=CONDITIONS, seed=args.seed, smoke_limit=args.smoke_limit,
                    full_source_coverage=args.smoke_limit==0, sample_rate=16000,
                    source_duration_filter=None, max_duration_seconds=None,
                    invalid_speech_policy=INVALID_SPEECH_POLICY,
                    noise_manifest_sha256=manifest_hashes, asset_split="test",
                    noise_families=NOISE_FAMILIES,
                    family_assignment="Seeded hash rank of source IDs, balanced six families; independent of manifest/dialect ordering.",
                    snr_db=[20,10,0], snr_definition="Full clean waveform RMS / added noise RMS.",
                    speech_target_rms=.05, peak_handling="One shared gain across clean/20/10/0 dB; no hard clipping.",
                    clean_reference="Original recording with shared level control; no denoising. Re-infer both models.",
                    rir_enabled=False, speed_or_specaugment=False,
                    shard_samples=args.shard_samples, min_free_gb=args.min_free_gb,
                    source_config_sha256=sha256_file(source_config))
    if (output/"protocol.json").exists():
        old = json.loads((output/"protocol.json").read_text())
        if not args.resume or old != json.loads(json.dumps(protocol)):
            raise ValueError("Existing suite protocol differs or --resume is missing")
    else:
        write_json(output/"protocol.json", protocol)
    (output/"source_datasets.yaml").write_text(yaml.safe_dump(dict(datasets=normalized),allow_unicode=True,sort_keys=False))
    (output/"ready").mkdir(exist_ok=True)
    yaml_sets, mapping = {}, {}
    for source, spec in sources.items():
        for condition in CONDITIONS:
            name = source+"__"+condition
            path = output/"webdataset"/name
            yaml_sets[name] = dict(enabled=True,tier="robustness",webdataset=str(path),
                                   language=spec["language"],normalizer=spec["normalizer"],
                                   metric=spec["metric"],split="test",source=source+" + frozen test noise")
            mapping[name] = dict(split="test",corpus=source,condition=condition,metric=spec["metric"],
                                 language=spec["language"],manifest=str(path/"manifest.jsonl"),
                                 n_utts=spec["n_utts"])
    (output/"datasets.yaml").write_text(yaml.safe_dump(dict(datasets=yaml_sets),allow_unicode=True,sort_keys=False))
    write_json(output/"dataset_map.json", mapping)
    jobs = [dict(output=str(output),source_name=n,benchmark_root=str(benchmark_root),assets=str(assets))
            for n in sorted(sources,key=lambda n:sources[n]["hours"])]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(build_dataset, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    evaluated = {r["source_dataset"]:r["evaluated_speech_utterances"] for r in results}
    for name,spec in mapping.items():
        spec["n_source_utts"] = spec["n_utts"]
        spec["n_utts"] = evaluated[spec["corpus"]]
        spec["manifest_sha256"] = sha256_file(spec["manifest"])
    write_json(output/"dataset_map.json", mapping)
    summary = dict(full_source_coverage=args.smoke_limit==0,speech_utterances=sum(s["n_utts"] for s in sources.values()),
                   evaluated_speech_utterances=sum(r["evaluated_speech_utterances"] for r in results),
                   invalid_source_utterances=sum(r["invalid_source_utterances"] for r in results),
                   rendered_utterances=sum(r["rendered_utterances"] for r in results),datasets=len(mapping),
                   source_hours=hours,rendered_hours=sum(r["evaluated_seconds"] for r in results)*len(CONDITIONS)/3600,
                   excluded_datasets=excluded_names,unaccounted_source_utterances=0,conditions=CONDITIONS,
                   max_encoded_snr_error_db=max(r["max_encoded_snr_error_db"] for r in results),
                   bytes=sum(c["bytes"] for r in results for c in r["conditions"].values()),
                   verification_passed=all(r["passed"] for r in results))
    write_json(output/"generation_summary.json",summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir",required=True)
    parser.add_argument("--benchmark_root",default="/root/autodl-tmp/workspace/project/hmasr_benchmark")
    parser.add_argument("--source_config")
    parser.add_argument("--assets_dir",default=str(ROOT/"data/augmentation/splits/v1"))
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--shard_samples",type=int,default=512)
    parser.add_argument("--min_free_gb",type=float,default=25)
    parser.add_argument("--smoke_limit",type=int,default=0)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--verify_only",action="store_true")
    args=parser.parse_args()
    if args.smoke_limit<0 or args.workers<1 or args.shard_samples<1:
        parser.error("Invalid limits/workers/shard size")
    if args.verify_only:
        out=Path(args.output_dir)
        protocol=json.loads((out/"protocol.json").read_text())
        reports=[verify_dataset(out,n,args.assets_dir) for n in protocol["sources"]]
        write_json(out/"suite_verification.json",dict(passed=all(r["passed"] for r in reports),
                    source_datasets=len(reports),speech_utterances=sum(r["source_speech_utterances"] for r in reports),
                    rendered_utterances=sum(r["rendered_utterances"] for r in reports),
                    full_source_coverage=protocol["full_source_coverage"],
                    invalid_source_utterances=sum(r["invalid_source_utterances"] for r in reports),
                    evaluated_speech_utterances=sum(r["evaluated_speech_utterances"] for r in reports),
                    unaccounted_source_utterances=0,test_noise_membership_verified=True,
                    all_encoded_audio_sha256_verified=True))
    else:
        build(args)


if __name__=="__main__":
    main()
