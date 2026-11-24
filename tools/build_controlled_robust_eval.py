#!/usr/bin/env python3
"""Freeze paired speech/noise/RIR dev/test robustness sets with explicit SNR."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import copy
import csv
import hashlib
import io
import json
import re
from pathlib import Path
import sys
import subprocess
import shutil
import tarfile

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetuning.augmentation.simulator.core import NoisePool, RIRPool, load_audio

CORPORA = {
    "gigaspeech": ("English", "librispeech", "wer", "gigaspeech_cleaned_test"),
    "wenetspeech": ("Chinese", "masr_zh", "cer", "wenetspeech_test_net"),
    "commonvoice_yue": ("Cantonese", "masr_zh_yue", "cer", "commonvoice_yue_test"),
    "kespeech": ("Chinese", "masr_zh", "cer", "kespeech_test"),
}
CONDITIONS = (
    "clean", "noise_snr20", "noise_snr10", "noise_snr0",
    "rir_real", "rir_simulated", "rir_real_noise_snr10", "rir_real_noise_snr0",
    "rir_simulated_noise_snr10", "rir_simulated_noise_snr0",
)
NOISE_FAMILIES = ("dns_audioset", "dns_freesound", "esc50",
                  "musan_music", "musan_noise", "musan_speech")


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("/".join(map(str, parts)).encode()).digest()[:8], "little")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def rms(wav):
    return float(np.sqrt(np.mean(np.asarray(wav, dtype=np.float64)**2)))


def pcm_hash(wav):
    pcm = np.rint(np.clip(wav, -1, 1 - 1 / 32768) * 32768).astype("<i2")
    return hashlib.sha256(pcm.tobytes()).hexdigest()


def mix_at_snr(signal, noise, snr_db):
    if (len(signal) != len(noise) or not np.isfinite(signal).all() or not np.isfinite(noise).all()
            or rms(signal) <= 1e-8 or rms(noise) <= 1e-8):
        raise ValueError("SNR requires equal-length, non-silent finite signals")
    scaled = np.asarray(noise, dtype=np.float32) * (rms(signal) / rms(noise) / 10**(snr_db / 20))
    actual = 20 * np.log10(rms(signal) / rms(scaled))
    if abs(actual - snr_db) > 1e-4:
        raise ValueError("SNR calibration failed")
    return np.asarray(signal + scaled, dtype=np.float32), float(actual)


def reverberate(signal, rir, dry_to_wet_db=6.0):
    wet = fftconvolve(signal, rir)[:len(signal)].astype(np.float32)
    if rms(wet) <= 1e-8:
        raise ValueError("RIR produced a silent wet component")
    wet *= rms(signal) / rms(wet) / 10**(dry_to_wet_db / 20)
    return np.asarray(signal + wet, dtype=np.float32)


def shared_peak_gain(variants, headroom=.98):
    peak = max(float(np.max(np.abs(wav))) for wav in variants.values())
    return min(1., headroom / max(peak, 1e-12))


def decode_speech_audio_bytes(data, max_seconds=15):
    # Pipe-encoded FLAC can omit total samples. libsndfile then reports an
    # unbounded frame count and cannot reliably seek even after a bounded read.
    with sf.SoundFile(io.BytesIO(data)) as handle:
        sr = handle.samplerate
        unknown_length = handle.frames >= 2**62
        if not unknown_length:
            audio = handle.read(frames=int(max_seconds*sr)+1,dtype="float32",always_2d=True)
    if unknown_length:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required for streaming FLAC without a total frame count")
        # Preserve native sample rate; resample below using the same soxr path.
        command = [ffmpeg,"-nostdin","-v","error","-threads","1","-i","pipe:0",
                   "-t",f"{max_seconds+1/sr:.9f}","-ac","1","-f","f32le","pipe:1"]
        result = subprocess.run(command,input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                check=True,timeout=30)
        wav = np.frombuffer(result.stdout,dtype="<f4").copy()
    else:
        wav = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav,orig_sr=sr,target_sr=16000,res_type="soxr_hq")
    return np.asarray(wav,dtype=np.float32)


def reference_text(value, corpus):
    text = str(value).split("<asr_text>",1)[-1]
    if corpus == "gigaspeech":
        punctuation = {"<COMMA>":",", "<PERIOD>":".", "<QUESTIONMARK>":"?", "<EXCLAMATIONPOINT>":"!"}
        for token, replacement in punctuation.items():
            text = text.replace(token,replacement)
        text = re.sub(r"<[^>]+>"," ",text)
    return " ".join(text.split())


def recording_id(corpus, key):
    if corpus == "gigaspeech":
        return key.rsplit("_S",1)[0]
    if corpus == "wenetspeech":
        return re.sub(r"^(DEV_|TEST_NET_|TEST_MEETING_)","",key).rsplit("_S",1)[0]
    return None


def noise_family(asset):
    source = asset["source"]
    if source == "DNS-AudioSet":
        return "dns_audioset"
    if source == "DNS-Freesound":
        return "dns_freesound"
    if source == "ESC-50":
        return "esc50"
    if source == "MUSAN":
        return "musan_" + asset["category"].split("/", 1)[0]
    raise ValueError(f"Unknown noise family: {source}")


def round_robin_rank(candidates, seed, split, corpus):
    """Seeded selection across KeSpeech subdialects, or the entire other corpus."""
    buckets = defaultdict(list)
    for row in candidates:
        stratum = str(row.get("stratum") or "unknown") if corpus == "kespeech" else "all"
        buckets[stratum].append(row)
    for rows in buckets.values():
        rows.sort(key=lambda row: stable_seed(seed, split, corpus, row["key"]))
    names = sorted(buckets)
    index = 0
    while any(index < len(rows) for rows in buckets.values()):
        for name in names:
            if index < len(buckets[name]):
                yield buckets[name][index]
        index += 1


class SpeechSources:
    def __init__(self, archive_report):
        self.stack = ExitStack()
        self.archives = {}
        self.members = {}
        self.development = defaultdict(list)
        report = json.loads(Path(archive_report).read_text())
        for entry in report["archives"]:
            path = Path(entry["local_path"])
            if path.stat().st_size != entry["size_bytes"] or sha256_file(path) != entry["sha256"]:
                raise ValueError(f"Changed source archive: {path}")
            corpus = "kespeech" if entry["name"].startswith("kespeech") else entry["name"]
            archive = self.archive(path)
            members = self.members[str(path)]
            for member in members.values():
                if not member.name.endswith(".json"):
                    continue
                key = member.name[:-5]
                audio_member = next((key + ext for ext in [".flac", ".wav"] if key + ext in members), None)
                if audio_member is None:
                    continue
                meta = json.loads(archive.extractfile(member).read())
                declared = str(meta.get("split", "dev")).lower()
                if "dev" not in declared:
                    raise ValueError(f"Non-dev sample in dev archive: {key}: {declared}")
                text = meta.get("text") or meta.get("sentence") or meta.get("transcript")
                if not text and key + ".txt" in members:
                    text = archive.extractfile(members[key + ".txt"]).read().decode()
                if not text:
                    continue
                self.development[corpus].append(dict(
                    key=key, text=str(text), archive=str(path), member=audio_member,
                    duration=meta.get("duration"), stratum=meta.get("subdialect", "unknown"),
                    speaker=meta.get("client_id") or meta.get("spk") or meta.get("speaker_id"),
                    origin=dict(repo_id=entry["repo_id"], revision=entry["revision"],
                                remote_path=entry["remote_path"], archive_sha256=entry["sha256"],
                                member=audio_member, declared_split=declared),
                ))

    def archive(self, path):
        path = str(path)
        if path not in self.archives:
            archive = self.stack.enter_context(tarfile.open(path, "r:"))
            self.archives[path] = archive
            self.members[path] = {m.name: m for m in archive.getmembers() if m.isfile()}
        return self.archives[path]

    def candidates(self, split, corpus, benchmark_root):
        if split == "dev" and corpus == "wenetspeech":
            root = Path("/root/autodl-tmp/workspace/dataset/wenet-speech")
            manifest = root / "wenetspeech_eval_wav/dev/metadata.csv"
            with manifest.open() as stream:
                return [dict(key=r["utt_id"], text=r["text"], path=str(root / r["wav_path"]),
                             duration=float(r["duration"]),
                             origin=dict(manifest=str(manifest), declared_split="dev"))
                        for r in csv.DictReader(stream)]
        if split == "dev":
            return self.development[corpus]
        root = Path(benchmark_root) / CORPORA[corpus][3]
        rows = []
        for line in (root / "manifest.jsonl").open():
            row = json.loads(line)
            speaker = None
            if corpus == "commonvoice_yue":
                speaker = row.get("source_id")
            elif corpus == "kespeech":
                speaker = row["id"].split("_", 1)[0]
            rows.append(dict(
                key=row["id"], text=row["reference"],
                archive=str(root / row["shard"]), member=row["id"] + ".flac",
                duration=row["duration"], stratum=row.get("source_id") or "unknown",
                speaker=speaker,
                origin=dict(manifest=str(root / "manifest.jsonl"),
                            declared_split=row["split"], member=row["id"] + ".flac"),
            ))
        return rows

    def waveform(self, candidate):
        if "path" in candidate:
            return load_audio(candidate["path"], target_sr=16000)
        archive = self.archive(candidate["archive"])
        member = self.members[candidate["archive"]][candidate["member"]]
        return decode_speech_audio_bytes(archive.extractfile(member).read())

    def close(self):
        self.stack.close()


def check_asset_trace(allowed, trace):
    # v1 asset IDs can be shared by multiple clips of one original recording.
    # Canonical path plus identity/group must match, not asset_id alone.
    key = (trace["asset_id"], trace["path"])
    if key not in allowed or allowed[key] != trace["group_id"]:
        raise ValueError("Asset trace differs from held-out manifest")


def verify_suite(out, assets):
    out, assets = Path(out), Path(assets)
    protocol = json.loads((out/"protocol.json").read_text())
    allowed = {}
    for kind in ["noise", "rir"]:
        allowed[kind] = {}
        group_sets = {}
        for split in ["train", "dev", "test"]:
            path = assets/f"{kind}.{split}.jsonl"
            if sha256_file(path) != protocol["asset_manifest_sha256"][f"{kind}.{split}"]:
                raise ValueError("Augmentation manifest changed")
            rows, groups = {}, set()
            with path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    rows[(row["asset_id"], str((path.parent/row["path"]).resolve()))] = row["group_id"]
                    groups.add(row["group_id"])
            allowed[kind][split] = rows
            group_sets[split] = groups
        for a,b in [("train","dev"),("train","test"),("dev","test")]:
            if group_sets[a] & group_sets[b]:
                raise ValueError("Cross-split asset group leakage")
    checksums = json.loads((out/"audio_checksums.json").read_text())
    specs = json.loads((out/"dataset_map.json").read_text())
    bundles, ids = defaultdict(dict), set()
    for spec in specs.values():
        if sha256_file(spec["manifest"]) != spec["manifest_sha256"]:
            raise ValueError("Evaluation manifest changed")
        for line in Path(spec["manifest"]).open():
            row = json.loads(line)
            if row["id"] in ids:
                raise ValueError("Duplicate rendered sample id")
            ids.add(row["id"])
            if row["augmentation_split"] != row["split"] or row["split"] != spec["split"]:
                raise ValueError("Wrong augmentation split")
            if sha256_file(row["audio"]) != row["audio_sha256"] or checksums[row["audio"]] != row["audio_sha256"]:
                raise ValueError("Rendered audio changed")
            for kind in ["noise", "rir"]:
                trace = row["trace"].get(kind)
                if trace:
                    check_asset_trace(allowed[kind][row["split"]], trace)
            bundles[row["source_id"]][row["condition"]] = row
    max_snr_error, max_wet_error = 0., 0.
    speech_pcm, speakers, recordings = defaultdict(set), defaultdict(set), defaultdict(set)
    for pair, bundle in bundles.items():
        if set(bundle) != set(CONDITIONS):
            raise ValueError("Incomplete matched conditions")
        audio = {}
        ref = bundle["clean"]
        speech_pcm[ref["split"]].add(ref["speech_origin"]["source_pcm_sha256"])
        recording = ref["speech_origin"].get("recording")
        if recording:
            recordings[ref["split"]].add((ref["corpus"],recording))
        speaker = ref["speech_origin"].get("speaker")
        if speaker:
            speakers[ref["split"]].add((ref["corpus"],speaker))
        for condition,row in bundle.items():
            wav,sr = sf.read(row["audio"],dtype="float32")
            if sr != 16000 or wav.ndim != 1 or len(wav)/sr != row["duration"] or not np.isfinite(wav).all():
                raise ValueError("Invalid rendered waveform")
            if row["text"] != ref["text"] or row["shared_gain"] != ref["shared_gain"]:
                raise ValueError("Pair reference or shared gain differs")
            if np.max(np.abs(wav)) > .981:
                raise ValueError("Clipping/headroom verification failed")
            audio[condition] = wav
        for condition,row in bundle.items():
            if "snr_db" in row["trace"]:
                reference = "rir_real" if condition.startswith("rir_real_") else ("rir_simulated" if condition.startswith("rir_simulated_") else "clean")
                actual = 20*np.log10(rms(audio[reference])/rms(audio[condition]-audio[reference]))
                error = abs(actual-row["trace"]["snr_db"])
                max_snr_error = max(max_snr_error,float(error))
                if error > .1:
                    raise ValueError("Encoded SNR deviates by more than 0.1 dB")
        for condition in ["rir_real","rir_simulated"]:
            actual = 20*np.log10(rms(audio["clean"])/rms(audio[condition]-audio["clean"]))
            max_wet_error = max(max_wet_error,abs(float(actual)-6.))
            if abs(actual-6.) > .1:
                raise ValueError("Encoded dry/wet component ratio deviates")
    if (speech_pcm["dev"] & speech_pcm["test"] or speakers["dev"] & speakers["test"]
            or recordings["dev"] & recordings["test"]):
        raise ValueError("Selected speech content or speaker crosses dev/test")
    report = dict(passed=True,paired_speech_utterances=len(bundles),rendered_audio_files=len(ids),
                  dataset_manifests=len(specs),all_audio_sha256_verified=True,asset_trace_membership_verified=True,
                  dev_test_speech_pcm_overlap=0,dev_test_speaker_overlap_where_available=0,
                  dev_test_recording_overlap_where_available=0,
                  max_encoded_snr_error_db=max_snr_error,max_encoded_dry_wet_error_db=max_wet_error)
    (out/"suite_verification.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def build(args):
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "generation_summary.json").exists():
        raise ValueError("Frozen suite already exists; choose a new version")
    assets = Path(args.assets_dir).resolve()
    asset_hashes = {f"{kind}.{split}": sha256_file(assets / f"{kind}.{split}.jsonl")
                    for kind in ["noise", "rir"] for split in ["train", "dev", "test"]}
    protocol = dict(
        schema_version=1, seed=args.seed, samples_per_corpus_per_split=args.samples,
        speech_splits=["dev", "test"], corpora=CORPORA, conditions=CONDITIONS,
        noise_families=NOISE_FAMILIES, snr_db=[20, 10, 0], sample_rate=16000,
        duration_range_seconds=[2, 15], dry_to_wet_component_rms_db=6.,
        speech_target_rms=.05, sample_selection="seeded SHA-256 rank; KeSpeech subdialect round robin",
        noise_sampling="six source families balanced by utterance, frozen across all noise conditions",
        rir_sampling="real and simulated separate; simulated small/medium/large balanced",
        snr_definition="Full reference-waveform RMS / added-noise RMS; RIR+noise uses reverberant reference RMS.",
        reference_condition="Original held-out recording, not denoised; may contain native background noise.",
        peak_handling="One shared scalar gain per utterance across every condition, no hard clipping.",
        speed_or_specaugment=False, synthetic_rir_fallback=False,
        asset_trace_identity=["asset_id","canonical_path","group_id"],
        test_used_for_tuning=False, asset_manifest_sha256=asset_hashes,
    )
    (out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    speech = SpeechSources(args.archives)
    all_pcm, selected_keys = {}, set()
    dev_speakers, dev_recordings = defaultdict(set), defaultdict(set)
    selected, dataset_map, yaml_sets, files, coverage = [], {}, {}, [], {}
    rejected = Counter()
    try:
        for split in ["dev", "test"]:
            noise = NoisePool(manifest_path=str(assets / f"noise.{split}.jsonl"),
                              split=split, max_cached_files=8)
            buckets = defaultdict(list)
            for path, asset in noise._assets.items():
                buckets[noise_family(asset)].append(path)
            if set(buckets) != set(NOISE_FAMILIES):
                raise ValueError("Missing a noise source family")
            noise_pools = {}
            for family in NOISE_FAMILIES:
                pool = copy.copy(noise)
                pool.files = sorted(buckets[family])
                noise_pools[family] = pool
            reverb = RIRPool(manifest_path=str(assets / f"rir.{split}.jsonl"),
                             split=split, max_cached_files=16)
            rir_pools = {}
            for kind, probability in [("real", 1.), ("simulated", 0.)]:
                pool = copy.copy(reverb)
                pool.real_prob = probability
                rir_pools[kind] = pool
            observed_noise, observed_rir, observed_rooms = set(), set(), set()
            rir_assets_by_source, rir_groups_by_source = defaultdict(set), defaultdict(set)
            for corpus, (language, normalizer, metric, _) in CORPORA.items():
                candidates = speech.candidates(split, corpus, args.benchmark_audio_root)
                candidates = [r for r in candidates if str(r["text"]).strip()
                              and (r.get("duration") is None or 2 <= float(r["duration"]) <= 15)]
                manifests = {condition: [] for condition in CONDITIONS}
                count = 0
                strata = Counter()
                for row in round_robin_rank(candidates, args.seed, split, corpus):
                    if count >= args.samples:
                        break
                    if (corpus, row["key"]) in selected_keys:
                        rejected["same_speech_id"] += 1
                        continue
                    text = reference_text(row["text"],corpus)
                    if not text:
                        rejected["empty_reference"] += 1
                        continue
                    recording = recording_id(corpus,row["key"])
                    if split == "test" and recording and recording in dev_recordings[corpus]:
                        rejected["dev_test_recording_overlap"] += 1
                        continue
                    speaker = row.get("speaker")
                    if split == "test" and speaker and speaker in dev_speakers[corpus]:
                        rejected["dev_test_speaker_overlap"] += 1
                        continue
                    wav = speech.waveform(row)
                    if not 2 <= len(wav) / 16000 <= 15 or rms(wav) < 1e-5 or not np.isfinite(wav).all():
                        rejected["invalid_duration_or_signal"] += 1
                        continue
                    fingerprint = pcm_hash(wav)
                    if fingerprint in all_pcm:
                        rejected["duplicate_speech_pcm"] += 1
                        continue
                    all_pcm[fingerprint] = split
                    selected_keys.add((corpus, row["key"]))
                    if split == "dev" and speaker:
                        dev_speakers[corpus].add(speaker)
                    if split == "dev" and recording:
                        dev_recordings[corpus].add(recording)
                    identifier = hashlib.sha256((corpus + "/" + row["key"]).encode()).hexdigest()[:16]
                    pair_id = f"{split}/{corpus}/{identifier}"
                    wav = np.asarray(wav * (.05 / rms(wav)), dtype=np.float32)
                    rng = np.random.default_rng(stable_seed(args.seed, pair_id))
                    family = NOISE_FAMILIES[count % len(NOISE_FAMILIES)]
                    npool = noise_pools[family]
                    for retry in range(50):
                        noise_clip = npool.sample(len(wav), rng)
                        if rms(noise_clip) > 1e-6:
                            break
                    else:
                        raise ValueError("No non-silent noise window")
                    noise_trace = dict(npool.last_sample)
                    noise_trace["source_family"] = family
                    observed_noise.add(noise_trace["path"])
                    variants = {"clean": wav}
                    traces = {}
                    for snr in [20, 10, 0]:
                        name = f"noise_snr{snr}"
                        variants[name], actual = mix_at_snr(wav, noise_clip, snr)
                        traces[name] = dict(noise=noise_trace, snr_db=snr, actual_snr_db=actual)
                    for kind, rpool in rir_pools.items():
                        room_size = [.1, .5, .9][(count // len(NOISE_FAMILIES)) % 3]
                        for retry in range(50):
                            rir = rpool.sample(rng, room_size=room_size)
                            if rir is not None:
                                break
                        else:
                            raise ValueError("No valid RIR in held-out pool; synthetic fallback forbidden")
                        rir_trace = dict(rpool.last_sample)
                        asset = rpool._assets[Path(rir_trace["path"])]
                        if asset["source"] != kind:
                            raise ValueError("RIR source selection failed")
                        rir_trace.update(source=kind, category=asset["category"], room_id=asset.get("room_id"))
                        observed_rir.add(rir_trace["path"])
                        observed_rooms.add(rir_trace["group_id"])
                        rir_assets_by_source[kind].add(rir_trace["path"])
                        rir_groups_by_source[kind].add(rir_trace["group_id"])
                        ref = reverberate(wav, rir)
                        name = f"rir_{kind}"
                        variants[name] = ref
                        traces[name] = dict(rir=rir_trace, dry_to_wet_component_rms_db=6.)
                        for snr in [10, 0]:
                            name = f"rir_{kind}_noise_snr{snr}"
                            variants[name], actual = mix_at_snr(ref, noise_clip, snr)
                            traces[name] = dict(noise=noise_trace, rir=rir_trace, snr_db=snr,
                                                actual_snr_db=actual, dry_to_wet_component_rms_db=6.)
                    gain = shared_peak_gain(variants)
                    origin = dict(row["origin"], key=row["key"], speaker=speaker, recording=recording,
                                  stratum=row.get("stratum"), source_pcm_sha256=fingerprint)
                    selected.append(dict(pair_id=pair_id, split=split, corpus=corpus,
                                         origin=origin, duration=len(wav)/16000))
                    for condition, audio in variants.items():
                        target = out / "audio" / split / corpus / condition / (identifier + ".flac")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        sf.write(target, audio * gain, 16000, format="FLAC", subtype="PCM_16")
                        item = dict(id=pair_id.replace("/", "__") + "__" + condition,
                                    audio=str(target), text=text, language=language, split=split,
                                    source_id=pair_id, duration=len(wav)/16000, corpus=corpus,
                                    condition=condition, speech_origin=origin, shared_gain=gain,
                                    noise_manifest_sha256=asset_hashes[f"noise.{split}"],
                                    rir_manifest_sha256=asset_hashes[f"rir.{split}"],
                                    augmentation_split=split, trace=traces.get(condition, {}),
                                    audio_sha256=sha256_file(target))
                        manifests[condition].append(item)
                        files.append((str(target), item["audio_sha256"]))
                    strata[str(row.get("stratum", "all"))] += 1
                    count += 1
                if count != args.samples:
                    raise ValueError(f"{split}/{corpus}: only {count} eligible held-out utterances")
                for condition, rows in manifests.items():
                    name = f"{split}_{corpus}_{condition}"
                    manifest = out / "manifests" / (name + ".jsonl")
                    manifest.parent.mkdir(exist_ok=True)
                    manifest.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows))
                    yaml_sets[name] = dict(enabled=True, tier="robustness", manifest=str(manifest),
                                           language=language, normalizer=normalizer, metric=metric,
                                           split=split, source=f"{corpus} official {split} + frozen {split} augmentation")
                    dataset_map[name] = dict(split=split, corpus=corpus, condition=condition,
                                             metric=metric, language=language, manifest=str(manifest),
                                             manifest_sha256=sha256_file(manifest), n_utts=len(rows))
                print(json.dumps(dict(prepared=f"{split}/{corpus}", utterances=count,
                                      conditions=len(CONDITIONS), strata=dict(strata))), flush=True)
            coverage[split] = dict(noise_assets=len(observed_noise), rir_assets=len(observed_rir),
                                   rir_source_groups=len(observed_rooms),
                                   rir_assets_by_source={k:len(v) for k,v in rir_assets_by_source.items()},
                                   rir_groups_by_source={k:len(v) for k,v in rir_groups_by_source.items()})
    finally:
        speech.close()
    (out / "datasets.yaml").write_text(yaml.safe_dump(dict(datasets=yaml_sets), allow_unicode=True, sort_keys=False))
    (out / "dataset_map.json").write_text(json.dumps(dataset_map, ensure_ascii=False, indent=2)+"\n")
    (out / "speech_selection.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in selected))
    (out / "audio_checksums.json").write_text(json.dumps(dict(files), indent=2)+"\n")
    summary = dict(speech_utterances=len(selected), rendered_utterances=len(files), datasets=len(dataset_map),
                   rejected=dict(rejected), coverage=coverage, selected_dev_test_speech_pcm_overlap=0,
                   selected_dev_test_speaker_overlap_where_metadata_available=0,
                   bytes=sum(Path(p).stat().st_size for p, _ in files),
                   protocol_sha256=sha256_file(out/"protocol.json"),
                   dataset_config_sha256=sha256_file(out/"datasets.yaml"))
    summary["verification"] = verify_suite(out, assets)
    (out / "generation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--archives", help="Verified official dev archive report")
    parser.add_argument("--verify_only", action="store_true")
    parser.add_argument("--assets_dir", default=str(ROOT/"data/augmentation/splits/v1"))
    parser.add_argument("--benchmark_audio_root", default="/root/autodl-tmp/workspace/dataset/hm_asr_wds")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if args.verify_only:
        print(json.dumps(verify_suite(args.output_dir,args.assets_dir),indent=2))
    else:
        if not args.archives:
            parser.error("--archives is required for preparation")
        build(args)


if __name__ == "__main__":
    main()
