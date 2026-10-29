#!/usr/bin/env python3
"""Real-RIR acoustic simulation for tts_radiology (built on mega-asr-sim primitives).

mega-asr-sim synthesises reverb from an exponential-decay noise RIR; here we instead
convolve with real/simulated room impulse responses from
raw/POC_train/rir/RIRS_NOISES (60k simulated rooms + 417 measured RIRs).

Scenarios (severity m ~ U(0,1), dual-purpose with the Mega-ASR primitives):
    real_rir              convolution reverb only
    real_rir+noise        reverb + additive target-domain noise (SNR 15..-5 dB)
    real_rir+obstructed   reverb + low-pass 4500..1500 Hz (mask/phone)
    real_rir+recording    reverb + 8 kHz resample gate + white recording noise

Output: <out>/<scenario>/<stem>.wav (16 kHz PCM_16) + manifest.jsonl
        {"audio"(rel), "text", "prompt", "scenario", "severity"}
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal as sps
from tqdm import tqdm

MEGA = Path("/root/autodl-tmp/workspace/project/mega-asr-sim")
sys.path.insert(0, str(MEGA))
from primitives import add_noise, add_resample, apply_filter, change_volume  # noqa: E402
from severity import sample_severity  # noqa: E402
from simulate import NoisePool  # noqa: E402

SR = 16000
PROMPT = "Transcribe the audio into text."
SCENARIOS = ["real_rir", "real_rir+noise", "real_rir+obstructed", "real_rir+recording",
             "mild_noise", "mild_env", "mild_env+noise"]
_G = {}


class RIRPool:
    def __init__(self, rir_dir: str):
        self.files = sorted(Path(rir_dir).rglob("*.wav"))
        if not self.files:
            raise FileNotFoundError(f"no RIR wavs under {rir_dir}")
        self.cache: dict[tuple[Path, int], np.ndarray] = {}

    def sample(self, sr: int, rng: np.random.Generator) -> np.ndarray:
        p = self.files[int(rng.integers(len(self.files)))]
        key = (p, sr)
        wav = self.cache.get(key)
        if wav is None:
            wav, rsr = sf.read(str(p), dtype="float32", always_2d=True)
            wav = wav.mean(axis=1)
            if rsr != sr:
                wav = sps.resample_poly(wav, sr, rsr).astype(np.float32)
            wav = wav / (np.sqrt(np.sum(wav.astype(np.float64) ** 2)) + 1e-12)
            if len(self.cache) > 512:
                self.cache.clear()
            self.cache[key] = wav
        return wav


def add_real_rir(wav, sr, rng, rir_pool, wet_level, dry_level=1.0):
    rir = rir_pool.sample(sr, rng)
    wet = sps.fftconvolve(wav, rir, mode="full")[: len(wav)].astype(np.float32)
    wet = wet / (np.max(np.abs(wet)) + 1e-12) * (np.max(np.abs(wav)) + 1e-12)
    out = dry_level * wav + wet_level * wet
    peak = np.max(np.abs(out))
    if peak > 1.0:
        out = out / peak * 0.99
    return out.astype(np.float32)


def apply_scenario(wav, sr, scenario, m, rng, noise_pool, rir_pool):
    # Mild set: real radiology is ~27 dB SNR / -22 LUFS; Mega-ASR's -5..10 dB noise is
    # far harsher. mild_noise uses 25..5 dB, mild_env adds a 5000..3000 Hz low-pass,
    # mild_env+noise both; loudness fixed at -22 LUFS.
    if scenario.startswith("mild"):
        if "env" in scenario:
            wav = apply_filter(wav, sr, rng, cutoff_hz=5000.0 - 2000.0 * m,
                               filter_type="lowpass", repeat=1, wet=0.9)
        if "noise" in scenario:
            wav = add_noise(wav, sr, rng, noise_db=25.0 - 20.0 * m, noise_pool=noise_pool)
        return change_volume(wav, sr, rng, target_lufs=-22.0)
    wet = 0.15 + 0.55 * m
    wav = add_real_rir(wav, sr, rng, rir_pool, wet_level=wet, dry_level=1.0)
    if "noise" in scenario:
        wav = add_noise(wav, sr, rng, noise_db=15.0 - 20.0 * m, noise_pool=noise_pool)
    if "obstructed" in scenario:
        wav = apply_filter(wav, sr, rng, cutoff_hz=4500.0 - 3000.0 * m,
                           filter_type="lowpass", repeat=1 + int(round(2 * m)), wet=0.9)
    if "recording" in scenario:
        wav = add_resample(wav, sr, rng, target_sr=8000, prob=min(1.0, 1.5 * m),
                           threshold=0.4, wet=1.0)
        wav = add_noise(wav, sr, rng, noise_db=12.0, use_white_noise=True)
    wav = change_volume(wav, sr, rng, target_lufs=-23.0 - 6.0 * m)
    return wav


def _init(rir_dir: str, noise_dir: str) -> None:
    _G["rir"] = RIRPool(rir_dir)
    _G["noise"] = NoisePool(noise_dir)


def _process(task):
    idx, audio_path, text, scenario, m, seed, out_dir, prompt = task
    rng = np.random.default_rng(seed)
    wav, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SR:
        wav = sps.resample_poly(wav, SR, sr).astype(np.float32)
    out_wav = apply_scenario(wav, SR, scenario, m, rng, _G["noise"], _G["rir"])
    rel = f"{scenario}/{Path(audio_path).stem}.wav"
    dst = Path(out_dir) / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), out_wav, SR, subtype="PCM_16")
    return {"audio": rel, "text": text, "prompt": prompt,
            "scenario": scenario, "severity": round(m, 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rir-dir", required=True)
    ap.add_argument("--noise-dir", required=True)
    ap.add_argument("--scenario", default=",".join(SCENARIOS),
                    help="comma list cycling over scenarios")
    ap.add_argument("--severity", type=float, default=None)
    ap.add_argument("--mapping", default="linear")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    samples = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        sys.exit("manifest empty")
    scenarios = [s for s in args.scenario.split(",") if s]
    for s in scenarios:
        if s not in SCENARIOS:
            sys.exit(f"unknown scenario {s}; choose from {SCENARIOS}")

    tasks = []
    for idx, s in enumerate(samples):
        rng = np.random.default_rng(args.seed + idx)
        m = args.severity if args.severity is not None else sample_severity(rng, args.mapping)
        tasks.append((idx, s["audio"], s["text"], scenarios[idx % len(scenarios)],
                      m, args.seed + idx, args.out_dir, PROMPT))

    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(args.rir_dir, args.noise_dir)) as ex:
        for r in tqdm(ex.map(_process, tasks), total=len(tasks)):
            results.append(r)
    with (Path(args.out_dir) / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for r in results:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"done {len(results)} -> {args.out_dir}")


if __name__ == "__main__":
    main()