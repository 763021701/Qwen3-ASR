#!/usr/bin/env python3
# coding=utf-8
"""
Long-audio inference for Qwen3-ASR via center trusted-zone tiling.

Qwen3-ASR degrades on long audio (loops / hallucinations on out-of-domain
dictation). Instead of hunting for the perfect cut point, this script makes
imperfect cuts harmless:

  1. Tile the timeline into trusted zones (<= --trusted_zone_sec, boundaries
     preferred inside silence pauses via FunASR FSMN-VAD).
  2. Transcribe each zone with PADDED windows [zone - pad, zone + pad]
     (window stays < 50s), so boundary words are fully heard by both
     neighbours.
  3. Align each window transcript with Qwen3-ForcedAligner (word/char-level
     timestamps) and KEEP only tokens whose center falls inside the zone.
  4. Trusted zones tile the timeline exactly, so every token is claimed by at
     most one zone — stitching ambiguity is eliminated by construction.
  5. Fallback: if the detected language is not alignable (or alignment
     fails), re-transcribe the exact zone without padding.

Usage:
    python long_inference.py audio.wav --model 0.6B
    python long_inference.py audio.wav --model 1.7B --trusted_zone_sec 35 --pad_sec 5
    python long_inference.py audio.wav --model /path/to/checkpoint
    python long_inference.py audio.wav --split_policy longest_zone
"""

import argparse
import os

import numpy as np
import soundfile as sf
import torch
from funasr import AutoModel

from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner
from qwen_asr.inference.utils import SAMPLE_RATE

FSMN_VAD_MODEL = "fsmn-vad"


def load_wav_16k(path: str) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr == SAMPLE_RATE:
        return wav
    dur = wav.shape[0] / float(sr)
    n16 = int(round(dur * SAMPLE_RATE))
    x_old = np.linspace(0.0, dur, num=wav.shape[0], endpoint=False)
    x_new = np.linspace(0.0, dur, num=n16, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)


def _ms_to_sample(ms: float, total: int) -> int:
    return int(max(0, min(total, round(float(ms) / 1000.0 * SAMPLE_RATE))))


def silence_from_speech_ms(speech_ms, total: int, min_silence_sec: float):
    """Gaps between FSMN-VAD speech segments, as sample ranges >= min_silence_sec."""
    min_n = int(min_silence_sec * SAMPLE_RATE)
    spans = sorted(
        (_ms_to_sample(s, total), _ms_to_sample(e, total))
        for s, e in speech_ms
        if e > s
    )
    runs = []
    cursor = 0
    for s, e in spans:
        if s > cursor and s - cursor >= min_n:
            runs.append((cursor, s))
        cursor = max(cursor, e)
    if total > cursor and total - cursor >= min_n:
        runs.append((cursor, total))
    return runs


def find_silence_runs_fsmn(audio_path: str, total: int, min_silence_sec: float):
    print(f"Loading FSMN-VAD: {FSMN_VAD_MODEL} (device: cpu)")
    vad = AutoModel(model=FSMN_VAD_MODEL, device="cpu", disable_update=True)
    result = vad.generate(input=audio_path)
    del vad
    speech_ms = result[0].get("value") or [] if result else []
    runs = silence_from_speech_ms(speech_ms, total, min_silence_sec)
    print(f"FSMN-VAD: {len(speech_ms)} speech segment(s), "
          f"{len(runs)} silence run(s) >= {min_silence_sec}s")
    return runs


def _pick_cut_run(candidates, policy: str):
    """Pick a silence run to cut at (center of the run)."""
    if policy == "longest_zone":
        return max(candidates, key=lambda r: (r[0] + r[1]) // 2)
    return max(candidates, key=lambda r: r[1] - r[0])


def split_at_silence(total: int, silences, max_sec: float, min_sec: float,
                     min_silence_sec: float, policy: str = "greedy"):
    """Split into zones <= max_sec, cutting inside a silence run in (min_sec, max_sec].

    greedy: cut at the longest pause (may yield short zones).
    longest_zone: cut at the latest pause, so each zone is as long as possible.
    Hard-cut when no silence is available — pads absorb the boundary.
    """
    max_n, min_n = int(max_sec * SAMPLE_RATE), int(min_sec * SAMPLE_RATE)
    if total <= max_n:
        return [(0, total)]

    cuts = []
    start = 0
    while total - start > max_n:
        lo, hi = start + min_n, start + max_n
        candidates = [r for r in silences if lo < (r[0] + r[1]) // 2 <= hi]
        if candidates:
            run = _pick_cut_run(candidates, policy)
            cut = (run[0] + run[1]) // 2
        else:
            cut = hi
            print(f"[warn] no silence >= {min_silence_sec}s in "
                  f"[{start / SAMPLE_RATE:.1f}s, {hi / SAMPLE_RATE:.1f}s]; zone cut at "
                  f"{cut / SAMPLE_RATE:.1f}s (pads will absorb the boundary)")
        cuts.append(cut)
        start = cut
    bounds = [0] + cuts + [total]
    return list(zip(bounds[:-1], bounds[1:]))


def _align_char_spans(transcript: str, items, is_kept):
    """Map each aligner item to its (start, end) char offsets in the transcript.

    Aligner items concatenate (in order) to a subsequence of the transcript's
    kept chars (letters/digits/apostrophe; punctuation is dropped by the
    aligner). Returns None if the mapping fails (caller falls back).
    """
    spans = []
    pos = 0
    for it in items:
        w = it.text
        while pos < len(transcript) and not is_kept(transcript[pos]):
            pos += 1
        start = pos
        wi = 0
        while wi < len(w) and pos < len(transcript):
            ch = transcript[pos]
            if is_kept(ch):
                if ch != w[wi]:
                    return None
                wi += 1
            pos += 1
        if wi < len(w):
            return None
        spans.append((start, pos))
    return spans


def _trim_to_zone(transcript: str, items, wa: int, zs: int, ze: int, is_kept):
    """Keep transcript text whose token centers fall inside trusted zone [zs, ze).

    Token times from the aligner are window-relative (window starts at `wa`).
    The text is cut in char-space: leading/trailing punctuation between the
    kept token span and its neighbours follows the zone that owns the words.
    Returns None when nothing can be attributed (caller falls back).
    """
    if not items:
        return None
    spans = _align_char_spans(transcript, items, is_kept)
    if spans is None:
        return None
    keep = [i for i, it in enumerate(items)
            if zs <= wa + (it.start_time + it.end_time) / 2.0 * SAMPLE_RATE < ze]
    if not keep:
        return None
    first, last = keep[0], keep[-1]
    end_char = spans[last + 1][0] if last + 1 < len(items) else len(transcript)
    return transcript[spans[first][0]:end_char].strip()


def _smart_join(a: str, b: str) -> str:
    """Join zone texts; insert a space before each subsequent non-empty zone."""
    if not a:
        return b
    if not b:
        return a
    return a.rstrip() + " " + b.lstrip()


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR long-audio inference (center trusted-zone tiling)")
    parser.add_argument("audio", type=str, help="Path to audio file")
    parser.add_argument("--model", type=str, default="1.7B",
                        help="Model: '0.6B', '1.7B', or local path (default: 1.7B)")
    parser.add_argument("--aligner", type=str, default="Qwen/Qwen3-ForcedAligner-0.6B",
                        help="ForcedAligner model for trusted-zone token timestamps")
    parser.add_argument("--language", type=str, default=None, help="Force language (e.g. 'Cantonese')")
    parser.add_argument("--context", type=str, default="", help="Context string (hotwords)")
    parser.add_argument("--trusted_zone_sec", type=float, default=35.0,
                        help="Trusted-zone upper bound in seconds (default: 35)")
    parser.add_argument("--pad_sec", type=float, default=5.0,
                        help="Padding on each side of a zone when building the transcribed window")
    parser.add_argument("--min_zone_sec", type=float, default=10.0,
                        help="Zone lower bound in seconds (default: 10)")
    parser.add_argument("--min_silence_sec", type=float, default=0.4,
                        help="Minimum pause length to be a zone-cut candidate (default: 0.4)")
    parser.add_argument("--split_policy", type=str, default="longest_zone",
                        choices=["greedy", "longest_zone"],
                        help="Zone cut policy: greedy=longest pause in window; "
                             "longest_zone=latest pause (zones as long as trusted_zone_sec allows)")
    args = parser.parse_args()

    if args.trusted_zone_sec + 2 * args.pad_sec >= 50:
        raise SystemExit("trusted_zone_sec + 2*pad_sec must stay < 50 "
                         "(Qwen3-ASR long-audio safety bound)")

    wav = load_wav_16k(args.audio)
    total = wav.shape[0]
    silences = find_silence_runs_fsmn(args.audio, total, args.min_silence_sec)
    zones = split_at_silence(total, silences, args.trusted_zone_sec, args.min_zone_sec,
                             args.min_silence_sec, args.split_policy)
    print(f"Audio: {total / SAMPLE_RATE:.1f}s -> {len(zones)} trusted zone(s) "
          f"(split_policy={args.split_policy})\n")

    if os.path.exists(args.model) or "/" in args.model or "\\" in args.model:
        model_name = args.model
    else:
        model_name = f"Qwen/Qwen3-ASR-{args.model}"
    print(f"Loading ASR model: {model_name}")
    model = Qwen3ASRModel.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda:0", max_new_tokens=1024)

    print(f"Loading aligner: {args.aligner}")
    aligner = Qwen3ForcedAligner.from_pretrained(args.aligner, dtype=torch.bfloat16, device_map="cuda:0")
    is_kept = aligner.aligner_processor.is_kept_char
    supported = {str(l).lower(): str(l) for l in (aligner.get_supported_languages() or [])}

    def transcribe(audio_tuple):
        return model.transcribe(audio=audio_tuple, context=args.context,
                                language=args.language, return_time_stamps=False)[0]

    texts = []
    for i, (zs, ze) in enumerate(zones, 1):
        wa, wb = max(0, int(zs - args.pad_sec * SAMPLE_RATE)), min(total, int(ze + args.pad_sec * SAMPLE_RATE))
        result = transcribe((wav[wa:wb], SAMPLE_RATE))

        zone_text, note = None, ""
        if result.text:
            lang_key = str(result.language or "").lower()
            if lang_key not in supported:
                note = f"FALLBACK (aligner language '{result.language}' unsupported)"
            else:
                try:
                    alignment = aligner.align(audio=(wav[wa:wb], SAMPLE_RATE),
                                              text=result.text,
                                              language=supported[lang_key])[0]
                    zone_text = _trim_to_zone(result.text, alignment.items, wa, zs, ze, is_kept)
                    if zone_text is None:
                        note = "FALLBACK (no tokens attributed to zone)"
                    else:
                        note = "kept tokens within zone"
                except Exception as exc:  # noqa: BLE001
                    note = f"FALLBACK (alignment failed: {type(exc).__name__})"
            if zone_text is None:
                result = transcribe((wav[zs:ze], SAMPLE_RATE))
                zone_text = result.text
        else:
            zone_text, note = "", "empty transcript"

        texts.append(zone_text)
        print(f"[zone {i}/{len(zones)} {zs / SAMPLE_RATE:7.1f}-{ze / SAMPLE_RATE:7.1f}s | "
              f"window {wa / SAMPLE_RATE:6.1f}-{wb / SAMPLE_RATE:6.1f}s] language={result.language!r} {note}")
        print(f"  {zone_text}")

    full = ""
    for t in texts:
        full = _smart_join(full, t)

    print("\n" + "=" * 60)
    print("Full text:")
    print(full)
    print("=" * 60)


if __name__ == "__main__":
    main()
