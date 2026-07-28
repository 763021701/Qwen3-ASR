#!/usr/bin/env python3
# coding=utf-8
"""
Prepare MAGICDATA-Mandarin-Read for Qwen3-ASR SFT.

Layout (per AGENTS/TRANS):
    {dataset_dir}/{split}/TRANS.txt   # tab: UtteranceID\tSpeakerID\tTranscription
    {dataset_dir}/{split}/{SpeakerID}/{UtteranceID}   # .wav

Outputs Qwen3 jsonl: {"audio": <abs path>, "text": "language Chinese<asr_text>{transcript}"}

Transcript normalization uses the same masr ``zh`` (t2s) normalizer the eval scripts
use, so train targets and eval references stay aligned. A small inline fallback
mirrors normalize_transcript.normalize_cantonese if masr is unavailable.

Train split is downsampled by cumulative-duration greedy sampling (fixed seed) to a
target hour budget so it balances against the Uyghur (~205h) set.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import wave
from typing import List, Optional, Tuple

_ASR_TEXT_TAG = "<asr_text>"


def _normalize_mandarin(raw: str) -> str:
    body = (raw or "").strip()
    if not body:
        return ""
    try:
        from masr_eval_pkg.normalizers import get_normalizer

        norm = get_normalizer("zh", zh_convert="t2s")
        out = norm.normalize(body)
    except Exception:
        # Fallback: NFKC + fullwidth->halfwidth + drop unicode punctuation + keep
        # CJK/ascii-alnum, matching normalize_transcript.normalize_cantonese.
        import re
        import unicodedata

        s = unicodedata.normalize("NFC", body)
        out_chars = []
        for ch in s:
            o = ord(ch)
            if 0xFF01 <= o <= 0xFF5E:
                ch = chr(o - 0xFEE0)
            elif o == 0x3000:
                ch = " "
            if unicodedata.category(ch).startswith("P"):
                ch = " "
            out_chars.append(ch)
        out = "".join(out_chars)
        kept = []
        for ch in out:
            if ch.isspace():
                kept.append(" ")
            elif ch.isdigit() or (ch.isascii() and ch.isalpha()):
                kept.append(ch.lower())
            else:
                kept.append(ch)
        out = "".join(kept)
    # Remove spaces between CJK characters (keep spaces around Latin/digits).
    import re

    out = re.sub(r"(?<=[一-鿿㐀-䶿])\s+(?=[一-鿿㐀-䶿])", "", out)
    out = " ".join(out.split())
    return out


def _wav_duration(path: str) -> Optional[float]:
    """Read WAV header only (no decode). Returns seconds or None on failure."""
    try:
        with wave.open(path, "rb") as w:
            n = w.getnframes()
            sr = w.getframerate()
            if sr <= 0:
                return None
            return n / float(sr)
    except Exception:
        return None


def _read_trans(trans_path: str) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    with open(trans_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            utt_id, speaker_id, transcript = parts[0], parts[1], "\t".join(parts[2:])
            rows.append((utt_id, speaker_id, transcript))
    return rows


def _build_split(
    split: str,
    dataset_dir: str,
    out_path: str,
    language: str,
    target_hours: Optional[float],
    seed: int,
    max_samples: int,
) -> Tuple[int, float, int]:
    trans_path = os.path.join(dataset_dir, split, "TRANS.txt")
    rows = _read_trans(trans_path)

    records = []  # (audio_path, text, dur_seconds)
    missing = 0
    empty = 0
    for utt_id, speaker_id, transcript in rows:
        audio_path = os.path.join(dataset_dir, split, speaker_id, utt_id)
        if not os.path.isfile(audio_path):
            missing += 1
            continue
        norm = _normalize_mandarin(transcript)
        if not norm:
            empty += 1
            continue
        dur = _wav_duration(audio_path) or 0.0
        text = "language %s%s%s" % (language, _ASR_TEXT_TAG, norm)
        records.append((audio_path, text, dur))

    if target_hours is not None and target_hours > 0:
        target_sec = target_hours * 3600.0
        rng = random.Random(seed)
        idx = list(range(len(records)))
        rng.shuffle(idx)
        kept = []
        total = 0.0
        for i in idx:
            kept.append(records[i])
            total += records[i][2]
            if total >= target_sec:
                break
        records = kept
    elif max_samples > 0:
        records = records[:max_samples]

    with open(out_path, "w", encoding="utf-8") as wf:
        for audio_path, text, _ in records:
            wf.write(json.dumps({"audio": audio_path, "text": text}, ensure_ascii=False) + "\n")

    total_dur = sum(r[2] for r in records)
    print(
        "[magicdata] split=%s wrote=%d (missing_audio=%d empty_after_norm=%d) dur=%.1fs=%.2fh"
        % (split, len(records), missing, empty, total_dur, total_dur / 3600.0)
    )
    return len(records), total_dur, missing


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare MAGICDATA Mandarin for Qwen3-ASR SFT.")
    p.add_argument("--dataset_dir", required=True, help="MAGICDATA-Mandarin-Read root.")
    p.add_argument("--output_dir", required=True, help="Where to write jsonl files.")
    p.add_argument("--language", default="Chinese", help="LanguageSpec atom for the prefix.")
    p.add_argument("--train_target_hours", type=float, default=200.0,
                   help="Downsample train to ~this many hours (0 = keep all).")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max_samples", type=int, default=0, help="Cap each split (0 = all, applied after downsample).")
    p.add_argument("--report_json", default="", help="Optional path to write a summary json.")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    summary = {}
    for split, target in [("train", args.train_target_hours), ("dev", None), ("test", None)]:
        out_path = os.path.join(args.output_dir, "%s.jsonl" % split)
        n, dur, missing = _build_split(
            split, args.dataset_dir, out_path, args.language,
            target if target and target > 0 else None,
            args.seed, args.max_samples,
        )
        summary[split] = {"path": out_path, "count": n, "seconds": dur, "hours": dur / 3600.0, "missing_audio": missing}

    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print("[magicdata] report -> %s" % args.report_json)


if __name__ == "__main__":
    main()
