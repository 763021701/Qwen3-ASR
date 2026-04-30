#!/usr/bin/env python3
# coding=utf-8
"""
Evaluate FireRedASR2 (FireRedASR2S stack) on a Qwen3-style jsonl manifest.

Use the same manifest format and scoring as `evaluation/cantonese/eval_cantonese_asr_jsonl.py`:
  - Each line: JSON with `audio` (path) and `text` (Qwen3 label, optional `language ...<asr_text>` prefix).
  - Metrics: corpus CER (jiwer if available) and sentence exact-match rate on normalized text.
  - Normalization: NFC, ZWSP strip, Hanzi OpenCC, lowercasing, cn2an ITN, Unicode punctuation removal,
    whitespace handling per flags.

FireRedASR2 supports Mandarin and many Chinese dialects/accents (including Cantonese and Sichuan) in one
model; pass any compatible jsonl (e.g. Cantonese or Sichuan benchmarks) with the same schema.

Setup (from FireRedASR2S README):
  - pip install -r refs/FireRedASR2S/requirements.txt
  - Download weights to e.g. ./pretrained_models/FireRedASR2-AED (or FireRedASR2-LLM)
  - This script adds `refs/FireRedASR2S` to sys.path so `fireredasr2s` imports work from the Qwen3-ASR tree.

Audio:
  - Official FireRed expects 16 kHz mono PCM. `audio_mode=librosa` resamples/converts to int16 for mp3/flac.
  - `audio_mode=path` forwards paths to FireRed (kaldiio load; best for 16 kHz wav), unless long-audio chunking is on
    (then audio is loaded in memory for splitting).
  - Utterances are padded to at least `min_samples_16k` samples per chunk to avoid empty filterbank rows on very short clips.
  - FireRedASR2-LLM supports about up to 30s per forward pass; by default (`--max_segment_seconds` auto) this script splits
    longer files into non-overlapping <=30s chunks, transcribes each, and concatenates hypotheses for scoring.

Example (Cantonese jsonl):
  python evaluation/cantonese/baselines/eval_cantonese_asr_fireredasr2s_jsonl.py \\
    --jsonl data/cantonese/common_voice_yue/cv_yue_test_qwen3.jsonl \\
    --model_dir pretrained_models/FireRedASR2-AED \\
    --asr_type aed \\
    --batch_size 4 \\
    --output_predictions outputs/firered_aed/cv_yue_predictions.jsonl

Example (Sichuan / other dialect, same jsonl schema):
  python evaluation/cantonese/baselines/eval_cantonese_asr_fireredasr2s_jsonl.py \\
    --jsonl data/your_chuan_manifest.jsonl \\
    --model_dir pretrained_models/FireRedASR2-AED \\
    --asr_type aed
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_FIRERED_REPO = os.path.join(_REPO_ROOT, "refs", "FireRedASR2S")

_ASR_TEXT_TAG = "<asr_text>"
_ZW_RE = re.compile(r"[\u200c\u200d\ufeff]")
_CN_NUMERAL_RE = re.compile(r"[零〇一二两三四五六七八九十百千万億亿壹贰叁肆伍陆柒捌玖拾佰仟萬廿卅]+")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FireRedASR2 on Qwen3 jsonl: CER + sentence accuracy (same scoring as eval_cantonese_asr_jsonl)."
    )
    p.add_argument("--jsonl", type=str, required=True, help="Manifest: lines with audio + text (Qwen3 label).")
    p.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory with FireRedASR2-AED or FireRedASR2-LLM weights (model.pth.tar, cmvn.ark, ...).",
    )
    p.add_argument("--asr_type", type=str, default="aed", choices=("aed", "llm"), help="FireRedASR2 backend.")
    p.add_argument(
        "--firered_repo",
        type=str,
        default=_DEFAULT_FIRERED_REPO,
        help="Path to FireRedASR2S repo root (contains fireredasr2s/). Used for sys.path only.",
    )
    p.add_argument("--max_samples", type=int, default=0, help="If >0, only first N rows.")
    p.add_argument("--batch_size", type=int, default=4, help="ASR batch size (see FireRed LLM notes on length mismatch).")
    p.add_argument("--use_gpu", type=int, default=1, choices=(0, 1), help="1=cuda ASR.")
    p.add_argument("--use_half", type=int, default=0, choices=(0, 1), help="1=FP16 weights on GPU.")
    p.add_argument("--beam_size", type=int, default=3)
    p.add_argument("--nbest", type=int, default=1)
    p.add_argument("--decode_max_len", type=int, default=0, help="0 = model default / unlimited where supported.")
    p.add_argument("--softmax_smoothing", type=float, default=1.25)
    p.add_argument("--aed_length_penalty", type=float, default=0.6)
    p.add_argument("--eos_penalty", type=float, default=1.0)
    p.add_argument("--return_timestamp", type=int, default=0, choices=(0, 1))
    p.add_argument("--decode_min_len", type=int, default=0)
    p.add_argument("--repetition_penalty", type=float, default=3.0)
    p.add_argument("--llm_length_penalty", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--elm_dir", type=str, default="", help="Optional external LM directory (AED).")
    p.add_argument("--elm_weight", type=float, default=0.0)
    p.add_argument(
        "--output_predictions",
        type=str,
        default="",
        help="If set, write jsonl with ref/hyp/errors per line (same fields as eval_cantonese_asr_jsonl).",
    )
    p.add_argument(
        "--keep_whitespace",
        action="store_true",
        help="Keep whitespace for CER. Default is to remove all whitespace before scoring.",
    )
    p.add_argument(
        "--hanzi_script_norm",
        type=str,
        default="to_traditional",
        choices=("off", "to_traditional", "to_simplified"),
        help="Normalize Hanzi script before scoring. Default: convert both ref/hyp to Traditional Chinese.",
    )
    p.add_argument(
        "--audio_mode",
        type=str,
        default="librosa",
        choices=("path", "librosa"),
        help="path=native file load via FireRed (16 kHz wav recommended). librosa=resample to 16 kHz mono int16.",
    )
    p.add_argument(
        "--min_samples_16k",
        type=int,
        default=16000,
        help="Minimum waveform length at 16 kHz (pad with zeros) before feature extraction.",
    )
    p.add_argument(
        "--eval_label",
        type=str,
        default="Chinese dialect ASR",
        help="Short label printed in the metrics header (e.g. Cantonese, Sichuan).",
    )
    p.add_argument(
        "--max_segment_seconds",
        type=float,
        default=-1.0,
        help="If >0, split each utterance into fixed windows of at most this duration (seconds) before ASR, "
        "then concatenate chunk transcripts. Default -1: auto — 30 for asr_type=llm (FireRedASR2-LLM limit), "
        "0 (disabled) for aed. Pass 0 to force no splitting even for llm.",
    )
    return p.parse_args()


def extract_reference_text(label: str) -> str:
    s = (label or "").strip()
    if not s:
        return ""
    if _ASR_TEXT_TAG in s:
        return s.split(_ASR_TEXT_TAG, 1)[1].strip()
    return s


def build_hanzi_script_converter(mode: str):
    key = (mode or "off").strip().lower()
    if key == "off":
        return None
    config_map = {
        "to_traditional": "s2t",
        "to_simplified": "t2s",
    }
    try:
        from opencc import OpenCC  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise SystemExit(
            "Install OpenCC first for Hanzi script normalization: pip install opencc-python-reimplemented"
        ) from exc
    cc = OpenCC(config_map[key])
    return cc.convert


def remove_unicode_punctuation(s: str) -> str:
    return "".join(ch for ch in s if not unicodedata.category(ch).startswith("P"))


def apply_number_itn(s: str) -> str:
    try:
        import cn2an  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise SystemExit("Install cn2an first for number ITN: pip install cn2an") from exc

    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        try:
            return str(cn2an.cn2an(token, "smart"))
        except Exception:
            return token

    return _CN_NUMERAL_RE.sub(repl, s)


def normalize_for_scoring(s: str, *, keep_whitespace: bool, hanzi_script_converter) -> str:
    s = unicodedata.normalize("NFC", (s or "").strip())
    s = _ZW_RE.sub("", s)
    if hanzi_script_converter is not None:
        s = hanzi_script_converter(s)
    s = s.lower()
    s = apply_number_itn(s)
    s = remove_unicode_punctuation(s)
    if keep_whitespace:
        s = " ".join(s.split())
    else:
        s = "".join(s.split())
    return s


def _levenshtein_1d(a: Sequence[Any], b: Sequence[Any]) -> int:
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def corpus_cer_fallback(refs: List[str], hyps: List[str]) -> float:
    c_err, c_den = 0, 0
    for r, h in zip(refs, hyps):
        rc = list(r)
        hc = list(h)
        if len(rc) == 0 and len(hc) == 0:
            continue
        if len(rc) == 0:
            c_err += len(hc)
            c_den += max(len(hc), 1)
        else:
            c_err += _levenshtein_1d(rc, hc)
            c_den += len(rc)
    return c_err / max(c_den, 1)


def corpus_cer_jiwer(refs: List[str], hyps: List[str]) -> float:
    import jiwer  # pyright: ignore[reportMissingImports]

    return float(jiwer.cer(refs, hyps))


def load_manifest(path: str, max_samples: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def _float_to_int16(wav: np.ndarray) -> np.ndarray:
    x = np.clip(np.rint(wav.astype(np.float64) * 32768.0), -32768, 32767)
    return x.astype(np.int16)


def load_audio_16k_int16(path: str, min_samples_16k: int) -> Tuple[int, np.ndarray]:
    import librosa  # pyright: ignore[reportMissingImports]

    floor = max(16000, int(min_samples_16k))
    wav, _ = librosa.load(path, sr=16000, mono=True)
    if wav.size == 0:
        wav = np.zeros(floor, dtype=np.float32)
    elif int(wav.shape[0]) < floor:
        wav = np.pad(wav, (0, floor - int(wav.shape[0])), mode="constant")
    return 16000, _float_to_int16(wav)


def load_audio_16k_int16_no_min_pad(path: str) -> np.ndarray:
    """Full utterance at 16 kHz mono int16 (no min-length pad); used before windowing into chunks."""
    import librosa  # pyright: ignore[reportMissingImports]

    wav, _ = librosa.load(path, sr=16000, mono=True)
    if wav.size == 0:
        return np.zeros(0, dtype=np.int16)
    return _float_to_int16(wav)


def _resolve_max_segment_seconds(asr_type: str, flag: float) -> float:
    if flag < 0:
        return 30.0 if asr_type == "llm" else 0.0
    return float(flag)


def _pad_min_int16(wav_i16: np.ndarray, min_samples_16k: int) -> np.ndarray:
    floor = max(16000, int(min_samples_16k))
    n = int(wav_i16.shape[0]) if wav_i16.size else 0
    if n >= floor:
        return wav_i16
    pad_n = floor - n
    pad = np.zeros(pad_n, dtype=np.int16)
    if n == 0:
        return pad
    return np.concatenate([wav_i16, pad], axis=0)


def split_int16_waveform_into_chunks(
    wav_i16: np.ndarray,
    sample_rate: int,
    max_segment_seconds: float,
    min_samples_16k: int,
) -> List[np.ndarray]:
    """
    Non-overlapping windows of at most max_segment_seconds; each chunk is padded to min_samples_16k if shorter.
    If max_segment_seconds <= 0, returns a single chunk for the whole waveform.
    """
    if max_segment_seconds <= 0:
        return [_pad_min_int16(wav_i16, min_samples_16k)]

    max_samples = int(round(max_segment_seconds * float(sample_rate)))
    max_samples = max(max_samples, 1)
    n = int(wav_i16.shape[0]) if wav_i16.size else 0
    if n <= max_samples:
        return [_pad_min_int16(wav_i16, min_samples_16k)]

    chunks: List[np.ndarray] = []
    for start in range(0, n, max_samples):
        seg = wav_i16[start : start + max_samples]
        chunks.append(_pad_min_int16(seg, min_samples_16k))
    return chunks


def transcribe_audio_path_with_optional_chunks(
    model: Any,
    audio_path: str,
    *,
    audio_mode: str,
    min_samples_16k: int,
    max_segment_seconds: float,
    uttid_prefix: str,
) -> str:
    """
    One manifest utterance -> one hypothesis string. Uses chunking when max_segment_seconds > 0.
    Chunking always loads via librosa; otherwise respects audio_mode.
    """
    if max_segment_seconds <= 0:
        uttids = [uttid_prefix]
        uttids, batch_inp = build_asr_batch_inputs(
            [audio_path], audio_mode, min_samples_16k, uttids
        )
        results = model.transcribe(uttids, batch_inp)
        return _hypotheses_from_results(uttids, results)[0]

    wav_i16 = load_audio_16k_int16_no_min_pad(audio_path)
    chunks = split_int16_waveform_into_chunks(wav_i16, 16000, max_segment_seconds, min_samples_16k)
    parts: List[str] = []
    for k, seg in enumerate(chunks):
        chunk_uttid = f"{uttid_prefix}_c{k:03d}"
        results = model.transcribe([chunk_uttid], [(16000, seg)])
        parts.append(_hypotheses_from_results([chunk_uttid], results)[0])
    return "".join(p for p in parts)


def build_asr_batch_inputs(
    paths: List[str], mode: str, min_samples_16k: int, uttids: List[str]
) -> Tuple[List[str], Union[List[str], List[Tuple[int, np.ndarray]]]]:
    if mode == "path":
        return uttids, paths
    wav_pairs = [load_audio_16k_int16(p, min_samples_16k) for p in paths]
    return uttids, wav_pairs


def _hypotheses_from_results(batch_uttids: List[str], results: List[Dict[str, Any]]) -> List[str]:
    """Map FireRed `transcribe` output to the batch order; missing uttids -> empty string."""
    by_utt = {str(r.get("uttid", "")): r for r in results}
    out: List[str] = []
    for u in batch_uttids:
        r = by_utt.get(u)
        if not r:
            out.append("")
        else:
            t = r.get("text", "")
            out.append((t or "").strip())
    return out


def main() -> None:
    args = parse_args()
    repo = os.path.abspath(args.firered_repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    rows = load_manifest(args.jsonl, args.max_samples)
    if not rows:
        print("No samples loaded.", file=sys.stderr)
        sys.exit(1)

    for i, ex in enumerate(rows):
        if "audio" not in ex or "text" not in ex:
            print(f"Line {i}: need 'audio' and 'text' fields.", file=sys.stderr)
            sys.exit(1)
        ap = ex["audio"]
        if not os.path.isfile(ap):
            print(f"Missing audio file: {ap}", file=sys.stderr)
            sys.exit(1)

    if not os.path.isdir(args.model_dir):
        print(f"model_dir not found: {args.model_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config  # type: ignore[import-not-found]
    except ImportError as exc:
        print(
            "Failed to import fireredasr2s. Install FireRedASR2S deps and ensure --firered_repo points to the repo root.\n"
            f"  Tried repo path: {repo!r}\n"
            "  See refs/FireRedASR2S/README.md (pip install -r requirements.txt).",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    asr_config = FireRedAsr2Config(
        use_gpu=bool(args.use_gpu),
        use_half=bool(args.use_half),
        beam_size=int(args.beam_size),
        nbest=int(args.nbest),
        decode_max_len=int(args.decode_max_len),
        softmax_smoothing=float(args.softmax_smoothing),
        aed_length_penalty=float(args.aed_length_penalty),
        eos_penalty=float(args.eos_penalty),
        return_timestamp=bool(args.return_timestamp),
        decode_min_len=int(args.decode_min_len),
        repetition_penalty=float(args.repetition_penalty),
        llm_length_penalty=float(args.llm_length_penalty),
        temperature=float(args.temperature),
        elm_dir=str(args.elm_dir or ""),
        elm_weight=float(args.elm_weight),
    )

    print(f"Loading FireRedASR2 ({args.asr_type}) from {args.model_dir!r} ...")
    model = FireRedAsr2.from_pretrained(args.asr_type, args.model_dir, asr_config)

    refs_raw = [extract_reference_text(ex["text"]) for ex in rows]
    audios = [ex["audio"] for ex in rows]
    min_s = max(0, int(args.min_samples_16k))
    max_seg = _resolve_max_segment_seconds(args.asr_type, float(args.max_segment_seconds))

    print(
        f"Transcribing {len(rows)} utterances (asr_type={args.asr_type!r}, audio_mode={args.audio_mode!r}, "
        f"min_samples_16k={min_s}, max_segment_seconds={max_seg}, batch_size={args.batch_size}) ..."
    )
    if max_seg > 0:
        print(
            f"Long-audio chunking is on (windows <= {max_seg}s, non-overlapping); "
            "each utterance is decoded sequentially. Audio is loaded with librosa for splitting.",
            flush=True,
        )

    predictions: List[str] = []
    if max_seg > 0:
        for i, path in enumerate(audios):
            uttid_prefix = f"eval_{i:09d}"
            predictions.append(
                transcribe_audio_path_with_optional_chunks(
                    model,
                    path,
                    audio_mode=args.audio_mode,
                    min_samples_16k=min_s,
                    max_segment_seconds=max_seg,
                    uttid_prefix=uttid_prefix,
                )
            )
    else:
        for start in range(0, len(audios), args.batch_size):
            batch_paths = audios[start : start + args.batch_size]
            uttids = [f"eval_{start + j:09d}" for j in range(len(batch_paths))]
            uttids, batch_inp = build_asr_batch_inputs(batch_paths, args.audio_mode, min_s, uttids)
            results = model.transcribe(uttids, batch_inp)
            predictions.extend(_hypotheses_from_results(uttids, results))

    if len(predictions) != len(refs_raw):
        print("Internal error: prediction count mismatch.", file=sys.stderr)
        sys.exit(1)

    hanzi_script_converter = build_hanzi_script_converter(args.hanzi_script_norm)
    refs = [
        normalize_for_scoring(r, keep_whitespace=args.keep_whitespace, hanzi_script_converter=hanzi_script_converter)
        for r in refs_raw
    ]
    hyps = [
        normalize_for_scoring(h, keep_whitespace=args.keep_whitespace, hanzi_script_converter=hanzi_script_converter)
        for h in predictions
    ]

    try:
        cer = corpus_cer_jiwer(refs, hyps)
        backend = "jiwer"
    except ImportError:
        cer = corpus_cer_fallback(refs, hyps)
        backend = "builtin_levenshtein"

    exact_matches = sum(1 for r, h in zip(refs, hyps) if r == h)
    sentence_accuracy = exact_matches / max(len(refs), 1)

    print("")
    print(f"=== {args.eval_label} (FireRedASR2) ===")
    print(f"Model dir:          {args.model_dir}")
    print(f"ASR type:           {args.asr_type}")
    print(f"Max segment (s):  {max_seg}")
    print(f"Samples:            {len(rows)}")
    print(
        "Scoring:            "
        f"NFC, ZWSP removed, English lowercased, number ITN, Unicode punctuation removed, "
        f"Hanzi={args.hanzi_script_norm}, whitespace="
        f"{'kept' if args.keep_whitespace else 'removed'}"
    )
    print(f"Backend:            {backend}")
    print(f"CER:                {cer * 100:.2f}%")
    print(f"Sentence Accuracy:  {sentence_accuracy * 100:.2f}%")
    print("")
    print("Note: For Han-character dialect transcripts, CER is the primary metric.")
    print("      WER based on whitespace tokenization is usually not meaningful and is omitted.")

    if args.output_predictions:
        out_path = args.output_predictions
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as wf:
            for ex, rr, hh, pr in zip(rows, refs, hyps, predictions):
                dist = _levenshtein_1d(list(rr), list(hh))
                rec = {
                    "audio": ex["audio"],
                    "reference_raw": extract_reference_text(ex["text"]),
                    "hypothesis_raw": pr,
                    "reference_norm": rr,
                    "hypothesis_norm": hh,
                    "utterance_char_errors": dist,
                    "reference_char_count": max(len(rr), 1),
                    "exact_match": rr == hh,
                }
                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote predictions to {out_path}")


if __name__ == "__main__":
    main()
