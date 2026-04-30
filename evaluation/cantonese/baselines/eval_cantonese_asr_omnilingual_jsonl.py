#!/usr/bin/env python3
# coding=utf-8
"""
Evaluate Meta omnilingual-asr on a Qwen3-style Cantonese jsonl manifest.

This follows the same omnilingual-asr inference path as
`evaluation/uyghur/baselines/eval_uyghur_asr_omnilingual_jsonl.py`, but uses Cantonese-friendly scoring:
  - CER (primary)
  - Sentence accuracy
  - Hanzi script normalization (default: convert both ref/hyp to Traditional Chinese)
  - Unicode punctuation removed before scoring

Recommended dependencies:
  - Install omnilingual-asr (editable recommended), e.g.:
      pip install -e /path/to/omnilingual-asr
  - fairseq2 and other deps per that repo's README
  - pip install jiwer opencc-python-reimplemented cn2an
  - librosa (for mp3 / mixed corpora)

Default language id is `yue_Hant`, which matches omnilingual-asr's documented
`{language_code}_{script}` naming. If your local omnilingual-asr install uses a
different Cantonese id, pass `--omnilingual_lang` explicitly.

Example:
  python evaluation/cantonese/baselines/eval_cantonese_asr_omnilingual_jsonl.py \
    --jsonl data/cantonese/wsyue_asr/wsyue_asr_eval_qwen3.jsonl \
    --model_card omniASR_LLM_1B_v2 \
    --omnilingual_lang yue_Hant \
    --batch_size 4 \
    --output_predictions outputs/omniASR_LLM_1B_v2/wsyue_predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import torch

_ASR_TEXT_TAG = "<asr_text>"
_ZW_RE = re.compile(r"[\u200c\u200d\ufeff]")
_CN_NUMERAL_RE = re.compile(r"[零〇一二两三四五六七八九十百千万億亿壹贰叁肆伍陆柒捌玖拾佰仟萬廿卅]+")

# Wav2Vec2 stack needs enough raw samples; shorter clips collapse to length 1 after striding and crash conv1d.
_WAV2VEC2_MIN_SAMPLES_16K = 16000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cantonese ASR eval with omnilingual-asr on Qwen3 jsonl: CER + sentence accuracy."
    )
    p.add_argument("--jsonl", type=str, required=True, help="Manifest: lines with audio + text.")
    p.add_argument(
        "--model_card",
        type=str,
        default="omniASR_LLM_7B_v2",
        help="omnilingual-asr hub model card (e.g. omniASR_LLM_3B_v2, omniASR_CTC_7B_v2).",
    )
    p.add_argument(
        "--omnilingual_lang",
        type=str,
        default="yue_Hant",
        help="fairseq2 language id passed to transcribe() (ignored for pure CTC cards).",
    )
    p.add_argument("--max_samples", type=int, default=0, help="If >0, only first N rows.")
    p.add_argument("--batch_size", type=int, default=4, help="Inference batch size.")
    p.add_argument(
        "--device",
        type=str,
        default="",
        help="cuda, cuda:0, cpu, etc. Empty = cuda if available else cpu.",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="bfloat16, float16, float32, or auto (bf16 on SM>=80 else fp16).",
    )
    p.add_argument(
        "--output_predictions",
        type=str,
        default="",
        help="If set, write jsonl with ref/hyp/errors per line.",
    )
    p.add_argument(
        "--audio_input",
        type=str,
        choices=("auto", "path", "librosa"),
        default="auto",
        help="path=native fairseq2 decode (wav/flac only, no min-length pad — short files may crash). "
        "auto and librosa both load with librosa at 16 kHz, pad to a safe minimum, and support mp3; "
        "use either for Common Voice or mixed corpora.",
    )
    p.add_argument(
        "--min_samples_16k",
        type=int,
        default=16000,
        help="Pad 16 kHz waveform to at least max(16000, this value) samples (trailing zeros).",
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


def resolve_torch_dtype(name: str) -> torch.dtype:
    key = str(name or "auto").strip().lower()
    if key == "auto":
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
            return torch.bfloat16
        return torch.float16
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if key not in mapping:
        raise ValueError(f"Unknown dtype {name!r}. Use auto, bfloat16, float16, or float32.")
    return mapping[key]


def load_audio_waveform_dict(path: str, min_samples_16k: int) -> Dict[str, Any]:
    import librosa

    floor = max(_WAV2VEC2_MIN_SAMPLES_16K, max(0, int(min_samples_16k)))
    wav, _sr = librosa.load(path, sr=16000, mono=True)
    if wav.size == 0:
        wav = np.zeros(floor, dtype=np.float32)
    elif int(wav.shape[0]) < floor:
        wav = np.pad(wav, (0, floor - int(wav.shape[0])), mode="constant")
    w = torch.from_numpy(wav.astype("float32", copy=False))
    return {"waveform": w, "sample_rate": 16000}


def build_transcribe_inputs(
    batch_paths: List[str], audio_input: str, min_samples_16k: int
) -> Union[List[str], List[Dict[str, Any]]]:
    if audio_input == "path":
        return batch_paths
    return [load_audio_waveform_dict(p, min_samples_16k) for p in batch_paths]


def main() -> None:
    args = parse_args()
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

    try:
        from omnilingual_asr.models.inference.pipeline import (  # type: ignore[import-not-found]
            ASRInferencePipeline,
        )
    except ImportError as exc:
        print(
            "Failed to import omnilingual_asr. Install the official repo, e.g.\n"
            "  pip install -e /path/to/omnilingual-asr",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    device = (args.device or "").strip()
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = resolve_torch_dtype(args.dtype)
    lang = (args.omnilingual_lang or "").strip() or None

    print(f"Loading omnilingual-asr model_card={args.model_card!r} device={device!r} dtype={dtype} ...")
    pipeline = ASRInferencePipeline(
        model_card=args.model_card,
        device=device,
        dtype=dtype,
    )

    refs_raw = [extract_reference_text(ex["text"]) for ex in rows]
    audios = [ex["audio"] for ex in rows]

    min_s = max(0, int(args.min_samples_16k))
    print(
        f"Transcribing {len(rows)} utterances (omnilingual_lang={lang!r}, audio_input={args.audio_input!r}, "
        f"min_samples_16k={min_s}) ..."
    )
    predictions: List[str] = []
    for start in range(0, len(audios), args.batch_size):
        batch_paths = audios[start : start + args.batch_size]
        batch_inp = build_transcribe_inputs(batch_paths, args.audio_input, min_s)
        if lang:
            langs = [lang] * len(batch_paths)
            texts = pipeline.transcribe(batch_inp, lang=langs, batch_size=len(batch_paths))
        else:
            texts = pipeline.transcribe(batch_inp, batch_size=len(batch_paths))
        predictions.extend(texts)

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
    print("=== Cantonese ASR metrics (omnilingual-asr) ===")
    print(f"Model card:         {args.model_card}")
    print(f"Lang id:            {lang}")
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
    print("Note: For Cantonese Han-character transcripts, CER is the primary metric.")
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
