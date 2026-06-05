#!/usr/bin/env python3
# coding=utf-8
"""
Evaluate FunASR Paraformer-family checkpoints on a Qwen3-style jsonl manifest.

Same inputs and metrics as `evaluation/cantonese/eval_cantonese_asr_jsonl.py`:
  - Each line: JSON with `audio` (path) and `text` (Qwen3 label, optional `...<asr_text>` prefix).
  - CER (primary) + sentence exact-match after the same text normalization (OpenCC, cn2an, etc.).

Inference uses `funasr.AutoModel` (see ModelScope / FunASR docs), e.g.:
  - SeACo Paraformer (hotword-capable, zh general):
      https://modelscope.cn/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
  - Community Sichuan (chuan) large offline:
      https://www.modelscope.cn/models/dengcunqin/speech_paraformer-large_asr_nat-chuan-16k-common-vocab8404-pytorch/summary
  - Community Cantonese / Mandarin / English online large:
      https://www.modelscope.cn/models/dengcunqin/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online

Dependencies:
  pip install -U funasr modelscope
  pip install jiwer opencc-python-reimplemented cn2an
  (torch / torchaudio per FunASR)

Examples:
  # Sichuan Paraformer (ModelScope id)
  python evaluation/cantonese/baselines/eval_cantonese_asr_paraformer_jsonl.py \\
    --jsonl data/chuan_manifest.jsonl \\
    --model dengcunqin/speech_paraformer-large_asr_nat-chuan-16k-common-vocab8404-pytorch \\
    --batch_size 8

  # SeACo with optional hotwords (space-separated)
  python evaluation/cantonese/baselines/eval_cantonese_asr_paraformer_jsonl.py \\
    --jsonl data/cantonese/common_voice_yue/cv_yue_test_qwen3.jsonl \\
    --model iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch \\
    --hotword "专有名词 品牌名"

  # Long-form: optional VAD (FunASR will run VAD then ASR per segment)
  python evaluation/cantonese/baselines/eval_cantonese_asr_paraformer_jsonl.py \\
    --jsonl data/longform.jsonl \\
    --model iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch \\
    --vad_model damo/speech_fsmn_vad_zh-cn-16k-common-pytorch
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from typing import Any, Dict, List

from masr_eval_pkg import compute_cer, compute_sentence_cer
from masr_eval_pkg.metrics.levenshtein import levenshtein_align
from masr_eval_pkg.normalizers import get_normalizer

import torch

_ASR_TEXT_TAG = "<asr_text>"

_ZH_CONVERT_MAP = {
    "off": "none",
    "to_traditional": "s2t",
    "to_simplified": "t2s",
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paraformer / FunASR ASR on Qwen3 jsonl: CER + sentence accuracy."
    )
    p.add_argument("--jsonl", type=str, required=True, help="Manifest: lines with audio + text.")
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help="FunASR model id (ModelScope / HF hub) or local directory, e.g. "
        "dengcunqin/speech_paraformer-large_asr_nat-chuan-16k-common-vocab8404-pytorch",
    )
    p.add_argument(
        "--model_revision",
        type=str,
        default="master",
        help="ModelScope / hub revision tag (default: master).",
    )
    p.add_argument(
        "--hub",
        type=str,
        default="ms",
        help="Download hub: ms (ModelScope) or hf (Hugging Face), per FunASR.",
    )
    p.add_argument("--max_samples", type=int, default=0, help="If >0, only first N rows.")
    p.add_argument("--batch_size", type=int, default=4, help="FunASR generate() batch size.")
    p.add_argument(
        "--device",
        type=str,
        default="",
        help="cuda, cuda:0, cpu, ... Empty = cuda if available else cpu.",
    )
    p.add_argument("--ncpu", type=int, default=4, help="CPU threads for feature extraction (FunASR).")
    p.add_argument(
        "--vad_model",
        type=str,
        default="",
        help="If set, load a VAD model id/path and run inference_with_vad (good for long audio).",
    )
    p.add_argument(
        "--vad_model_revision",
        type=str,
        default="master",
        help="Revision for vad_model.",
    )
    p.add_argument(
        "--punc_model",
        type=str,
        default="",
        help="If set, load punctuation model id/path (requires compatible FunASR pipeline).",
    )
    p.add_argument(
        "--punc_model_revision",
        type=str,
        default="master",
        help="Revision for punc_model.",
    )
    p.add_argument(
        "--hotword",
        type=str,
        default="",
        help="For SeACo / hotword-capable models: space-separated hotwords passed to generate().",
    )
    p.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Forward trust_remote_code=True to FunASR AutoModel when needed.",
    )
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
        "--eval_label",
        type=str,
        default="Paraformer ASR",
        help="Short label printed in the metrics header.",
    )
    return p.parse_args()

def extract_reference_text(label: str) -> str:
    s = (label or "").strip()
    if not s:
        return ""
    if _ASR_TEXT_TAG in s:
        return s.split(_ASR_TEXT_TAG, 1)[1].strip()
    return s

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

def hypothesis_text_from_funasr_item(item: Any) -> str:
    """Best-effort text field from a FunASR `generate` / `inference` result dict."""
    if not isinstance(item, dict):
        return ""
    for key in ("text", "pred_text"):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    val = item.get("text")
    if val is not None:
        s = str(val).strip()
        if s:
            return s
    return ""

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
        from funasr import AutoModel  # type: ignore[import-not-found]
    except ImportError as exc:
        print(
            "Failed to import funasr. Install with: pip install -U funasr modelscope",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    device = (args.device or "").strip()
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    load_kw: Dict[str, Any] = dict(
        model=args.model,
        model_revision=args.model_revision,
        hub=args.hub,
        device=device,
        ncpu=int(args.ncpu),
        disable_update=True,
    )
    if args.trust_remote_code:
        load_kw["trust_remote_code"] = True
    if (args.vad_model or "").strip():
        load_kw["vad_model"] = args.vad_model.strip()
        load_kw["vad_model_revision"] = args.vad_model_revision
    if (args.punc_model or "").strip():
        load_kw["punc_model"] = args.punc_model.strip()
        load_kw["punc_model_revision"] = args.punc_model_revision

    print(f"Loading FunASR model={args.model!r} revision={args.model_revision!r} device={device!r} hub={args.hub!r} ...")
    model = AutoModel(**load_kw)

    refs_raw = [extract_reference_text(ex["text"]) for ex in rows]
    audios = [ex["audio"] for ex in rows]

    print(f"Transcribing {len(rows)} utterances (batch_size={args.batch_size}) ...")
    predictions: List[str] = []
    gen_kw: Dict[str, Any] = dict(batch_size=int(args.batch_size), disable_pbar=True)
    hw = (args.hotword or "").strip()
    if hw:
        gen_kw["hotword"] = hw

    for beg in range(0, len(audios), args.batch_size):
        batch_paths = audios[beg : beg + args.batch_size]
        results = model.generate(input=batch_paths, **gen_kw)
        if not isinstance(results, list):
            print("Unexpected generate() return type; expected list.", file=sys.stderr)
            sys.exit(1)
        if len(results) != len(batch_paths):
            print(
                f"Warning: batch size mismatch (got {len(results)} results for {len(batch_paths)} inputs). "
                "Padding with empty hypotheses.",
                file=sys.stderr,
            )
        for j, path in enumerate(batch_paths):
            item = results[j] if j < len(results) else {}
            predictions.append(hypothesis_text_from_funasr_item(item))

    if len(predictions) != len(refs_raw):
        print("Internal error: prediction count mismatch.", file=sys.stderr)
        sys.exit(1)

    # --- Normalization via MASR_Eval_Pkg ChineseNormalizer ---
    zh_convert = _ZH_CONVERT_MAP[args.hanzi_script_norm]
    normalizer = get_normalizer(
        "zh",
        zh_convert=zh_convert,
        number_normalize="to_arabic",
    )

    if args.keep_whitespace:
        # Spaces count as characters: use normalize() + collapse whitespace, then manual CER
        def _norm_ws(s: str) -> str:
            n = normalizer.normalize(s)
            return re.sub(r"\s+", " ", n).strip()

        refs = [_norm_ws(r) for r in refs_raw]
        hyps = [_norm_ws(h) for h in predictions]

        total_cer_errors = 0
        total_cer_n = 0
        per_sample_errors = []
        for rr, hh in zip(refs, hyps):
            rc, hc = list(rr), list(hh)
            if len(rc) == 0 and len(hc) == 0:
                per_sample_errors.append(0)
                continue
            if len(rc) == 0:
                total_cer_errors += len(hc)
                total_cer_n += max(len(hc), 1)
                per_sample_errors.append(len(hc))
            else:
                s, d, ins, _ = levenshtein_align(rc, hc)
                err = s + d + ins
                total_cer_errors += err
                total_cer_n += len(rc)
                per_sample_errors.append(err)
        cer = total_cer_errors / max(total_cer_n, 1)
        backend = "masr_levenshtein"
        ref_cer = refs
        hyp_cer = hyps
    else:
        # Standard CER: remove all whitespace
        ref_cer = [normalizer.normalize_for_cer(r) for r in refs_raw]
        hyp_cer = [normalizer.normalize_for_cer(h) for h in predictions]
        cer_result = compute_cer(ref_cer, hyp_cer, per_sample=True)
        cer = cer_result["cer"]
        backend = "masr"
        per_sample_cer = cer_result["per_sample"]

    sentence_accuracy = exact_matches / max(len(refs), 1)

    print("")
    print(f"=== {args.eval_label} (FunASR Paraformer) ===")
    print(f"Model:              {args.model}")
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
            if args.keep_whitespace:
                for ex, pr, rr, hh, dist in zip(rows, predictions, ref_cer, hyp_cer, per_sample_errors):
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
            else:
                for ex, pr, rc, hc, sc in zip(rows, predictions, ref_cer, hyp_cer, per_sample_cer):
                    rec = {
                        "audio": ex["audio"],
                        "reference_raw": extract_reference_text(ex["text"]),
                        "hypothesis_raw": pr,
                        "reference_norm": rc,
                        "hypothesis_norm": hc,
                        "utterance_char_errors": sc["substitutions"] + sc["deletions"] + sc["insertions"],
                        "reference_char_count": max(sc["n_ref_chars"], 1),
                        "exact_match": rc == hc,
                    }
                    wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote predictions to {{out_path}}")
if __name__ == "__main__":
    main()
