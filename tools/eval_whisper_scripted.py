#!/usr/bin/env python3
"""Evaluate base and fine-tuned Whisper on scripted metadata CSV."""

import csv
import json
import re
import sys
import time
from pathlib import Path

from faster_whisper import WhisperModel

# --- Config ---
CSV_PATH = "raw/scripted_metadata.csv"
OUT_DIR = Path("outputs/scripted_eval")

BASE_MODEL = "Systran/faster-whisper-large-v3"
FT_MODEL = "/root/autodl-tmp/workspace/project/Export/whisper-large-v3-finetune-ct2"
# Second fine-tuned variant
FT_MIXED_MODEL = "/root/autodl-tmp/workspace/project/Export/whisper-large-v3-finetune-ct2-mixed"

DEVICE = "cuda"
COMPUTE_TYPE = "float16"
BATCH_SIZE = 16


def normalize(text: str) -> str:
    """Match existing eval normalization: lowercase, strip punctuation to spaces."""
    text = text.lower()
    # Replace punctuation (except apostrophes within words) with space
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove standalone apostrophes that became separated
    text = re.sub(r"\b'\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return text.split()


def compute_metrics(ref: str, hyp: str):
    """Token WER and Char CER."""
    ref_tokens = tokenize(ref)
    hyp_tokens = tokenize(hyp)

    # Simple Levenshtein for tokens
    n, m = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    token_errs = dp[n][m]
    token_count = n

    # Char CER
    ref_chars = ref.replace(" ", "")
    hyp_chars = hyp.replace(" ", "")
    nc, mc = len(ref_chars), len(hyp_chars)
    dpc = [[0] * (mc + 1) for _ in range(nc + 1)]
    for i in range(nc + 1):
        dpc[i][0] = i
    for j in range(mc + 1):
        dpc[0][j] = j
    for i in range(1, nc + 1):
        for j in range(1, mc + 1):
            cost = 0 if ref_chars[i - 1] == hyp_chars[j - 1] else 1
            dpc[i][j] = min(dpc[i - 1][j] + 1, dpc[i][j - 1] + 1, dpc[i - 1][j - 1] + cost)

    char_errs = dpc[nc][mc]
    char_count = nc

    return {
        "reference_tokens": ref_tokens,
        "hypothesis_tokens": hyp_tokens,
        "utterance_token_errors": token_errs,
        "reference_token_count": token_count,
        "utterance_char_errors": char_errs,
        "reference_char_count": char_count,
        "exact_match": token_errs == 0,
    }


def load_csv(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def transcribe_dataset(model: WhisperModel, rows: list[dict], model_name: str) -> list[dict]:
    results = []
    audio_paths = [r["audio_path"] for r in rows]
    references = [r["text"] for r in rows]

    # Check all audio files exist first
    for ap in audio_paths:
        if not Path(ap).exists():
            print(f"WARNING: audio not found: {ap}")

    start = time.time()
    segments_list, info = model.transcribe(
        audio_paths,
        batch_size=BATCH_SIZE,
        language=None,  # auto-detect
        without_timestamps=True,
    )
    # faster-whisper batch transcribe returns segments and info differently
    # Actually for batched, we get segments sequentially

    elapsed = time.time() - start
    print(f"Transcription finished in {elapsed:.1f}s for {model_name}")

    return segments_list  # placeholder


def transcribe_one_by_one(model: WhisperModel, rows: list[dict], model_name: str) -> list[dict]:
    """Transcribe each audio individually and return predictions."""
    results = []
    total = len(rows)
    start = time.time()

    for i, row in enumerate(rows):
        audio_path = row["audio_path"]
        ref_raw = row["text"]

        if not Path(audio_path).exists():
            print(f"  [{i+1}/{total}] SKIP (not found): {audio_path}")
            continue

        segments, info = model.transcribe(audio_path, language=None, without_timestamps=True)
        hyp_raw = " ".join(seg.text.strip() for seg in segments).strip()

        ref_norm = normalize(ref_raw)
        hyp_norm = normalize(hyp_raw)

        metrics = compute_metrics(ref_norm, hyp_norm)
        detected_lang = info.language if info else "unknown"
        lang_prob = info.language_probability if info else 0.0

        results.append({
            "audio": audio_path,
            "reference_raw": ref_raw,
            "hypothesis_raw": hyp_raw,
            "reference_norm": ref_norm,
            "hypothesis_norm": hyp_norm,
            **metrics,
            "detected_language": detected_lang,
            "language_probability": lang_prob,
        })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            eta = elapsed / (i + 1) * (total - i - 1)
            print(f"  [{i+1}/{total}] {elapsed:.0f}s elapsed, eta {eta:.0f}s | {model_name}")

    elapsed = time.time() - start
    print(f"  [{total}/{total}] done in {elapsed:.1f}s | {model_name}")
    return results


def compute_summary(results: list[dict]) -> dict:
    total_err = sum(r["utterance_token_errors"] for r in results)
    total_tok = sum(r["reference_token_count"] for r in results)
    total_cerr = sum(r["utterance_char_errors"] for r in results)
    total_char = sum(r["reference_char_count"] for r in results)
    exact = sum(1 for r in results if r["exact_match"])

    return {
        "num_utterances": len(results),
        "token_wer": f"{total_err / total_tok * 100:.2f}%" if total_tok > 0 else "N/A",
        "char_cer": f"{total_cerr / total_char * 100:.2f}%" if total_char > 0 else "N/A",
        "total_token_errors": total_err,
        "total_reference_tokens": total_tok,
        "exact_matches": exact,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_csv(CSV_PATH)
    print(f"Loaded {len(rows)} utterances from {CSV_PATH}")

    models = {
        "whisper_large_v3_base": BASE_MODEL,
        "whisper_large_v3_finetune_ct2": FT_MODEL,
        "whisper_large_v3_finetune_ct2_mixed": FT_MIXED_MODEL,
    }

    # Load and run each model
    summaries = {}

    for name, model_path in models.items():
        print(f"\n{'='*60}")
        print(f"Model: {name}  ({model_path})")
        print(f"{'='*60}")

        model = WhisperModel(model_path, device=DEVICE, compute_type=COMPUTE_TYPE)
        results = transcribe_one_by_one(model, rows, name)
        summary = compute_summary(results)
        summaries[name] = summary

        # Save predictions
        out_path = OUT_DIR / f"{name}_predictions.jsonl"
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Predictions saved to {out_path}")

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"{'Model':<45s} {'WER':>8s} {'CER':>8s} {'Exact':>6s} {'ErrTok':>8s} {'RefTok':>8s}")
    print("-" * 80)
    for name, s in summaries.items():
        print(f"{name:<45s} {s['token_wer']:>8s} {s['char_cer']:>8s} {s['exact_matches']:>5d}/{s['num_utterances']} {s['total_token_errors']:>8d} {s['total_reference_tokens']:>8d}")

    # Save summary
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
