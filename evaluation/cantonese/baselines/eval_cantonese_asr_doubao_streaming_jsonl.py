#!/usr/bin/env python3
# coding=utf-8
"""
Evaluate Doubao streaming ASR on a Qwen3-style Cantonese jsonl manifest.

This script uses the Volcengine / Doubao WebSocket streaming ASR protocol shown in
`refs/asr_doubao.py`, but runs it over a dataset manifest and reports Cantonese CER.

Dependencies:
  pip install soundfile numpy "websockets>=11"
  pip install jiwer opencc-python-reimplemented cn2an  # recommended for scoring

Required env vars:
  DOUBAO_ASR_APP_ID
  DOUBAO_ASR_ACCESS_KEY

Optional env vars:
  DOUBAO_ASR_ENDPOINT
  DOUBAO_ASR_RESOURCE_ID
  DOUBAO_ASR_MODEL

Example:
  export DOUBAO_ASR_APP_ID=...
  export DOUBAO_ASR_ACCESS_KEY=...
  python evaluation/cantonese/baselines/eval_cantonese_asr_doubao_streaming_jsonl.py \
    --jsonl data/cantonese/wsyue_asr/wsyue_asr_eval_qwen3.jsonl \
    --pace_ms 0 \
    --concurrency 8 \
    --output_predictions outputs/doubao_asr/wsyue_eval/predictions.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import re
import struct
import sys
import unicodedata
import uuid
from typing import Any, Dict, List

from masr_eval_pkg import compute_cer, compute_sentence_cer
from masr_eval_pkg.metrics.levenshtein import levenshtein_align
from masr_eval_pkg.normalizers import get_normalizer

import numpy as np

try:
    import soundfile as sf
except ImportError as exc:
    raise SystemExit("Missing soundfile: pip install soundfile") from exc

ENDPOINT = os.getenv(
    "DOUBAO_ASR_ENDPOINT",
    "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async",
)
SAMPLE_RATE = 16000
BLOCK_SIZE = 3200  # 200ms @ 16kHz mono int16 -> 6400 bytes
_ASR_TEXT_TAG = "<asr_text>"

_ZH_CONVERT_MAP = {
    "off": "none",
    "to_traditional": "s2t",
    "to_simplified": "t2s",
}

# Volcengine binary protocol constants
PROTOCOL_VERSION = 0b0001
HEADER_SIZE = 0b0001
FULL_CLIENT_REQUEST = 0b0001
AUDIO_ONLY_REQUEST = 0b0010
FULL_SERVER_RESPONSE = 0b1001
SERVER_ERROR_RESPONSE = 0b1111
NO_SEQUENCE = 0b0000
POS_SEQUENCE = 0b0001
NEG_SEQUENCE = 0b0010
JSON_SER = 0b0001
GZIP_COMP = 0b0001

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cantonese ASR eval with Doubao WebSocket streaming API on Qwen3 jsonl."
    )
    p.add_argument("--jsonl", type=str, required=True, help="Manifest: lines with audio + text.")
    p.add_argument("--max_samples", type=int, default=0, help="If >0, only first N rows.")
    p.add_argument(
        "--pace_ms",
        type=float,
        default=0.0,
        help="Delay between streaming chunks in ms; 0 = send as fast as possible.",
    )
    p.add_argument(
        "--connect_timeout_sec",
        type=float,
        default=30.0,
        help="WebSocket connect timeout in seconds.",
    )
    p.add_argument(
        "--receive_timeout_sec",
        type=float,
        default=120.0,
        help="Per-message receive timeout in seconds.",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Max concurrent WebSocket recognition sessions (1 = serial, same as before).",
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
        "--output_predictions",
        type=str,
        default="",
        help="If set, write jsonl with ref/hyp/errors per line.",
    )
    return p.parse_args()

def make_header(msg_type: int, flags: int = NO_SEQUENCE) -> bytearray:
    h = bytearray(4)
    h[0] = (PROTOCOL_VERSION << 4) | HEADER_SIZE
    h[1] = (msg_type << 4) | flags
    h[2] = (JSON_SER << 4) | GZIP_COMP
    h[3] = 0x00
    return h

def parse_response(res: bytes) -> dict:
    header_size = res[0] & 0x0F
    message_type = res[1] >> 4
    flags = res[1] & 0x0F
    serial = res[2] >> 4
    compress = res[2] & 0x0F
    payload = res[header_size * 4 :]
    result: dict = {"is_last_package": bool(flags & 0x02)}
    if flags & 0x01:
        result["payload_sequence"] = int.from_bytes(payload[:4], "big", signed=True)
        payload = payload[4:]
    payload_msg: bytes | None = None
    if message_type == FULL_SERVER_RESPONSE:
        payload_msg = payload[4:]
    elif message_type == SERVER_ERROR_RESPONSE:
        result["code"] = int.from_bytes(payload[:4], "big", signed=False)
        payload_msg = payload[8:]
    if payload_msg is None:
        return result
    if compress == GZIP_COMP and payload_msg:
        payload_msg = gzip.decompress(payload_msg)
    if serial == JSON_SER and payload_msg:
        result["payload_msg"] = json.loads(payload_msg.decode("utf-8"))
    elif payload_msg:
        result["payload_msg"] = payload_msg.decode("utf-8", errors="replace")
    return result

async def send_init(ws, uid: str, model: str):
    cfg = {
        "user": {"uid": uid},
        "audio": {"format": "pcm", "codec": "raw", "sample_rate": SAMPLE_RATE, "channel": 1},
        "request": {"model_name": model, "enable_punc": True, "enable_itn": True},
    }
    payload = gzip.compress(json.dumps(cfg, ensure_ascii=False).encode("utf-8"))
    frame = bytearray(make_header(FULL_CLIENT_REQUEST, flags=POS_SEQUENCE))
    frame.extend((1).to_bytes(4, "big", signed=True))
    frame.extend(struct.pack(">I", len(payload)))
    frame.extend(payload)
    await ws.send(bytes(frame))

async def send_audio(ws, audio: bytes, last: bool = False):
    payload = gzip.compress(audio) if audio else gzip.compress(b"")
    flags = NEG_SEQUENCE if last else NO_SEQUENCE
    frame = bytearray(make_header(AUDIO_ONLY_REQUEST, flags=flags))
    frame.extend(struct.pack(">I", len(payload)))
    frame.extend(payload)
    await ws.send(bytes(frame))

def load_audio_pcm_mono_int16(path: str, target_sr: int = SAMPLE_RATE) -> bytes:
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    if data.shape[1] > 1:
        data = np.mean(data, axis=1)
    else:
        data = data[:, 0]
    if sr != target_sr:
        if len(data) == 0:
            return b""
        new_len = max(1, int(round(len(data) * target_sr / sr)))
        old_idx = np.arange(len(data), dtype=np.float64)
        new_idx = np.linspace(0.0, len(data) - 1.0, new_len)
        data = np.interp(new_idx, old_idx, data.astype(np.float64)).astype(np.float32)
    data = np.clip(data, -1.0, 1.0)
    pcm = (data * 32767.0).astype(np.int16)
    return pcm.tobytes()

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

async def transcribe_streaming_once(
    audio_path: str,
    *,
    pace_ms: float,
    connect_timeout_sec: float,
    receive_timeout_sec: float,
) -> str:
    try:
        import websockets  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise SystemExit("Missing websockets: pip install 'websockets>=11'") from exc

    app_id = os.getenv("DOUBAO_ASR_APP_ID")
    access_key = os.getenv("DOUBAO_ASR_ACCESS_KEY")
    if not app_id or not access_key:
        raise SystemExit("[x] Set DOUBAO_ASR_APP_ID and DOUBAO_ASR_ACCESS_KEY")

    conn_id = str(uuid.uuid4())
    headers = {
        "X-Api-Resource-Id": os.getenv("DOUBAO_ASR_RESOURCE_ID", "volc.seedasr.sauc.duration"),
        "X-Api-Access-Key": access_key,
        "X-Api-App-Key": app_id,
        "X-Api-Connect-Id": conn_id,
        "X-Api-Request-Id": conn_id,
    }
    model = os.getenv("DOUBAO_ASR_MODEL", "bigmodel")
    pcm = load_audio_pcm_mono_int16(audio_path)
    frame_bytes = BLOCK_SIZE * 2

    async def _connect():
        last_err = None
        for key in ("additional_headers", "extra_headers"):
            try:
                return await asyncio.wait_for(
                    websockets.connect(ENDPOINT, max_size=2**24, **{key: headers}),
                    timeout=connect_timeout_sec,
                )
            except TypeError as exc:
                last_err = exc
                continue
        raise RuntimeError(
            "websockets missing additional_headers / extra_headers; pip install -U 'websockets>=11'"
        ) from last_err

    async def _recv(ws):
        return await asyncio.wait_for(ws.recv(), timeout=receive_timeout_sec)

    ws = await _connect()
    try:
        await send_init(ws, uid=str(uuid.uuid4()), model=model)
        init_resp = parse_response(await _recv(ws))
        if "code" in init_resp:
            raise RuntimeError(f"Init failed: {init_resp}")

        offset = 0
        while offset < len(pcm):
            chunk = pcm[offset : offset + frame_bytes]
            offset += len(chunk)
            await send_audio(ws, chunk, last=False)
            if pace_ms > 0 and offset < len(pcm):
                await asyncio.sleep(pace_ms / 1000.0)
        await send_audio(ws, b"", last=True)

        final_segments: List[str] = []
        seen_final = 0
        latest_result_text = ""
        while True:
            res = parse_response(await _recv(ws))
            if "code" in res:
                raise RuntimeError(f"Server error {res.get('code')}: {res.get('payload_msg')}")
            body = res.get("payload_msg") or {}
            result = body.get("result") or {}
            latest_result_text = result.get("text") or latest_result_text
            utts = result.get("utterances") or []
            finals = [u.get("text", "") for u in utts if u.get("definite")]
            for txt in finals[seen_final:]:
                final_segments.append(txt)
                seen_final += 1
            if res.get("is_last_package"):
                break

        final_text = "".join(t for t in final_segments if t)
        if final_text:
            return final_text.strip()
        return str(latest_result_text or "").strip()
    finally:
        await ws.close()

async def evaluate_many(
    audios: List[str],
    *,
    pace_ms: float,
    connect_timeout_sec: float,
    receive_timeout_sec: float,
    concurrency: int,
) -> List[str]:
    """Run streaming ASR on many files with a concurrency cap; results stay manifest-ordered."""
    n = len(audios)
    predictions: List[str] = [""] * n
    if n == 0:
        return predictions

    sem = asyncio.Semaphore(concurrency)

    async def worker(sample_index: int, audio_path: str) -> None:
        display_idx = sample_index + 1
        async with sem:
            print(
                f"[{display_idx}/{n}] start {audio_path}",
                file=sys.stderr,
                flush=True,
            )
            try:
                hyp = await transcribe_streaming_once(
                    audio_path,
                    pace_ms=pace_ms,
                    connect_timeout_sec=connect_timeout_sec,
                    receive_timeout_sec=receive_timeout_sec,
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(
                    f"[{display_idx}/{n}] failed {audio_path}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                hyp = ""
            predictions[sample_index] = hyp
            print(
                f"[{display_idx}/{n}] done {audio_path}",
                file=sys.stderr,
                flush=True,
            )

    await asyncio.gather(
        *(worker(i, path) for i, path in enumerate(audios)),
    )
    return predictions

def main() -> None:
    args = parse_args()
    if args.concurrency < 1:
        print("[x] --concurrency must be >= 1", file=sys.stderr)
        sys.exit(1)
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

    refs_raw = [extract_reference_text(ex["text"]) for ex in rows]
    audios = [ex["audio"] for ex in rows]

    print(
        f"Evaluating {len(rows)} utterances with Doubao streaming ASR "
        f"(pace_ms={args.pace_ms}, concurrency={args.concurrency}, endpoint={ENDPOINT!r}) ..."
    )

    pace_ms = max(0.0, args.pace_ms)
    predictions = asyncio.run(
        evaluate_many(
            audios,
            pace_ms=pace_ms,
            connect_timeout_sec=args.connect_timeout_sec,
            receive_timeout_sec=args.receive_timeout_sec,
            concurrency=args.concurrency,
        )
    )

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
    print("=== Cantonese ASR metrics (Doubao streaming) ===")
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
    print("Note: Uses Doubao WebSocket streaming ASR and Cantonese CER normalization.")

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
