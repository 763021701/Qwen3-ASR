"""Biasing pipeline tests on the real Qwen3-ASR-1.7B and an AuT-CTC experiment (GPU):
B1 the ctc_rag_hw piece table; B2 the shared AuT pass returns the LLM's own audio embeddings (bit-exact for one
utterance, bf16 noise when packed); B3 fed the LLM's own embeddings, the pipeline's generate path transcribes exactly
like Qwen3ASRModel.transcribe; B4 the AuT runs once per batch and generate() never re-encodes the audio.
  python -m aut_ctc.tests.test_biasing [--exp_dir exp/mix3000h_stream --ckpt last]"""
from __future__ import annotations

import argparse
import json
import os

import torch

from qwen_asr.inference.utils import parse_asr_output

from ..biasing import CTCRagQwen3ASR, CTCRagTokenizer
from ..data import read_audio
from ..frontend import compute_features, extract_aut_features
from ..train import cap_gpu_memory
from .common import MODEL_DIR

DEV = "/root/autodl-tmp/huangpu/aut_ctc_data/manifests/mix200h"


def dev_clips():
    """Three clips (>= 1 s) each from WenetSpeech, GigaSpeech and WenetSpeech-Yue dev, one of them over 8 s."""
    wavs = []
    for name in ("dev_wenetspeech", "dev_gigaspeech", "dev_yue"):
        items = [e for e in map(json.loads, open(f"{DEV}/{name}.jsonl")) if e["duration"] >= 1]
        items = [next(e for e in items if 8 < e["duration"] <= 20)] + items[:2]
        wavs += [read_audio(e["audio"], tar_offset=e["tar_offset"], tar_size=e["tar_size"]) for e in items]
    return wavs


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", default="exp/mix3000h_stream")
    p.add_argument("--ckpt", default="last")
    args = p.parse_args()
    cap_gpu_memory(torch.device("cuda:0"), 8)
    pipe = CTCRagQwen3ASR.from_pretrained(str(MODEL_DIR), args.exp_dir, args.ckpt)
    asr, thinker = pipe.asr, pipe.asr.model.thinker

    tok = pipe.ctc.tokenizer
    table = CTCRagTokenizer(tok)
    assert table.decode(tok.encode("hello world 2024年")) == " hello world 2024年"
    assert table.decode(tok.encode("今天天气不错")) == "今天天气不错"
    assert table.decode([tok.BLANK, tok.UNK, tok.tag_id("<|yue|>")]) == ""
    print("B1 ok: pieces keep word boundaries, blank/unk/tags decode to ''", flush=True)

    wavs = dev_clips()
    feats, mel_lens = compute_features(asr.processor.feature_extractor, wavs)

    def aut_out(batch):  # LLM audio embeddings of feats[batch] from one packed AuT pass
        return extract_aut_features(pipe.aut, feats[batch], mel_lens[batch], return_output=True)[2]

    def rel(a, b):
        return ((a.float() - b.float()).abs().mean() / b.float().abs().mean()).item()

    with torch.no_grad():
        mask = (torch.arange(feats.size(-1))[None] < mel_lens[:, None]).to(asr.device)
        packed = aut_out(slice(None))
        ref = list(thinker.get_audio_features(feats.to(asr.device, asr.dtype), mask).split([len(x) for x in packed]))
        assert all(torch.equal(aut_out(slice(k, k + 1))[0], ref[k]) for k in range(len(wavs)))
        bf16 = max(rel(a, b) for a, b in zip(packed, ref))
        pipe.aut.float()
        torch.backends.cudnn.allow_tf32 = False  # the AuT convs would otherwise run in TF32
        try:
            fp32 = max(rel(a, aut_out(slice(k, k + 1))[0]) for k, a in enumerate(aut_out(slice(None))))
        finally:
            pipe.aut.to(asr.dtype)
            torch.backends.cudnn.allow_tf32 = True
    print(f"B2 ok: one utterance bit-exact with get_audio_features; packing {len(wavs)} clips changes the embeddings "
          f"by {fp32:.1e} in fp32 (exact) and {bf16:.1e} in bf16 (rounding)", flush=True)
    assert fp32 < 1e-4, fp32

    expected = [r.text for r in asr.transcribe([(w, 16000) for w in wavs])]
    with torch.no_grad():
        got = [parse_asr_output(r)[1] for r in pipe._generate(ref, [""] * len(wavs), None)]
    assert got == expected, list(zip(got, expected))
    e2e = [r.text for r in pipe.transcribe([(w, 16000) for w in wavs], context="")]
    same = sum(a == b for a, b in zip(e2e, expected))
    print(f"B3 ok: generate path == transcribe on {len(wavs)}/{len(wavs)}; "
          f"end to end with the packed AuT {same}/{len(wavs)} identical", flush=True)
    for a, b in zip(e2e, expected):
        if a != b:
            print(f"   packed: {a}\n   transcribe: {b}")

    calls = {"aut": 0, "get_audio_features": 0}
    hook = pipe.aut.register_forward_hook(lambda *_: calls.update(aut=calls["aut"] + 1))
    orig = thinker.get_audio_features

    def counting(*a, **kw):
        calls["get_audio_features"] += 1
        return orig(*a, **kw)

    thinker.get_audio_features = counting
    pipe.max_batch_size, pipe.max_batch_seconds = 4, 1000.0
    try:
        pipe.transcribe([(w, 16000) for w in wavs], context="")
    finally:
        hook.remove()
        del thinker.get_audio_features
    assert calls == {"aut": 3, "get_audio_features": 0}, calls
    print("B4 ok: 9 segments in 3 batches -> 3 AuT passes, get_audio_features never called", flush=True)


if __name__ == "__main__":
    main()
