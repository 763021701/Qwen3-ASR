"""Shared test fixtures: random-weight AuT, a Whisper feature extractor, a synthetic SentencePiece vocab."""
from __future__ import annotations

import json
import os
import tempfile
import types
from pathlib import Path

import numpy as np
import torch

from ..frontend import _qwen3_asr_modeling, load_feature_extractor
from ..tokenizer import DEFAULT_SPECIALS

MODEL_DIR = Path(os.environ.get("AUT_MODEL_DIR", "/root/autodl-tmp/models/Qwen3-ASR-1.7B"))


def random_aut(num_layers=4, attn_implementation="sdpa", seed=0):
    """AuT with the Qwen3-ASR-1.7B audio config (optionally fewer layers) and random weights."""
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig

    audio_config = json.loads((MODEL_DIR / "config.json").read_text())["thinker_config"]["audio_config"]
    if num_layers:
        audio_config["encoder_layers"] = num_layers
    config = Qwen3ASRAudioEncoderConfig(**audio_config)
    config._attn_implementation = attn_implementation
    torch.manual_seed(seed)
    return _qwen3_asr_modeling().Qwen3ASRAudioEncoder(config).eval().requires_grad_(False)


def random_frontend(num_layers=4, layer_ids=(-1,)):
    """Duck-typed AuTFrontend on CPU with a random-weight AuT (for streaming / decoding tests)."""
    from ..frontend import compute_features, extract_aut_features

    fe, aut = load_feature_extractor(MODEL_DIR), random_aut(num_layers)
    return types.SimpleNamespace(
        fe=fe, aut=aut, layer_ids=tuple(layer_ids), device=torch.device("cpu"),
        features=lambda audios: compute_features(fe, audios),
        encode=lambda feats, lens, n_window_infer=800: extract_aut_features(aut, feats, lens, layer_ids, n_window_infer),
    )


def test_audio(seconds: float, seed=0, click=True) -> np.ndarray:
    """Noise-modulated tones; a loud click at the start makes the running log-mel max equal the global max."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * 16000)) / 16000
    wav = 0.1 * np.sin(2 * np.pi * 220 * t * (1 + 0.3 * np.sin(2 * np.pi * 0.5 * t))) + 0.02 * rng.standard_normal(len(t))
    if click:
        wav[:400] += 0.9 * np.hanning(400)
    return wav.astype(np.float32)


ZH_EN_CORPUS = [
    "我们用 python 写一个 demo", "这个 feature 下周上线", "今天天气不错", "hello world", "会议定在三点",
    "我们今天讨论一下模型训练的计划", "the quick brown fox jumps over the lazy dog", "语音识别是一个有意思的方向",
    "please send me the report by friday", "这个问题我们明天再说", "speech recognition with ctc", "中英文混合的句子 is fine",
]


def synthetic_sp_model(path: Path, extra_lines=(), vocab_size=400) -> Path:
    """Train a small unigram SentencePiece model that contains the 32 SenseVoice tags as user symbols."""
    import sentencepiece as spm

    path.parent.mkdir(parents=True, exist_ok=True)
    corpus = Path(tempfile.mkdtemp()) / "corpus.txt"
    corpus.write_text("\n".join(list(ZH_EN_CORPUS) * 20 + list(extra_lines)), encoding="utf-8")
    prefix = str(path.with_suffix(""))
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=prefix, vocab_size=vocab_size, model_type="unigram",
                                   character_coverage=1.0, user_defined_symbols=list(DEFAULT_SPECIALS),
                                   hard_vocab_limit=False, minloglevel=2)
    return Path(prefix + ".model")
