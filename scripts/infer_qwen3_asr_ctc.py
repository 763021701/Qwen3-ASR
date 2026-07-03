#!/usr/bin/env python3
# coding=utf-8
import argparse
import importlib.util
import os
import sys
import types
from pathlib import Path

import librosa
import torch
from safetensors import safe_open

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

qwen_asr_pkg = sys.modules.get("qwen_asr")
if qwen_asr_pkg is None:
    qwen_asr_pkg = types.ModuleType("qwen_asr")
    qwen_asr_pkg.__path__ = [str(ROOT_DIR / "qwen_asr")]
    sys.modules["qwen_asr"] = qwen_asr_pkg

qwen_asr_core_pkg = sys.modules.get("qwen_asr.core")
if qwen_asr_core_pkg is None:
    qwen_asr_core_pkg = types.ModuleType("qwen_asr.core")
    qwen_asr_core_pkg.__path__ = [str(ROOT_DIR / "qwen_asr" / "core")]
    sys.modules["qwen_asr.core"] = qwen_asr_core_pkg

from qwen_asr.core.transformers_backend import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)
from transformers import AutoConfig, AutoModel, AutoProcessor

AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)


def default_ctc_vocab_path() -> str:
    candidates = [
        os.environ.get("CTC_VOCAB_PATH", ""),
        "/root/.cache/modelscope/hub/models/FunAudioLLM/Fun-ASR-Nano-2512/multilingual.tiktoken",
        os.path.abspath(os.path.join(ROOT_DIR, "..", "Fun-ASR-Nano-2512", "multilingual.tiktoken")),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return candidates[1]


def maybe_add_funasr_path(funasr_path: str):
    candidates = [
        funasr_path,
        os.environ.get("FUNASR_PATH", ""),
        os.path.abspath(os.path.join(ROOT_DIR, "..", "FunASR")),
        os.path.abspath(os.path.join(ROOT_DIR, "..", "Fun-ASR")),
    ]
    for path in candidates:
        if path and os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


def load_ctc_tokenizer(vocab_path: str, funasr_path: str = ""):
    maybe_add_funasr_path(funasr_path)
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"CTC vocab not found: {vocab_path}")

    roots = [
        funasr_path,
        os.environ.get("FUNASR_PATH", ""),
        os.path.abspath(os.path.join(ROOT_DIR, "..", "FunASR")),
        os.path.abspath(os.path.join(ROOT_DIR, "..", "Fun-ASR")),
    ]
    for root in roots:
        if not root:
            continue
        candidates = [
            os.path.join(root, "funasr", "models", "sense_voice", "whisper_lib", "tokenizer.py"),
            os.path.join(root, "models", "sense_voice", "whisper_lib", "tokenizer.py"),
        ]
        for tokenizer_py in candidates:
            if os.path.exists(tokenizer_py):
                spec = importlib.util.spec_from_file_location("_qwen3_asr_sensevoice_tokenizer", tokenizer_py)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module.get_tokenizer(
                    multilingual=True,
                    num_languages=8749,
                    vocab_path=vocab_path,
                )

    from funasr.tokenizer.whisper_tokenizer import SenseVoiceTokenizer

    return SenseVoiceTokenizer(
        vocab_path=vocab_path,
        is_multilingual=True,
        num_languages=8749,
    )


def enable_ctc_training(model, vocab_path: str):
    thinker = model.thinker if hasattr(model, "thinker") else model
    audio_config = getattr(getattr(thinker, "config", None), "audio_config", None)
    ctc_input_dim = getattr(audio_config, "d_model", 1280)
    ctc_config = {
        "enabled": True,
        "input_dim": int(ctc_input_dim),
        "model_dim": 512,
        "ffn_dim": 2048,
        "n_layer": 5,
        "attention_heads": 8,
        "dropout": 0.0,
        "vocab_size": 60515,
        "blank_id": 60514,
        "time_step_sec": 0.08,
        "tokenizer": {
            "name": "SenseVoiceTokenizer",
            "vocab_path": vocab_path,
            "is_multilingual": True,
            "num_languages": 8749,
        },
    }
    thinker.enable_ctc(ctc_config)
    if hasattr(model, "config") and hasattr(model.config, "thinker_config"):
        model.config.thinker_config.ctc_config = ctc_config
    for _, param in model.named_parameters():
        param.requires_grad = False
    for module in (thinker.ctc_decoder, thinker.ctc_head):
        for param in module.parameters():
            param.requires_grad = True
    return ctc_config


def parse_args():
    parser = argparse.ArgumentParser("Run minimal Qwen3-ASR CTC inference on one audio file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Local checkpoint directory with CTC weights.")
    parser.add_argument("--audio", type=str, required=True, help="Input audio path.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--ctc_vocab_path", type=str, default="")
    parser.add_argument("--funasr_path", type=str, default="")
    return parser.parse_args()


def resolve_dtype(dtype_name: str):
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_ctc_checkpoint(checkpoint: str, vocab_path: str, funasr_path: str, device: str, dtype_name: str):
    weights_path = os.path.join(checkpoint, "model.safetensors")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"CTC checkpoint weights not found: {weights_path}")

    model_dtype = resolve_dtype(dtype_name)
    model_kwargs = {"dtype": model_dtype}
    if device != "cpu":
        model_kwargs["device_map"] = device

    model = AutoModel.from_pretrained(checkpoint, **model_kwargs)
    processor = AutoProcessor.from_pretrained(checkpoint, fix_mistral_regex=True)
    ctc_tokenizer = load_ctc_tokenizer(vocab_path, funasr_path)
    enable_ctc_training(model, vocab_path)

    decoder_state = {}
    head_state = {}
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith("thinker.ctc_decoder."):
                decoder_state[key[len("thinker.ctc_decoder.") :]] = handle.get_tensor(key)
            elif key.startswith("thinker.ctc_head."):
                head_state[key[len("thinker.ctc_head.") :]] = handle.get_tensor(key)

    thinker = model.thinker if hasattr(model, "thinker") else model
    thinker.ctc_decoder.load_state_dict(decoder_state, strict=True)
    thinker.ctc_head.load_state_dict(head_state, strict=True)
    model.eval()
    return model, processor, ctc_tokenizer


def load_audio(path: str, sr: int = 16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


@torch.no_grad()
def infer_one(model, processor, ctc_tokenizer, audio_path: str) -> str:
    audio = load_audio(audio_path)
    inputs = processor.feature_extractor(
        [audio],
        sampling_rate=16000,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    model_device = getattr(model, "device", None)
    if model_device is None:
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")
    model_dtype = getattr(model, "dtype", torch.float32)
    input_features = inputs["input_features"].to(device=model_device, dtype=model_dtype)
    feature_attention_mask = inputs["attention_mask"].to(device=model_device)
    outputs = model.generate_ctc(
        input_features=input_features,
        feature_attention_mask=feature_attention_mask,
        tokenizer=ctc_tokenizer,
        return_timestamps=False,
    )
    if not outputs:
        return ""
    return (outputs[0].get("ctc_text") or "").strip()


def main():
    args = parse_args()
    if not os.path.exists(args.audio):
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    vocab_path = args.ctc_vocab_path or default_ctc_vocab_path()
    model, processor, ctc_tokenizer = load_ctc_checkpoint(
        checkpoint=args.checkpoint,
        vocab_path=vocab_path,
        funasr_path=args.funasr_path,
        device=args.device,
        dtype_name=args.dtype,
    )
    print(infer_one(model, processor, ctc_tokenizer, args.audio))


if __name__ == "__main__":
    main()
