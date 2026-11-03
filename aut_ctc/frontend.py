"""AuT front end: Whisper-style log-mel, frozen AuT loading, packed-batch feature extraction.

AuT facts this module relies on (DESIGN.md §2):
  * mel (100 frames/s) is split into 1 s chunks; each chunk goes through 3 stride-2 convs -> 13 frames,
    so the output rate is 13 frames/s and frame k of chunk c is centred at c + 0.08*k s.
  * attention is block-diagonal over windows of ``n_window_infer`` mel frames aligned to each utterance's
    start, so a finished window never depends on later audio.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SAMPLE_RATE = 16000
MIN_INPUT_SECONDS = 0.5
AUT_FRAMES_PER_CHUNK = 13  # AuT frames per 1 s chunk
AUT_MEL_PER_CHUNK = 100  # mel frames per chunk (= 2 * n_window)
AUT_PREFIX = "thinker.audio_tower."
_REPO_ROOT = Path(__file__).resolve().parents[1]


def aut_output_lengths(mel_lens: torch.Tensor) -> torch.Tensor:
    """Valid mel frames -> AuT output frames (same as the official _get_feat_extract_output_lengths)."""
    rem = mel_lens % AUT_MEL_PER_CHUNK
    for _ in range(3):  # three k=3, s=2, p=1 convs
        rem = torch.where(rem > 0, (rem - 1) // 2 + 1, torch.zeros_like(rem))
    return rem + (mel_lens // AUT_MEL_PER_CHUNK) * AUT_FRAMES_PER_CHUNK


def aut_frame_times(num_frames: int, upsample: int = 1) -> torch.Tensor:
    """Centre time (s) of every output (sub)frame. Not uniform i/12.5: use this for timestamps."""
    idx = torch.arange(num_frames * upsample)
    frame, sub = idx // upsample, idx % upsample
    chunk, k = frame // AUT_FRAMES_PER_CHUNK, frame % AUT_FRAMES_PER_CHUNK
    return chunk.float() + 0.08 * k.float() + 0.08 * sub.float() / upsample


def _qwen3_asr_modeling():
    from qwen_asr.core.transformers_backend import modeling_qwen3_asr

    if not Path(modeling_qwen3_asr.__file__).resolve().is_relative_to(_REPO_ROOT):
        # An installed qwen-asr wheel lacks the sdpa/eager window-mask fix, so its AuT ignores n_window_infer.
        raise ImportError(
            f"qwen_asr was imported from {modeling_qwen3_asr.__file__}, not from {_REPO_ROOT}. "
            "Run from the repo root (python -m aut_ctc....) or `pip install -e .`."
        )
    return modeling_qwen3_asr


def load_feature_extractor(model_dir):
    from transformers import WhisperFeatureExtractor

    return WhisperFeatureExtractor.from_pretrained(model_dir)


def compute_features(fe, audios, sampling_rate: int = SAMPLE_RATE):
    """Batch log-mel -> (input_features (B, 128, T), mel_lens (B,)).

    padding="longest" is required: the extractor's default pads every clip to 30 s (and truncates).
    Clips shorter than 0.5 s are zero-padded to 0.5 s and the padding counts as valid, like official Qwen3-ASR.
    """
    min_len = int(MIN_INPUT_SECONDS * sampling_rate)
    audios = [a if len(a) >= min_len else np.pad(a, (0, min_len - len(a))) for a in audios]
    out = fe(
        audios,
        sampling_rate=sampling_rate,
        padding="longest",
        truncation=False,
        return_attention_mask=True,
        return_tensors="pt",
    )
    return out["input_features"], out["attention_mask"].sum(-1)


def load_aut(model_dir, device="cpu", dtype=torch.bfloat16, attn_implementation: str = "sdpa"):
    """Build the AuT from a qwen-asr checkpoint, reading only the thinker.audio_tower.* tensors."""
    modeling = _qwen3_asr_modeling()
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
    from safetensors import safe_open

    model_dir = Path(model_dir)
    audio_config = json.loads((model_dir / "config.json").read_text())["thinker_config"]["audio_config"]
    config = Qwen3ASRAudioEncoderConfig(**audio_config)
    config._attn_implementation = attn_implementation
    aut = modeling.Qwen3ASRAudioEncoder(config)
    state = {}
    for shard in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            for key in f.keys():
                if key.startswith(AUT_PREFIX):
                    state[key[len(AUT_PREFIX) :]] = f.get_tensor(key)
    aut.load_state_dict(state, strict=True)
    return aut.to(device=device, dtype=dtype).eval().requires_grad_(False)


def aut_fingerprint(model_dir) -> str:
    """Revision id for a local checkpoint (no HF commit hash): config.json + shard names/sizes."""
    model_dir = Path(model_dir)
    h = hashlib.sha256((model_dir / "config.json").read_bytes())
    for shard in sorted(model_dir.glob("*.safetensors")):
        h.update(f"{shard.name}:{shard.stat().st_size}".encode())
    return h.hexdigest()[:16]


@torch.no_grad()  # not inference_mode: inference tensors cannot be saved for backward by the decoder
def extract_aut_features(aut, input_features, mel_lens, layer_ids=(-1,), n_window_infer: int = 800):
    """Run the frozen AuT on a padded batch -> ([(B, T, D) for each id in layer_ids], lens).

    The batch is packed along time (the AuT forward takes concatenated mel + per-utterance lengths),
    which is exact because every window starts inside its own utterance.
    layer_ids: -1 = ln_post output (before proj1/proj2); i >= 0 = output of transformer layer i
    (0-based, un-normalized residual stream).
    n_window_infer: 100/200/400/800 mel frames -> 1/2/4/8 s attention windows.
    """
    assert n_window_infer % AUT_MEL_PER_CHUNK == 0, n_window_infer
    aut.n_window_infer = n_window_infer
    param = next(aut.parameters())
    mel_lens = mel_lens.to(param.device)
    chunks = [f[:, :n] for f, n in zip(input_features, mel_lens.tolist())]
    if int(mel_lens.max()) < AUT_MEL_PER_CHUNK:
        # The AuT pads tail chunks only up to the longest chunk in the pack, so without any full 1 s chunk the
        # convs would see fewer zeros than in training batches. A silent 1 s dummy (own window) fixes that.
        chunks.append(input_features.new_zeros(input_features.size(1), AUT_MEL_PER_CHUNK))
    feature_lens = torch.tensor([c.size(1) for c in chunks], device=param.device)
    packed = torch.cat(chunks, dim=1)
    grabbed, hooks = {}, []
    for i in set(layer_ids):
        module = aut.ln_post if i < 0 else aut.layers[i]
        hooks.append(
            module.register_forward_hook(lambda m, args, out, i=i: grabbed.__setitem__(i, out if i < 0 else out[0]))
        )
    try:
        aut(packed.to(device=param.device, dtype=param.dtype), feature_lens=feature_lens)
    finally:
        for h in hooks:
            h.remove()
    lens = aut_output_lengths(mel_lens)
    split = aut_output_lengths(feature_lens).tolist()
    feats = [nn.utils.rnn.pad_sequence(grabbed[i].split(split)[: len(lens)], batch_first=True) for i in layer_ids]
    return feats, lens


class AuTFrontend:
    """Feature extractor + frozen AuT on one device: what training, evaluation and streaming share."""

    def __init__(self, model_dir, device="cuda", dtype=torch.bfloat16, layer_ids=(-1,), attn_implementation="sdpa"):
        self.model_dir = str(model_dir)
        self.fe = load_feature_extractor(model_dir)
        self.aut = load_aut(model_dir, device, dtype, attn_implementation)
        self.layer_ids = tuple(layer_ids)
        self.device = torch.device(device)
        self.fingerprint = aut_fingerprint(model_dir)

    def features(self, audios):
        return compute_features(self.fe, audios)

    def encode(self, input_features, mel_lens, n_window_infer: int = 800):
        return extract_aut_features(self.aut, input_features, mel_lens, self.layer_ids, n_window_infer)
