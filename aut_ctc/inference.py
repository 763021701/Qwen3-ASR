"""AuT-CTC head for inference on AuT features computed elsewhere (e.g. by the LLM's own AuT, see biasing.py)."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from .decode import ctc_greedy, hyps_to_text
from .evaluate import resolve_checkpoint
from .frontend import AUT_FRAMES_PER_CHUNK
from .model import load_decoder
from .tokenizer import load_tokenizer


class AuTCTCHead:
    """CTC decoder + tokenizer of one experiment directory, decoding offline (8 s AuT windows)."""

    def __init__(self, exp_dir, ckpt: str = "best", device="cuda"):
        exp = Path(exp_dir)
        cfg = yaml.safe_load((exp / "config.yaml").read_text())
        self.tokenizer = load_tokenizer(cfg["tokenizer"]["sp_model"], exp / "vocab.json")
        decoder, _ = load_decoder(resolve_checkpoint(exp, ckpt))
        assert decoder.cfg.num_queries == 0, "queries are phase 2"
        self.decoder = decoder.to(device)
        self.layer_ids = decoder.cfg.layer_ids
        self.aut_model_dir = cfg["aut"]["model_dir"]
        self.aut_fingerprint = json.loads((exp / "meta.json").read_text())["aut_fingerprint"]
        self.local_frames = int(cfg.get("eval", {}).get("local_frames", 0))
        # 13 AuT frames per 1 s chunk (0.08 s apart inside a chunk): the mean shift is close enough for retrieval
        self.frame_shift_ms = 1000.0 / AUT_FRAMES_PER_CHUNK / decoder.cfg.upsample

    @torch.no_grad()
    def log_probs(self, layers, lens):
        """AuT features ([(B, T, D)] for ``layer_ids``, lens) -> (log-softmax (B, T', V) float32, lens)."""
        dev = lens.device
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out = self.decoder(layers, lens, local_frames=self.local_frames)
        return out["logits"].float().log_softmax(-1), out["lens"]

    def greedy_text(self, log_probs, lens) -> list[str]:
        return hyps_to_text(self.tokenizer, ctc_greedy(log_probs, lens, upsample=self.decoder.cfg.upsample))
