"""Conformer-lite CTC decoder on top of frozen AuT features (DESIGN.md §5).

AuT features (B, T, 1024) -> LayerMix -> Linear 1024->512 -> [query embeddings] ->
N x pre-LN Block(MHSA+RoPE -> causal depthwise conv -> FFN), self-conditioned InterCTC after block
``inter_ctc_layer`` -> [2x upsample] -> LayerNorm -> Linear 512->V.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerMix(nn.Module):
    """SUPERB-style learned layer weighting: each layer gets an affine-free LayerNorm (intermediate layers
    are the un-normalized pre-LN residual stream), then a softmax-weighted sum. With one layer it is just a
    LayerNorm."""

    def __init__(self, num_layers: int, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.logits = nn.Parameter(torch.zeros(num_layers))

    def forward(self, layers: list[torch.Tensor]) -> torch.Tensor:
        w = self.logits.softmax(0)
        return sum(w[i] * self.norm(h.float()) for i, h in enumerate(layers))


class RotaryEmbedding(nn.Module):
    """RoPE over global frame indices: AuT positions reset every 1 s chunk, so cross-chunk / cross-window
    order has to come from the decoder."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, T: int, device, dtype):
        f = torch.outer(torch.arange(T, device=device, dtype=torch.float32), self.inv_freq.to(device))
        emb = torch.cat([f, f], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class SelfAttention(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        self.h, self.dh, self.p = n_heads, d // n_heads, dropout
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x, attn_mask, cos, sin):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.p if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class ConvModule(nn.Module):
    """Conformer conv module: local continuity that smooths AuT 1 s chunk / window borders.
    Streaming needs causal=True, fixed at training time (WeNet U2 recipe)."""

    def __init__(self, d: int, kernel_size: int = 9, dropout: float = 0.1, causal: bool = False):
        super().__init__()
        self.k, self.causal = kernel_size, causal
        self.pw1 = nn.Conv1d(d, 2 * d, 1)
        self.dw = nn.Conv1d(d, d, kernel_size, groups=d)
        self.norm = nn.LayerNorm(d)
        self.pw2 = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask):  # x: (B, T, D); pad_mask: (B, T), True = valid frame
        m = pad_mask[:, None, :]
        x = F.glu(self.pw1(x.transpose(1, 2).masked_fill(~m, 0.0)), dim=1).masked_fill(~m, 0.0)
        pad = (self.k - 1, 0) if self.causal else ((self.k - 1) // 2, self.k // 2)
        x = self.dw(F.pad(x, pad))
        x = F.silu(self.norm(x.transpose(1, 2))).transpose(1, 2)
        return self.drop(self.pw2(x).transpose(1, 2))


class Block(nn.Module):
    """pre-LN: MHSA(RoPE) -> Conv -> FFN, same pre-LN + GELU FFN style as the AuT layers."""

    def __init__(self, d, n_heads, ffn_mult=4, kernel_size=9, dropout=0.1, causal_conv=False):
        super().__init__()
        self.ln_att, self.ln_conv, self.ln_ffn = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = SelfAttention(d, n_heads, dropout)
        self.conv = ConvModule(d, kernel_size, dropout, causal_conv)
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_mult * d, d)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask, attn_mask, cos, sin):
        x = x + self.drop(self.att(self.ln_att(x), attn_mask, cos, sin))
        x = x + self.conv(self.ln_conv(x), pad_mask)
        return x + self.drop(self.ffn(self.ln_ffn(x)))


def build_attn_mask(lens, T, chunk_frames=0, left_chunks=-1, local_frames=0, num_queries=0):
    """(B, 1, T, T) bool mask, True = visible. T and lens already include the num_queries leading queries.

    chunk_frames > 0: chunk-causal (streaming), 13 * n_window_infer / 100 so chunks line up with AuT windows;
                      left_chunks < 0 = unlimited left context. AuT windows never see each other, so all
                      cross-window context comes from here.
    local_frames > 0: banded local attention for long offline audio (avoids RoPE extrapolation).
    num_queries > 0: SenseVoice-style leading queries. Every position sees the queries; when streaming the
                     queries belong to chunk 0 (fixed once chunk 0 is done), offline they see everything.
    """
    i = torch.arange(T, device=lens.device)
    Q = num_queries
    allow = torch.ones(T, T, dtype=torch.bool, device=lens.device)
    if chunk_frames > 0:
        chunk = torch.where(i < Q, torch.zeros_like(i), (i - Q).clamp(min=0) // chunk_frames)
        diff = chunk[:, None] - chunk[None, :]
        allow = diff >= 0
        if left_chunks >= 0:
            allow &= diff <= left_chunks
    if local_frames > 0:
        allow &= (i[:, None] - i[None, :]).abs() <= local_frames
        allow[:Q, :] = True
    allow[:, :Q] = True
    mask = allow[None] & (i[None, None, :] < lens[:, None, None])
    mask |= torch.eye(T, dtype=torch.bool, device=lens.device)[None]  # padding rows see themselves: no NaN
    return mask[:, None]


@dataclass
class CTCDecoderConfig:
    in_dim: int = 1024  # AuT d_model (896 for the Qwen3-ASR-0.6B AuT)
    layer_ids: tuple = (-1,)  # AuT layers to use; (-1,) = ln_post output
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    ffn_mult: int = 4
    kernel_size: int = 9  # ~0.7 s at 13 Hz
    dropout: float = 0.1
    vocab_size: int = 19911  # set from the tokenizer; 0 = blank, 1 = <unk>
    inter_ctc_layer: int = 3  # self-conditioned InterCTC after this block; 0 = off
    tie_cond: bool = True  # conditioning projection = softmax(p) @ head.weight (saves ~10M params at V~20k)
    upsample: int = 1  # 2 = 26 Hz output, eases the CTC T >= U + repeats constraint
    causal_conv: bool = True  # unified offline/streaming model; must be fixed at training time
    num_queries: int = 0  # 0 = off; 2 = [language, ITN style]
    num_query_types: int = 16  # rows of the query embedding table

    def __post_init__(self):
        self.layer_ids = tuple(self.layer_ids)


class AuTCTCDecoder(nn.Module):
    def __init__(self, cfg: CTCDecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.mix = LayerMix(len(cfg.layer_ids), cfg.in_dim)
        self.in_proj = nn.Sequential(nn.Linear(cfg.in_dim, cfg.d_model), nn.Dropout(cfg.dropout))
        self.rope = RotaryEmbedding(cfg.d_model // cfg.n_heads)
        self.blocks = nn.ModuleList(
            Block(cfg.d_model, cfg.n_heads, cfg.ffn_mult, cfg.kernel_size, cfg.dropout, cfg.causal_conv)
            for _ in range(cfg.n_layers)
        )
        self.ln_out = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)  # shared by the final and the inter CTC
        if cfg.inter_ctc_layer:
            self.ln_inter = nn.LayerNorm(cfg.d_model)
            if not cfg.tie_cond:
                self.cond = nn.Linear(cfg.vocab_size, cfg.d_model)
        if cfg.upsample > 1:
            self.up = nn.Linear(cfg.d_model, cfg.upsample * cfg.d_model)
        if cfg.num_queries:
            self.query_embed = nn.Embedding(cfg.num_query_types, cfg.d_model)

    def forward(self, layers, lens, chunk_frames=0, left_chunks=-1, local_frames=0, query_ids=None):
        """layers: [(B, T, in_dim)]; lens: (B,) AuT frames; query_ids: (B, num_queries) or None.
        Returned lens include the query positions (times upsample); CTC targets must start with the
        corresponding num_queries tags."""
        x = self.in_proj(self.mix(layers))
        Q = self.cfg.num_queries
        if Q:
            assert query_ids is not None and query_ids.shape == (x.size(0), Q)
            x = torch.cat([self.query_embed(query_ids).to(x.dtype), x], dim=1)
            lens = lens + Q
        B, T, _ = x.shape
        pad = torch.arange(T, device=x.device)[None] < lens[:, None]
        mask = build_attn_mask(lens, T, chunk_frames, left_chunks, local_frames, Q)
        cos, sin = self.rope(T, x.device, x.dtype)
        inter_logits = None
        for i, blk in enumerate(self.blocks, 1):
            x = blk(x, pad, mask, cos, sin)
            if i == self.cfg.inter_ctc_layer:  # Self-conditioned CTC (Nozaki & Komatsu, 2021)
                inter_logits = self.head(self.ln_inter(x))
                p = inter_logits.softmax(-1)
                x = x + (p @ self.head.weight if self.cfg.tie_cond else self.cond(p))
        out_lens = lens
        if self.cfg.upsample > 1:
            u = self.cfg.upsample
            x, out_lens = self.up(x).view(B, T * u, -1), lens * u
        return {"logits": self.head(self.ln_out(x)), "lens": out_lens,
                "inter_logits": inter_logits, "inter_lens": lens, "num_queries": Q}


def save_decoder(path, decoder: AuTCTCDecoder, **extra) -> None:
    torch.save({"config": asdict(decoder.cfg), "model": decoder.state_dict(), **extra}, path)


def load_decoder(path, map_location="cpu") -> tuple[AuTCTCDecoder, dict]:
    """Returns (decoder in eval mode, the full checkpoint dict)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    decoder = AuTCTCDecoder(CTCDecoderConfig(**ckpt["config"]))
    decoder.load_state_dict(ckpt["model"])
    return decoder.eval(), ckpt
