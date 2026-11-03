"""Streaming recognizer (DESIGN.md §10.2), v1: the decoder is recomputed over all frames with a chunk-causal mask.

Decoder chunks line up with AuT windows (W mel frames -> 13*W/100 AuT frames). A window is finished once its
mel frames are fully determined (W*160 samples plus the n_fft/2 look-ahead of its last STFT frame); its AuT
output is cached and equals the offline windowed result. Chunk-causal attention + causal convolution make the
decoder output of finished chunks independent of later audio, so their tokens are final.
Only difference from offline decoding with the same W / left_chunks: Whisper's log-mel floor (max - 8) uses the
maximum of the audio received so far instead of the whole utterance (bins > 80 dB below the peak only).
"""
from __future__ import annotations

import numpy as np
import torch

from .frontend import AUT_FRAMES_PER_CHUNK, AUT_MEL_PER_CHUNK, aut_frame_times, compute_features, extract_aut_features


class StreamingRecognizer:
    def __init__(self, frontend, decoder, tokenizer, window: int = 200, left_chunks: int = 4):
        assert window % AUT_MEL_PER_CHUNK == 0, window
        assert decoder.cfg.num_queries == 0, "queries are phase 2"
        self.frontend, self.decoder, self.tokenizer = frontend, decoder.eval(), tokenizer
        self.window, self.left_chunks = window, left_chunks
        self.chunk_frames = AUT_FRAMES_PER_CHUNK * window // AUT_MEL_PER_CHUNK
        self.hop = frontend.fe.hop_length
        self.lookahead = frontend.fe.n_fft // 2
        self.reset()

    def reset(self):
        self.audio = np.zeros(0, dtype=np.float32)
        self.done_windows = 0
        self.layers = [[] for _ in self.frontend.layer_ids]  # per layer: AuT features of finished windows
        self.final = []  # [(token_id, time_s)]
        self.last_label = None  # argmax of the last finished output frame, to collapse repeats across chunks

    # ------------------------------------------------------------------ internals
    def _aut(self, mel):
        """AuT on one window's mel (128, n) as an independent utterance -> per-layer (13*n/100, D)."""
        feats, _ = extract_aut_features(self.frontend.aut, mel[None], torch.tensor([mel.size(1)]),
                                        self.frontend.layer_ids, n_window_infer=self.window)
        return [f[0] for f in feats]

    @torch.no_grad()
    def _logits(self, extra=None):
        layers = [torch.cat(parts + ([extra[i]] if extra else []), dim=0)[None] for i, parts in enumerate(self.layers)]
        T = layers[0].size(1)
        dev = self.frontend.device
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out = self.decoder(layers, torch.tensor([T], device=dev), chunk_frames=self.chunk_frames,
                               left_chunks=self.left_chunks)
        return out["logits"][0].float()

    def _greedy(self, logits, first_frame: int, prev):
        """Collapse the argmax of output frames starting at AuT frame ``first_frame`` -> (tokens, last label)."""
        u = self.decoder.cfg.upsample
        ids = logits[first_frame * u :].argmax(-1).tolist()
        times = aut_frame_times(first_frame + len(ids) // u, u)[first_frame * u :].tolist()
        tokens = []
        for t, label in zip(times, ids):
            if label != 0 and label != prev:
                tokens.append((label, t))
            prev = label
        return tokens, prev

    def _finish(self, mel):
        """Append a finished window (its mel frames) and return its final tokens."""
        for parts, f in zip(self.layers, self._aut(mel)):
            parts.append(f)
        tokens, self.last_label = self._greedy(self._logits(), self.done_windows * self.chunk_frames, self.last_label)
        self.done_windows += 1
        self.final += tokens
        return tokens

    def _mel(self, num_samples: int):
        return compute_features(self.frontend.fe, [self.audio[:num_samples]])[0][0]

    # ------------------------------------------------------------------ API
    def accept_waveform(self, pcm) -> dict:
        """Append 16 kHz audio (float in [-1, 1] or int16).

        Returns {"final": tokens finalized by this call, "partial": provisional tokens of the unfinished window
        (replaced on the next call)}; tokens are (token_id, time_s).
        """
        pcm = np.asarray(pcm)
        pcm = pcm.astype(np.float32) / 32768.0 if pcm.dtype == np.int16 else pcm.astype(np.float32)
        self.audio = np.concatenate([self.audio, pcm])
        W, new_final = self.window, []
        while len(self.audio) >= (self.done_windows + 1) * W * self.hop + self.lookahead:
            k = self.done_windows
            new_final += self._finish(self._mel((k + 1) * W * self.hop + self.lookahead)[:, k * W : (k + 1) * W])
        partial = []
        rest = len(self.audio) // self.hop - self.done_windows * W
        if rest > 0:
            mel = self._mel(len(self.audio))[:, self.done_windows * W :]
            partial, _ = self._greedy(self._logits(self._aut(mel)), self.done_windows * self.chunk_frames,
                                      self.last_label)
        return {"final": new_final, "partial": partial}

    def finalize(self) -> dict:
        """End of stream: finish the remaining (possibly partial) windows. Returns {"final": new tokens}."""
        mel_all = self._mel(len(self.audio))
        W, new_final = self.window, []
        while self.done_windows * W < mel_all.size(1):
            k = self.done_windows
            new_final += self._finish(mel_all[:, k * W : (k + 1) * W])
        return {"final": new_final}

    def text(self) -> str:
        return self.tokenizer.decode([tid for tid, _ in self.final])
