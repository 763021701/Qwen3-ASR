"""Streaming engine test (DESIGN.md §15 T19): chunk-by-chunk output == offline decoding with the same chunk mask,
finalized tokens never change, nothing duplicated at chunk borders.  python -m aut_ctc.tests.test_streaming"""
from __future__ import annotations

import types

import numpy as np
import torch

from ..decode import ctc_greedy
from ..frontend import AUT_FRAMES_PER_CHUNK
from ..model import AuTCTCDecoder, CTCDecoderConfig
from ..streaming import StreamingRecognizer
from .common import random_frontend, test_audio


@torch.no_grad()
def offline_logits(frontend, decoder, wav, window, left_chunks):
    feats, mel_lens = frontend.features([wav])
    layers, lens = frontend.encode(feats, mel_lens, window)
    return decoder(layers, lens, chunk_frames=AUT_FRAMES_PER_CHUNK * window // 100, left_chunks=left_chunks)


def main():
    torch.set_num_threads(8)
    frontend = random_frontend(num_layers=4)
    tokenizer = types.SimpleNamespace(decode=lambda ids: " ".join(map(str, ids)))
    for window, left, upsample, seconds in ((200, 2, 1, 7.31), (100, -1, 2, 4.05), (400, 4, 1, 9.0)):
        torch.manual_seed(window)
        decoder = AuTCTCDecoder(CTCDecoderConfig(vocab_size=64, upsample=upsample)).eval()
        wav = test_audio(seconds, seed=window)
        logits = offline_logits(frontend, decoder, wav, window, left)["logits"][0]
        with torch.no_grad():  # blank on about half of the frames, so collapsing across chunk borders is exercised
            decoder.head.bias[0] += (logits[:, 1:].max(-1).values - logits[:, 0]).median()
        out = offline_logits(frontend, decoder, wav, window, left)
        ref = ctc_greedy(out["logits"], out["lens"], upsample=upsample)[0]
        blank = (out["logits"][0].argmax(-1) == 0).float().mean().item()

        rec = StreamingRecognizer(frontend, decoder, tokenizer, window=window, left_chunks=left)
        finals, rng, pos = [], np.random.default_rng(0), 0
        while pos < len(wav):
            step = int(rng.integers(800, 9000))  # 50-560 ms pieces, not aligned with windows
            finals += rec.accept_waveform(wav[pos : pos + step])["final"]
            assert rec.final == finals  # finalized tokens are never revised
            pos += step
        finals += rec.finalize()["final"]
        got = [(i, round(t, 4)) for i, t in finals]
        want = [(i, round(t, 4)) for i, t in ref]
        assert len(want) > 10 and got == want, (window, got[:10], want[:10])
        print(f"[ok] T19 W={window} left={left} upsample={upsample}, {seconds} s fed in random pieces: {len(got)} "
              f"streamed tokens == offline chunk-masked decoding (blank on {blank:.0%} of frames), never revised")
    print("ALL STREAMING TESTS PASSED")


if __name__ == "__main__":
    main()
