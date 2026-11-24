"""Chunk boundary, attention metadata and feature-gradient regression tests."""
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from transformers.modeling_outputs import BaseModelOutput

from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
from qwen_asr.core.transformers_backend import modeling_qwen3_asr as modeling


def tiny_config():
    config = Qwen3ASRAudioEncoderConfig(
        num_mel_bins=8, encoder_layers=2, encoder_attention_heads=2,
        encoder_ffn_dim=16, d_model=8, output_dim=8,
        downsample_hidden_size=2, n_window=50, n_window_infer=800,
    )
    config._attn_implementation = "eager"
    return config


class AudioEncoderMetadataTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_chunk_boundaries_and_reused_attention_lengths(self):
        encoder = modeling.Qwen3ASRAudioEncoder(tiny_config())
        lengths = torch.tensor([1, 8, 99, 100, 101, 799, 800, 801, 1601])
        features = torch.randn(8, int(lengths.sum()), requires_grad=True)
        captured = []
        original = modeling.eager_attention_forward

        def capture(module, query, key, value, **kwargs):
            captured.append((
                kwargs["cu_seq_lens_q"].tolist(),
                kwargs["max_length_q"],
                kwargs["max_length_k"],
            ))
            return original(module, query, key, value, **kwargs)

        with mock.patch.object(modeling, "eager_attention_forward", side_effect=capture):
            output = encoder(features, feature_lens=lengths).last_hidden_state
            output.square().sum().backward()

        # Expected windows include both 100-frame convolution boundaries and
        # the 104-token inference-attention boundary (8 * 13 output frames).
        expected = [0, 1, 2, 15, 28, 42, 146, 250, 354, 355, 459, 563, 564]
        self.assertEqual(tuple(output.shape), (564, 8))
        self.assertEqual(captured, [(expected, 104, 104)] * 2)
        self.assertEqual(features.grad.shape, features.shape)
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertTrue(all(p.grad is not None for p in encoder.parameters()))

    def test_serial_audio_order_and_host_lengths(self):
        calls = []

        class CaptureEncoder:
            def __call__(self, features, feature_lens):
                calls.append((features.shape[-1], feature_lens.device.type, feature_lens.tolist()))
                return BaseModelOutput(last_hidden_state=features.T)

        thinker = SimpleNamespace(audio_tower=CaptureEncoder())
        features = torch.arange(3 * 8 * 101, dtype=torch.float32).reshape(3, 8, 101)
        mask = torch.arange(101).unsqueeze(0) < torch.tensor([1, 100, 101]).unsqueeze(1)
        actual = modeling.Qwen3ASRThinkerForConditionalGeneration.get_audio_features(
            thinker, features, feature_attention_mask=mask)
        expected = torch.cat([features[0, :, :1].T, features[1, :, :100].T, features[2].T])
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(calls, [(1, "cpu", [1]), (100, "cpu", [100]), (101, "cpu", [101])])

    def test_attention_accepts_precomputed_maximum_and_fallback(self):
        attention = modeling.Qwen3ASRAudioAttention(tiny_config()).eval()
        hidden = torch.randn(13, 8)
        cu = torch.tensor([0, 8, 13], dtype=torch.int32)
        expected = attention(hidden, cu)
        with mock.patch.object(torch.Tensor, "max", side_effect=AssertionError("recomputed maximum")):
            actual = attention(hidden, cu, max_seqlen=8)
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
