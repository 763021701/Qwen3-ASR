"""Regression tests for controlled degradation and robustness metric semantics."""
import unittest
import io
import soundfile as sf

import numpy as np

from tools.build_controlled_robust_eval import (
    mix_at_snr, noise_family, reverberate, rms, round_robin_rank, shared_peak_gain, reference_text, decode_speech_audio_bytes, check_asset_trace,
)
from tools.score_controlled_robust_eval import derived_metrics, normalized_noise_auc


class ControlledRobustEvalTest(unittest.TestCase):
    def setUp(self):
        t = np.arange(16000, dtype=np.float32)/16000
        self.signal = (.1*np.sin(2*np.pi*233*t)).astype(np.float32)
        self.noise = np.random.default_rng(12).standard_normal(16000).astype(np.float32)

    def test_requested_snr_and_no_hidden_clipping(self):
        for snr in [20, 10, 0]:
            with self.subTest(snr=snr):
                mixed, actual = mix_at_snr(self.signal, self.noise, snr)
                measured = 20*np.log10(rms(self.signal)/rms(mixed-self.signal))
                self.assertAlmostEqual(actual,snr,places=4)
                self.assertAlmostEqual(measured,snr,places=4)

    def test_shared_headroom_preserves_relative_levels(self):
        variants = {"a":self.signal*30,"b":self.signal*4}
        gain = shared_peak_gain(variants)
        for wav in variants.values():
            self.assertLessEqual(float(np.max(np.abs(wav*gain))),.980001)
        self.assertAlmostEqual(rms(variants["a"]*gain)/rms(variants["b"]*gain),7.5,places=5)

    def test_reverb_component_ratio_and_length(self):
        rir = np.array([0.,1.,.5,.25],dtype=np.float32)
        output = reverberate(self.signal,rir,6.)
        self.assertEqual(len(output),len(self.signal))
        ratio = 20*np.log10(rms(self.signal)/rms(output-self.signal))
        self.assertAlmostEqual(ratio,6.,places=4)

    def test_invalid_energy_and_nonfinite_inputs_fail(self):
        for noise in [np.zeros_like(self.signal),np.full_like(self.signal,np.nan)]:
            with self.assertRaises(ValueError):
                mix_at_snr(self.signal,noise,10)
        with self.assertRaises(ValueError):
            reverberate(self.signal,np.zeros(4,dtype=np.float32))

    def test_seeded_dialect_sampling_balances_strata(self):
        rows=[dict(key=f"{group}-{i}",stratum=group) for group in ["A","B","C"] for i in range(5)]
        a=list(round_robin_rank(rows,42,"dev","kespeech"))
        b=list(round_robin_rank(rows,42,"dev","kespeech"))
        self.assertEqual(a,b)
        self.assertEqual(sorted(r["stratum"] for r in a[:6]),["A","A","B","B","C","C"])

    def test_smaller_extra_degradation_can_coexist_with_worse_absolute_error(self):
        result=derived_metrics(.1,.2,.3,.35)
        self.assertAlmostEqual(result["baseline_degradation_pp"],20.)
        self.assertAlmostEqual(result["lora_degradation_pp"],15.)
        self.assertAlmostEqual(result["degradation_advantage_pp"],5.)
        self.assertAlmostEqual(result["absolute_error_delta_pp"],5.)
        self.assertLess(result["relative_error_reduction_percent"],0)

    def test_auc_is_span_normalized_trapezoid_area(self):
        self.assertAlmostEqual(normalized_noise_auc(.1,.2,.3),.2)
        self.assertAlmostEqual(normalized_noise_auc(.1,.2,.5),.25)

    def test_flac_without_total_sample_count_decodes_with_bounded_memory(self):
        buffer=io.BytesIO()
        sf.write(buffer,self.signal,16000,format="FLAC",subtype="PCM_16")
        data=bytearray(buffer.getvalue())
        # Zero the FLAC STREAMINFO total-samples field, as in a pipe encoder.
        bits=int.from_bytes(data[18:26],"big") & ~((1<<36)-1)
        data[18:26]=bits.to_bytes(8,"big")
        actual=decode_speech_audio_bytes(bytes(data))
        self.assertEqual(len(actual),len(self.signal))
        np.testing.assert_allclose(actual,self.signal,atol=1/32768)

    def test_asset_id_shared_by_clips_requires_path_and_group(self):
        allowed={("same-id","/clip0.wav"):"recording",("same-id","/clip1.wav"):"recording"}
        check_asset_trace(allowed,dict(asset_id="same-id",path="/clip0.wav",group_id="recording"))
        with self.assertRaises(ValueError):
            check_asset_trace(allowed,dict(asset_id="same-id",path="/other.wav",group_id="recording"))
        with self.assertRaises(ValueError):
            check_asset_trace(allowed,dict(asset_id="same-id",path="/clip0.wav",group_id="other-recording"))

    def test_gigaspeech_control_tags_are_not_words(self):
        self.assertEqual(reference_text("HELLO <COMMA> WORLD <PERIOD> <NOISE>","gigaspeech"),"HELLO , WORLD .")
        self.assertEqual(reference_text("<SIL> <MUSIC>","gigaspeech"),"")

    def test_all_six_noise_families_are_explicit(self):
        cases=[("DNS-AudioSet","audioset","dns_audioset"),
               ("DNS-Freesound","door","dns_freesound"),("ESC-50","dog","esc50"),
               ("MUSAN","speech/librivox","musan_speech"),
               ("MUSAN","music/fma","musan_music"),("MUSAN","noise/free-sound","musan_noise")]
        for source,category,expected in cases:
            self.assertEqual(noise_family(dict(source=source,category=category)),expected)


if __name__=="__main__":
    unittest.main()
