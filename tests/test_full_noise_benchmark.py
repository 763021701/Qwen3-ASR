"""Full coverage, complete duration and WebDataset corruption integrity."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from tools.build_full_noise_benchmark import (
    CONDITIONS, ShardWriter, excluded, family_assignments, render_variants,
    invalid_speech_reason, digest_bytes,
    verify_dataset, write_json,
)
from tools.build_controlled_robust_eval import sha256_file


class FullNoiseBenchmarkTest(unittest.TestCase):
    def test_explicit_excluded_families_only(self):
        for name in ["alimeeting_test","ami_cleaned_test","renji_easy","renji_hard"]:
            self.assertTrue(excluded(name))
        for name in ["ramc_test","wenetspeech_test_meeting","commonvoice_yue_test"]:
            self.assertFalse(excluded(name))

    def test_family_assignment_is_balanced_and_independent_of_dialect_order(self):
        ids=[f"dialect{d}-utterance{i}" for d in range(9) for i in range(120)]
        a=family_assignments(ids,42,"kespeech")
        self.assertEqual(a,family_assignments(list(reversed(ids)),42,"kespeech"))
        counts={f:list(a.values()).count(f) for f in set(a.values())}
        self.assertLessEqual(max(counts.values())-min(counts.values()),1)
        for d in range(9):
            self.assertEqual(len({a[k] for k in ids if k.startswith(f"dialect{d}-")}),6)

    def test_long_audio_is_not_truncated_and_snr_survives_encoding(self):
        n=16000*41+73
        t=np.arange(n,dtype=np.float32)/16000
        signal=.3*np.sin(t*2*np.pi*181)
        noise=np.random.default_rng(24).standard_normal(n).astype(np.float32)
        encoded,normalization,gain,snrs=render_variants(signal,noise)
        self.assertGreater(normalization,0)
        self.assertGreater(gain,0)
        for name,data in encoded.items():
            wav,sr=sf.read(io.BytesIO(data),dtype="float32")
            self.assertEqual(sr,16000)
            self.assertEqual(len(wav),n)
        for snr in [20,10,0]:
            self.assertLess(abs(snrs[f"noise_snr{snr}"]-snr),.01)
        with self.assertRaises(ValueError):
            render_variants(np.zeros(160),np.ones(160))

    def test_invalid_source_requires_explicit_quarantine(self):
        self.assertEqual(invalid_speech_reason(np.zeros(100)),"silent_pcm")
        self.assertEqual(invalid_speech_reason(np.array([np.nan])),"nonfinite_pcm")
        self.assertEqual(invalid_speech_reason(np.array([])),"empty_pcm")
        self.assertIsNone(invalid_speech_reason(np.array([.01,-.01])))

    def test_sharded_manifest_round_trip_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/"source"
            source.mkdir()
            (source/"manifest.jsonl").write_text(json.dumps(dict(id="u1",reference="hello world"))+"\n")
            write_json(root/"protocol.json",dict(
                sources={"demo":dict(n_utts=1,webdataset=str(source),manifest_sha256=sha256_file(source/"manifest.jsonl"))},
                noise_manifest_sha256={"test":"frozen-noise"},smoke_limit=0))
            signal=np.sin(np.arange(16001,dtype=np.float32)*.05)*.1
            noise=np.random.default_rng(2).standard_normal(len(signal)).astype(np.float32)
            encoded,normalization,gain,measured=render_variants(signal,noise)
            trace=dict(asset_id="clip",path="/fixture.wav",group_id="heldout-recording",
                       source_family="esc50",num_samples=len(signal))
            for condition in CONDITIONS:
                name="demo__"+condition
                writer=ShardWriter(root/"webdataset",name,1,0)
                corruption={} if condition=="clean" else dict(
                    noise=trace,snr_db=int(condition.removeprefix("noise_snr")),actual_encoded_snr_db=measured[condition])
                writer.add("u1",encoded[condition],dict(
                    reference="hello world",condition=condition,split="test",source_id="test/demo/u1",
                    speech_origin={"utterance_id":"u1"},shared_gain=gain,normalization_gain=normalization,
                    frames=len(signal),duration=len(signal)/16000,noise_manifest_sha256="frozen-noise",trace=corruption))
                writer.finish()
            allowed=[dict(asset_id="clip",path=Path("/fixture.wav"),group_id="heldout-recording")]
            with patch("tools.build_full_noise_benchmark.read_asset_manifest",return_value=allowed):
                result=verify_dataset(root,"demo",root)
                self.assertTrue(result["passed"])
                self.assertEqual(result["rendered_utterances"],4)
                import tarfile
                zeros=np.zeros(1600,dtype=np.float32)
                buffer=io.BytesIO();sf.write(buffer,zeros,16000,format="FLAC",subtype="PCM_16")
                with tarfile.open(source/"source.tar","w") as archive:
                    info=tarfile.TarInfo("u2.flac");info.size=len(buffer.getvalue())
                    archive.addfile(info,io.BytesIO(buffer.getvalue()))
                with (source/"manifest.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(id="u2",reference="silent source",shard="source.tar"))+"\n")
                protocol=json.loads((root/"protocol.json").read_text())
                protocol["sources"]["demo"]["n_utts"]=2
                protocol["sources"]["demo"]["manifest_sha256"]=sha256_file(source/"manifest.jsonl")
                write_json(root/"protocol.json",protocol)
                (root/"quarantine").mkdir()
                quarantined=[dict(id="u2",reference="silent source",reason="silent_pcm",
                                  source_float32_sha256=digest_bytes(zeros.tobytes()))]
                write_json(root/"quarantine/demo.json",quarantined)
                audited=verify_dataset(root,"demo",root)
                self.assertEqual(audited["source_speech_utterances"],2)
                self.assertEqual(audited["evaluated_speech_utterances"],1)
                self.assertEqual(audited["invalid_source_utterances"],1)
                self.assertEqual(audited["unaccounted_source_utterances"],0)
                quarantined[0]["reason"]="nonfinite_pcm"
                write_json(root/"quarantine/demo.json",quarantined)
                with self.assertRaises(AssertionError): verify_dataset(root,"demo",root)
                quarantined[0]["reason"]="silent_pcm"
                write_json(root/"quarantine/demo.json",quarantined)
                manifest=root/"webdataset/demo__noise_snr10/manifest.jsonl"
                manifest.write_text(manifest.read_text().replace("hello world","different text"))
                with self.assertRaises(AssertionError):
                    verify_dataset(root,"demo",root)


if __name__=="__main__":
    unittest.main()
