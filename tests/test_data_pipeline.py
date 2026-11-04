# coding=utf-8
"""Tests for the local JSONL / WebDataset data pipeline (P1/P2)."""

import io
import itertools
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from finetuning.data_pipeline import (
    build_data_pipeline,
    build_eval_dataset,
    load_data_config,
    parse_data_config,
)
from finetuning.data_pipeline.datasets import (
    ManifestDataset,
    WeightedSourceMixer,
    iter_webdataset_shard,
    load_manifest_rows,
)
from finetuning.data_pipeline.index import index_source, split_shards_for_worker
from finetuning.data_pipeline.labels import make_target
from finetuning.data_pipeline.mixer import select_source_shards
from finetuning.data_pipeline.spec import Fields, MixtureSpec, SourceSpec

SR = 16000


def _wav(seconds: float = 0.2, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(SR * seconds), dtype=np.float32) / SR
    return (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _wav_bytes(seconds: float = 0.2) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, _wav(seconds), SR, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _write_wav(path: Path, seconds: float = 0.2) -> None:
    sf.write(str(path), _wav(seconds), SR, subtype="PCM_16")


def _add_tar_member(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(data))


def _write_tar(path: Path, samples, audio_ext: str = "wav") -> None:
    with tarfile.open(path, "w") as archive:
        for sample in samples:
            audio = sample["audio"]
            if isinstance(audio, np.ndarray):
                buffer = io.BytesIO()
                sf.write(buffer, audio, SR, format="WAV", subtype="PCM_16")
                audio = buffer.getvalue()
            _add_tar_member(archive, f"{sample['key']}.{audio_ext}", audio)
            _add_tar_member(archive, f"{sample['key']}.txt", sample["text"].encode("utf-8"))
            meta = sample.get("meta", {})
            _add_tar_member(
                archive,
                f"{sample['key']}.json",
                json.dumps(meta, ensure_ascii=False).encode("utf-8"),
            )


def _write_manifest(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class LabelTest(unittest.TestCase):
    def test_template(self):
        self.assertEqual(
            make_target("你好", "Chinese", "language {language}<asr_text>{text}"),
            "language Chinese<asr_text>你好",
        )
        self.assertEqual(
            make_target("你好", None, "language {language}<asr_text>{text}"),
            "language None<asr_text>你好",
        )

    def test_raw(self):
        self.assertEqual(make_target("language English<asr_text>hi", None, "raw"),
                         "language English<asr_text>hi")

    def test_rejects_double_prefix_and_empty(self):
        with self.assertRaises(ValueError):
            make_target("language English<asr_text>hi", "English",
                        "language {language}<asr_text>{text}")
        with self.assertRaises(ValueError):
            make_target("", "English", "language {language}<asr_text>{text}")


class ConfigTest(unittest.TestCase):
    def test_load_nested_yaml_resolves_relative_path(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_wav(root / "a.wav")
            _write_manifest(root / "train.jsonl", [
                {"audio": "a.wav", "text": "你好"},
            ])
            config_path = root / "data.yaml"
            config_path.write_text(
                "data:\n"
                "  mode: auto\n"
                "  sources:\n"
                "    - name: zh\n"
                "      backend: jsonl\n"
                "      path: train.jsonl\n"
                "      language: Chinese\n"
                "      weight: 1.0\n",
                encoding="utf-8",
            )
            config = load_data_config(str(config_path))
            self.assertEqual(config.sources[0].path, str((root / "train.jsonl").resolve()))
            self.assertEqual(config.sources[0].language, "Chinese")

    def test_modelscope_requires_cache_dir_and_streaming_mode_is_allowed(self):
        with self.assertRaises(ValueError):
            parse_data_config({
                "sources": [{
                    "name": "remote", "backend": "webdataset", "provider": "modelscope",
                    "repo_id": "owner/data", "path_in_repo": "shards",
                }]
            })
        config = parse_data_config({
            "mode": "streaming",
            "sources": [{
                "name": "remote", "backend": "webdataset", "provider": "modelscope",
                "repo_id": "owner/data", "path_in_repo": "shards",
            }],
            "cache": {"dir": "/tmp/qwen3_asr_cache", "max_gb": 10},
        })
        self.assertEqual(config.mode, "streaming")
        self.assertEqual(config.cache.dir, "/tmp/qwen3_asr_cache")
        self.assertEqual(config.sources[0].provider, "modelscope")

    def test_rejects_bad_template(self):
        with self.assertRaises(ValueError):
            parse_data_config({
                "sources": [{
                    "name": "x", "backend": "jsonl", "path": "/tmp/x.jsonl",
                    "text_template": "no placeholder",
                }]
            })


class ManifestPipelineTest(unittest.TestCase):
    def _make_config(self, root: Path):
        for index in range(8):
            _write_wav(root / f"a{index}.wav")
        for index in range(4):
            _write_wav(root / f"b{index}.wav")
        _write_manifest(root / "a.jsonl", [
            {"audio": str(root / f"a{index}.wav"), "text": f"a{index}"}
            for index in range(8)
        ])
        _write_manifest(root / "b.jsonl", [
            {"audio": str(root / f"b{index}.wav"), "text": f"b{index}"}
            for index in range(4)
        ])
        return {
            "sources": [
                {
                    "name": "a", "backend": "jsonl", "path": str(root / "a.jsonl"),
                    "language": "Chinese", "weight": 0.75,
                    "text_template": "language {language}<asr_text>{text}",
                },
                {
                    "name": "b", "backend": "jsonl", "path": str(root / "b.jsonl"),
                    "language": "Chinese", "weight": 0.25,
                    "text_template": "language {language}<asr_text>{text}",
                },
            ],
            "mixture": {"samples_per_epoch": 40, "seed": 7},
        }

    def test_map_style_dataset_and_quota_sampler(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pipeline = build_data_pipeline(self._make_config(root))
            self.assertEqual(pipeline.mode, "offline_jsonl")
            self.assertFalse(pipeline.is_streaming)
            self.assertEqual(len(pipeline.dataset), 12)
            self.assertEqual(len(pipeline.sampler), 40)

            item = pipeline.dataset[0]
            self.assertEqual(item["source"], "a")
            self.assertTrue(Path(item["audio"]).is_file())
            self.assertTrue(item["target"].startswith("language Chinese<asr_text>"))

            indices = list(pipeline.sampler)
            self.assertEqual(len(indices), 40)
            self.assertEqual(sum(index < 8 for index in indices), 30)
            self.assertEqual(sum(index >= 8 for index in indices), 10)

            pipeline.sampler.set_epoch(0)
            first = list(pipeline.sampler)
            pipeline.sampler.set_epoch(0)
            self.assertEqual(first, list(pipeline.sampler))
            pipeline.sampler.set_epoch(1)
            self.assertNotEqual(first, list(pipeline.sampler))

    def test_build_eval_dataset(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_wav(root / "a.wav")
            _write_manifest(root / "dev.jsonl", [{
                "audio": str(root / "a.wav"),
                "text": "language Chinese<asr_text>你好",
            }])
            dataset = build_eval_dataset(str(root / "dev.jsonl"))
            self.assertEqual(len(dataset), 1)
            row = dataset[0]
            self.assertTrue(row["target"].startswith("language Chinese<asr_text>"))
            self.assertTrue(Path(row["audio"]).is_file())

    def test_manifest_dataset_checks_audio(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_manifest(root / "train.jsonl", [
                {"audio": str(root / "missing.wav"), "text": "x"},
            ])
            source = SourceSpec(
                name="x", backend="jsonl", path=str(root / "train.jsonl"),
                language="Chinese",
            )
            dataset = ManifestDataset(source, load_manifest_rows(str(root / "train.jsonl")))
            with self.assertRaises(FileNotFoundError):
                _ = dataset[0]

    def test_mixed_jsonl_and_webdataset_sources(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_wav(root / "a.wav")
            _write_manifest(root / "train.jsonl", [
                {"audio": str(root / "a.wav"), "text": "你好"},
            ])
            _write_tar(root / "shard-00000.tar", [{
                "key": "s0",
                "audio": _wav(),
                "text": "世界",
                "meta": {"duration": 0.2},
            }])
            config = {
                "sources": [
                    {
                        "name": "jsonl", "backend": "jsonl",
                        "path": str(root / "train.jsonl"),
                        "language": "Chinese", "weight": 0.5,
                    },
                    {
                        "name": "wd", "backend": "webdataset",
                        "path": str(root / "shard-00000.tar"),
                        "split": "train", "language": "Chinese", "weight": 0.5,
                        "fields": {"audio": "wav", "text": "txt", "meta": "json"},
                    },
                ],
                "mixture": {"seed": 5, "sample_shuffle_buffer": 1},
            }
            pipeline = build_data_pipeline(config)
            self.assertTrue(pipeline.is_streaming)
            samples = list(itertools.islice(pipeline.dataset, 30))
            self.assertEqual({sample["source"] for sample in samples}, {"jsonl", "wd"})


class WeightedMixerTest(unittest.TestCase):
    def test_weighted_source_mix_ratio(self):
        mixer = WeightedSourceMixer(
            [
                itertools.cycle([{"source": "a"}]),
                itertools.cycle([{"source": "b"}]),
            ],
            weights=[0.75, 0.25],
            seed=5,
        )
        samples = list(itertools.islice(mixer, 4000))
        count_a = sum(sample["source"] == "a" for sample in samples)
        self.assertAlmostEqual(count_a / len(samples), 0.75, delta=0.04)


class WebDatasetPipelineTest(unittest.TestCase):
    def _make_root(self, root: Path, with_info: bool = True) -> Path:
        shards = []
        for shard_index in range(2):
            samples = []
            for sample_index in range(3):
                key = f"s{shard_index}_{sample_index}"
                samples.append({
                    "key": key,
                    "audio": _wav(),
                    "text": f"文本{key}",
                    "meta": {"duration": 0.2, "language": "Chinese"},
                })
            shard = root / f"shard-{shard_index:05d}.tar"
            _write_tar(shard, samples)
            shards.append(shard)
        if with_info:
            info = {
                "splits": {
                    "train": {
                        "samples": 6,
                        "hours": 0.001,
                        "shards": [
                            {"file": shard.name, "samples": 3, "hours": 0.0005}
                            for shard in shards
                        ],
                    }
                }
            }
            (root / "dataset_info.json").write_text(
                json.dumps(info), encoding="utf-8"
            )
        return root

    def _config(self, root: Path):
        return {
            "sources": [{
                "name": "wd",
                "backend": "webdataset",
                "path": str(root),
                "split": "train",
                "language": "Chinese",
                "text_template": "language {language}<asr_text>{text}",
                "fields": {"audio": "wav", "text": "txt", "meta": "json"},
                "weight": 1.0,
            }],
            "mixture": {"seed": 11, "sample_shuffle_buffer": 2},
        }

    def test_streaming_pipeline_and_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._make_root(Path(td))
            pipeline = build_data_pipeline(self._config(root))
            self.assertEqual(pipeline.mode, "local_webdataset")
            self.assertTrue(pipeline.is_streaming)
            self.assertEqual(len(pipeline.shards), 2)
            samples = list(itertools.islice(pipeline.dataset, 6))
            self.assertEqual(len(samples), 6)
            for sample in samples:
                self.assertIsInstance(sample["audio"], np.ndarray)
                self.assertGreater(sample["audio"].size, 0)
                self.assertTrue(sample["target"].startswith("language Chinese<asr_text>"))
                self.assertEqual(sample["language"], "Chinese")
                self.assertAlmostEqual(sample["duration"], 0.2)
            self.assertEqual(len({sample["key"] for sample in samples}), 6)

    def test_streaming_is_deterministic(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._make_root(Path(td))
            first = [item["key"] for item in itertools.islice(build_data_pipeline(self._config(root)).dataset, 8)]
            second = [item["key"] for item in itertools.islice(build_data_pipeline(self._config(root)).dataset, 8)]
            self.assertEqual(first, second)

    def test_index_without_dataset_info_scans_tar(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._make_root(Path(td), with_info=False)
            source = SourceSpec(
                name="wd", backend="webdataset", path=str(root), split="train",
                language="Chinese", fields=Fields(audio="wav", text="txt", meta="json"),
            )
            index = index_source(source)
            self.assertEqual(len(index.shards), 2)
            self.assertEqual(index.total_samples, 6)
            self.assertAlmostEqual(index.total_hours, 6 * 0.2 / 3600.0)

    def test_shard_selection_repeats_to_target_hours(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._make_root(Path(td))
            source = SourceSpec(
                name="wd", backend="webdataset", path=str(root), split="train",
                language="Chinese", fields=Fields(audio="wav", text="txt", meta="json"),
            )
            index = index_source(source)
            mixture = MixtureSpec(seed=3, repeat_small_sources=True)
            # Each shard reports 0.0005h, so a 0.0015h target needs three shards.
            selected = select_source_shards(source, index.shards, 0.0015, mixture, pass_index=0)
            self.assertEqual(len(selected), 3)
            self.assertAlmostEqual(sum(shard.hours for shard in selected), 0.0015)

    def test_worker_shard_assignment(self):
        shards = list(range(5))
        self.assertEqual(split_shards_for_worker(shards, 0, 2), [0, 2, 4])
        self.assertEqual(split_shards_for_worker(shards, 1, 2), [1, 3])
        self.assertEqual(split_shards_for_worker(shards, 0, 0), shards)

    def test_incomplete_sample_is_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            tar_path = Path(td) / "shard.tar"
            with tarfile.open(tar_path, "w") as archive:
                _add_tar_member(archive, "ok.wav", _wav_bytes())
                _add_tar_member(archive, "ok.txt", b"hello")
                _add_tar_member(archive, "ok.json", b'{"duration": 0.2}')
                _add_tar_member(archive, "bad.wav", _wav_bytes())
                _add_tar_member(archive, "bad.json", b'{"duration": 0.2}')
            source = SourceSpec(
                name="wd", backend="webdataset", path=str(tar_path), split="train",
                language="Chinese", fields=Fields(audio="wav", text="txt", meta="json"),
            )
            samples = list(iter_webdataset_shard(tar_path, source))
            self.assertEqual([sample["key"] for sample in samples], ["ok"])


class CollatorIntegrationTest(unittest.TestCase):
    def test_accepts_array_audio_and_missing_prefix(self):
        from finetuning import qwen3_asr_sft as sft

        processor = SimpleNamespace(
            tokenizer=SimpleNamespace(eos_token="EOS"),
            apply_chat_template=lambda *args, **kwargs: ["PREFIX"],
        )
        collator = sft.DataCollatorForQwen3ASRFinetuning(processor)
        features = [{
            "audio": np.zeros(8, dtype=np.float32),
            "target": "language Chinese<asr_text>你好",
            "prompt": "",
        }]
        inputs = {
            "input_ids": torch.tensor([[0, 1, 2, 3]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1]]),
        }
        prefix = {"attention_mask": torch.tensor([[1, 1]])}
        with patch.object(sft, "load_audio", side_effect=AssertionError("should not load path")), \
             patch.object(sft, "build_processor_batch_inputs", return_value=(inputs, prefix)) as build:
            collator(features)
        self.assertEqual(
            build.call_args.args[1],
            ["PREFIXlanguage Chinese<asr_text>你好EOS"],
        )

    def test_curriculum_uses_none_prefix_when_target_none_missing(self):
        from finetuning import qwen3_asr_sft as sft

        processor = SimpleNamespace(
            tokenizer=SimpleNamespace(eos_token="EOS"),
            apply_chat_template=lambda *args, **kwargs: ["PREFIX"],
        )
        collator = sft.DataCollatorForQwen3ASRFinetuning(
            processor, curriculum=True, curriculum_switch_epoch=0.0
        )
        collator.current_epoch = 1.0
        features = [{
            "audio": np.zeros(8, dtype=np.float32),
            "target": "language Chinese<asr_text>你好",
            "prompt": "",
        }]
        inputs = {
            "input_ids": torch.tensor([[0, 1, 2, 3]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1]]),
        }
        prefix = {"attention_mask": torch.tensor([[1, 1]])}
        with patch.object(sft, "build_processor_batch_inputs", return_value=(inputs, prefix)) as build:
            collator(features)
        self.assertEqual(
            build.call_args.args[1],
            ["PREFIXlanguage None<asr_text>你好EOS"],
        )

    def test_trainer_uses_custom_sampler(self):
        from finetuning.qwen3_asr_sft import CastFloatInputsTrainer

        sampler = object()
        trainer = object.__new__(CastFloatInputsTrainer)
        trainer.custom_train_sampler = sampler
        self.assertIs(trainer._get_train_sampler(), sampler)

    def test_build_training_pipeline_streaming_without_curriculum_arg(self):
        from finetuning.qwen3_asr_sft import build_training_pipeline

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_tar(root / "shard-00000.tar", [{
                "key": "s0",
                "audio": _wav(),
                "text": "你好",
                "meta": {"duration": 0.2},
            }])
            config_path = root / "data.yaml"
            config_path.write_text(
                "data:\n"
                "  sources:\n"
                "    - name: wd\n"
                "      backend: webdataset\n"
                f"      path: {root}\n"
                "      split: train\n"
                "      language: Chinese\n"
                "      fields: {audio: wav, text: txt, meta: json}\n"
                "  mixture: {seed: 1, sample_shuffle_buffer: 1}\n",
                encoding="utf-8",
            )
            args = SimpleNamespace(data_config=str(config_path), max_steps=1)
            pipeline = build_training_pipeline(args)
            self.assertTrue(pipeline.is_streaming)


if __name__ == "__main__":
    unittest.main()
