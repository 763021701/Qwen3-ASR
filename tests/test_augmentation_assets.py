"""Source leakage, split allowlists and evaluation provenance integration tests."""
import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from finetuning.augmentation.assets import read_asset_manifest
from finetuning.augmentation.service import SimulationConfig, SimulationService
from finetuning.augmentation.simulator import NoisePool, RIRPool
from finetuning.data_pipeline.spec import load_data_config
from tools.materialize_robust_eval import materialize_robust_eval
from tools.split_augmentation_assets import (
    audio_fingerprint, build, real_room, verify,
)


def write_wav(path, seed, sr=16000, seconds=0.1):
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.random.default_rng(seed).normal(0, 0.1, int(sr * seconds))
    sf.write(path, audio, sr, subtype="PCM_16")
    return path


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("augmentation_splits")
    noise = root / "noise"
    rir = root / "rir"
    meta = root / "esc50.csv"
    meta.write_text(
        "filename,fold,target,category,esc10,src_file,take\n"
        "1-111-A-0.wav,1,0,dog,True,111,A\n"
        "4-777-A-0.wav,4,0,dog,True,777,A\n"
        "5-777-B-1.wav,5,1,rooster,True,777,B\n"
        "4-888-A-0.wav,4,0,dog,True,888,A\n"
        "5-999-A-0.wav,5,0,dog,True,999,A\n")
    for i, name in enumerate(("1-111-A-0.wav", "4-777-A-0.wav",
                              "5-777-B-1.wav", "4-888-A-0.wav",
                              "5-999-A-0.wav")):
        write_wav(noise / "ESC-50" / name, i, seconds=0.3)
    write_wav(noise / "DNS-Challenge" / "door_Freesound_validated_777_0.wav", 13)
    write_wav(noise / "DNS-Challenge" / "mislabeled_door_Freesound_validated_777_1.wav", 14)
    for i in range(24):
        write_wav(noise / "DNS-Challenge" / f"video{i:06d}.wav", i + 30)
    for i in range(8):
        write_wav(noise / "musan/noise/free-sound" / f"noise-{i}.wav", 100 + i)
    real = rir / "RIRS_NOISES/real_rirs_isotropic_noises"
    real.mkdir(parents=True)
    names = ["RWCP_type1_rir_circle_e1a_imp000.wav",
             "RWCP_type1_rir_circle_e1b_imp010.wav"]
    for i, name in enumerate(names):
        write_wav(real / name, 200 + i)
    (real / "rir_list").write_text("".join(
        f"--rir-id {i} --room-id fine-{i} RIRS_NOISES/real_rirs_isotropic_noises/{name}\n"
        for i, name in enumerate(names)))
    for size in ("smallroom", "mediumroom", "largeroom"):
        for room in range(10):
            for index in range(2):
                write_wav(rir / f"RIRS_NOISES/simulated_rirs/{size}/Room{room:03d}/"
                          f"Room{room:03d}-{index:05d}.wav", 500 + room * 3 + index)
    write_wav(rir / "DNS-Challenge/SLR26/simulated_rirs_48k/smallroom/Room000/"
              "Room000-00000.wav", 500, sr=48000)
    output = root / "splits"
    build(argparse.Namespace(
        output_dir=str(output), noise_dir=str(noise), rir_dir=str(rir),
        esc_metadata=str(meta), workers=1, seed=42, rehash=False))
    return root, output


def all_rows(output, kind):
    return [json.loads(line) for split in ("train", "dev", "test")
            for line in (output / f"{kind}.{split}.jsonl").read_text().splitlines()]


def test_source_groups_override_folds_and_categories(corpus):
    _, output = corpus
    rows = [r for r in all_rows(output, "noise") if "freesound:777" in r["origin_ids"]]
    assert len(rows) == 4
    assert {r["split"] for r in rows} == {"test"}
    assert len({r["group_id"] for r in rows}) == 1
    assert verify(output)["passed"]


def test_rir_sample_rate_aliases_and_room_positions_stay_together(corpus):
    _, output = corpus
    rows = all_rows(output, "rir")
    alias = next(r for r in rows if r["origin_ids"] == [
        "slr26:smallroom:Room000:Room000-00000"])
    assert len(alias["aliases"]) == 1
    assert alias["sample_rate"] == 16000
    assert alias["aliases"][0]["sample_rate"] == 48000
    real = [r for r in rows if r["source"] == "real"]
    assert len({r["group_id"] for r in real}) == 1
    assert real_room("air_type1_air_binaural_office_1_2") == real_room(
        "air_type1_air_phone_office_hhp")
    assert real_room("RWCP_type4_rir_m10l")[1] is True


def test_cropped_identical_audio_has_shared_window_fingerprint(tmp_path):
    path = write_wav(tmp_path / "full.wav", 1, seconds=10)
    audio, sr = sf.read(path, dtype="float32")
    sf.write(tmp_path / "crop.wav", audio[2 * sr:9 * sr], sr, subtype="PCM_16")
    a = audio_fingerprint((str(path), True))
    b = audio_fingerprint((str(tmp_path / "crop.wav"), True))
    assert a["pcm_sha256"] != b["pcm_sha256"]
    assert set(a["window_hashes"]) & set(b["window_hashes"])


def test_allowlists_fail_closed_and_never_load_aliases(corpus, tmp_path):
    _, output = corpus
    manifest = output / "noise.train.jsonl"
    rows = read_asset_manifest(str(manifest), "noise", "train")
    pool = NoisePool(manifest_path=str(manifest), split="train")
    assert set(pool.files) == {r["path"] for r in rows}
    for i in range(20):
        pool.sample(160, np.random.default_rng(i))
        assert pool.last_sample["group_id"] in {r["group_id"] for r in rows}
    with pytest.raises(ValueError, match="split"):
        NoisePool(manifest_path=str(output / "noise.dev.jsonl"), split="train")
    with pytest.raises(FileNotFoundError):
        NoisePool(manifest_path=str(tmp_path / "missing.jsonl"), split="train")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError, match="Empty"):
        NoisePool(manifest_path=str(empty), split="train")
    with pytest.raises(ValueError, match="not both"):
        NoisePool(str(tmp_path), manifest_path=str(manifest), split="train")
    rirs = RIRPool(manifest_path=str(output / "rir.train.jsonl"), split="train")
    assert len(rirs.files) == len(read_asset_manifest(
        str(output / "rir.train.jsonl"), "rir", "train"))


def test_independent_audit_detects_origin_leakage(corpus):
    _, output = corpus
    path = output / "noise.dev.jsonl"
    original = path.read_text()
    row = json.loads(original.splitlines()[0])
    train = json.loads((output / "noise.train.jsonl").read_text().splitlines()[0])
    row["origin_ids"].append(train["origin_ids"][0])
    lines = original.splitlines()
    lines[0] = json.dumps(row)
    try:
        path.write_text("\n".join(lines) + "\n")
        with pytest.raises(ValueError, match="Cross-split leakage"):
            verify(output)
    finally:
        path.write_text(original)


def test_config_resolves_manifest_paths(corpus, tmp_path):
    _, output = corpus
    config = tmp_path / "data.yaml"
    config.write_text(
        "data:\n  sources:\n    - name: train\n      backend: jsonl\n"
        "      path: train.jsonl\n  augmentation:\n    simulation:\n"
        "      enabled: true\n      spool_dir: spool\n      asset_split: train\n"
        f"      noise_manifest: {output / 'noise.train.jsonl'}\n"
        "      reverb_mode: real\n"
        f"      rir_manifest: {output / 'rir.train.jsonl'}\n")
    spec = load_data_config(str(config))
    simulation = SimulationConfig.from_spec(spec.augmentation.simulation)
    assert simulation.asset_split == "train"
    assert simulation.noise_dir is None
    assert len(RIRPool(manifest_path=simulation.rir_manifest, split="train").files) > 0


def test_eval_records_exact_asset_provenance_and_rejects_wrong_split(corpus, tmp_path):
    _, output = corpus
    clean = write_wav(tmp_path / "clean.wav", 0, seconds=1)
    kwargs = dict(scenarios=("far_field+noise",), severity=0.5,
                  noise_manifest=str(output / "noise.test.jsonl"),
                  rir_manifest=str(output / "rir.test.jsonl"),
                  reverb_mode="real", asset_split="test")
    path = materialize_robust_eval(
        [{"audio": str(clean), "text": "language English<asr_text>test"}],
        str(tmp_path / "eval"), **kwargs)
    row = json.loads(path.read_text().splitlines()[0])
    assert row["augmentation_split"] == "test"
    assert row["noise_asset_id"]
    assert row["rir_asset_id"]
    assert row["noise_start_sample"] >= 0
    assert len(row["noise_manifest_sha256"]) == 64
    with pytest.raises(ValueError, match="split"):
        materialize_robust_eval(
            [{"audio": str(clean), "text": "test"}], str(tmp_path / "bad"),
            **{**kwargs, "asset_split": "train"})


def test_spawn_service_can_consume_manifest_assets(corpus, tmp_path):
    _, output = corpus
    config = SimulationConfig(
        spool_dir=str(tmp_path / "spool"), asset_split="train",
        noise_manifest=str(output / "noise.train.jsonl"),
        rir_manifest=str(output / "rir.train.jsonl"),
        reverb_mode="real", scenarios=("far_field+noise",),
        num_simulators=1, ready_queue_size=4, clean_queue_size=4)
    sample = dict(audio=np.ones(1600, dtype=np.float32) * 0.05,
                  target="language English<asr_text>test", sample_seed=42)
    with SimulationService([sample], config) as service:
        result = list(service.dataset)
    assert len(result) == 1
    assert result[0]["augmentation_split"] == "train"
    assert result[0]["noise_asset_id"]
    assert result[0]["rir_asset_id"]


def test_training_rejects_dev_or_test_augmentation_config(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from finetuning import data_pipeline
    from finetuning.qwen3_asr_sft import build_training_data
    from finetuning.data_pipeline.spec import parse_data_config
    for split in ("dev", "test"):
        config = parse_data_config({
            "sources": [{"name": "train", "backend": "jsonl", "path": "train.jsonl"}],
            "augmentation": {"simulation": {
                "enabled": True, "spool_dir": str(tmp_path),
                "noise_manifest": "noise.jsonl", "asset_split": split}}})
        monkeypatch.setattr(data_pipeline, "build_data_pipeline",
                            lambda *args: SimpleNamespace(config=config))
        with pytest.raises(ValueError, match="asset_split=train"):
            build_training_data(SimpleNamespace(data_config="data.yaml"))


def test_split_generation_is_reproducible_and_frozen(corpus):
    root, output = corpus
    other = root / "repeated"
    build(argparse.Namespace(
        output_dir=str(other), noise_dir=str(root / "noise"), rir_dir=str(root / "rir"),
        esc_metadata=str(root / "esc50.csv"), workers=1, seed=42, rehash=False))
    for kind in ("noise", "rir"):
        for split in ("train", "dev", "test"):
            assert (output / f"{kind}.{split}.jsonl").read_bytes() == (
                other / f"{kind}.{split}.jsonl").read_bytes()
    with pytest.raises(ValueError, match="Frozen split already exists"):
        build(argparse.Namespace(output_dir=str(output)))
