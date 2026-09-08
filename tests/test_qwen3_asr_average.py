"""Unit tests for finetuning.qwen3_asr_average."""

import json
import os
import subprocess
import sys

import torch
from safetensors.torch import save_file

from finetuning.qwen3_asr_average import (
    average_checkpoints,
    average_state_dicts,
    copy_sidecar_files,
    fill_tied_weights,
    list_safetensor_files,
    load_weight_tensors,
)


def _write_ckpt(path, tensors, sidecars=None, config=None):
    os.makedirs(path, exist_ok=True)
    save_file(tensors, os.path.join(path, "model.safetensors"), metadata={"format": "pt"})
    cfg = config or {"architectures": ["Qwen3ASRForConditionalGeneration"], "model_type": "qwen3_asr"}
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    with open(os.path.join(path, "generation_config.json"), "w", encoding="utf-8") as f:
        json.dump({"eos_token_id": 1, "do_sample": False}, f)
    with open(os.path.join(path, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump({"model_max_length": 32768}, f)
    with open(os.path.join(path, "preprocessor_config.json"), "w", encoding="utf-8") as f:
        json.dump({"sampling_rate": 16000}, f)
    # Trainer-only artifacts that must NOT be copied into the averaged model.
    with open(os.path.join(path, "optimizer.pt"), "wb") as f:
        f.write(b"fake-optimizer")
    with open(os.path.join(path, "trainer_state.json"), "w", encoding="utf-8") as f:
        json.dump({"global_step": 250}, f)
    if sidecars:
        for name, content in sidecars.items():
            with open(os.path.join(path, name), "w", encoding="utf-8") as f:
                f.write(content)
    return path


def test_average_state_dicts_linear_mix():
    state_a = {
        "w": torch.tensor([[0.0, 10.0], [20.0, 30.0]], dtype=torch.float32),
        "ids": torch.tensor([1, 2, 3], dtype=torch.int64),
    }
    state_b = {
        "w": torch.tensor([[10.0, 0.0], [0.0, 10.0]], dtype=torch.float32),
        "ids": torch.tensor([1, 2, 3], dtype=torch.int64),
    }
    out = average_state_dicts(state_a, state_b, alpha=0.3)
    expected = 0.3 * state_a["w"] + 0.7 * state_b["w"]
    assert torch.allclose(out["w"], expected)
    assert torch.equal(out["ids"], state_a["ids"])


def test_average_state_dicts_alpha_endpoints_are_copies():
    state_a = {"w": torch.tensor([1.0, 2.0])}
    state_b = {"w": torch.tensor([9.0, 8.0])}
    assert torch.equal(average_state_dicts(state_a, state_b, 1.0)["w"], state_a["w"])
    assert torch.equal(average_state_dicts(state_a, state_b, 0.0)["w"], state_b["w"])


def test_average_state_dicts_rejects_key_and_shape_mismatch():
    a = {"w": torch.zeros(2), "extra": torch.zeros(1)}
    b = {"w": torch.zeros(2)}
    try:
        average_state_dicts(a, b, 0.5)
        assert False, "key mismatch should raise"
    except ValueError as exc:
        assert "keys do not match" in str(exc)

    a = {"w": torch.zeros(2, 2)}
    b = {"w": torch.zeros(3)}
    try:
        average_state_dicts(a, b, 0.5)
        assert False, "shape mismatch should raise"
    except ValueError as exc:
        assert "Shape mismatch" in str(exc)


def test_average_state_dicts_rejects_differing_integer_tensors():
    a = {"ids": torch.tensor([1, 2])}
    b = {"ids": torch.tensor([1, 9])}
    try:
        average_state_dicts(a, b, 0.5)
        assert False, "integer mismatch should raise"
    except ValueError as exc:
        assert "Non-floating" in str(exc)


def test_average_state_dicts_bf16_roundtrip():
    a = {"w": torch.tensor([1.0, 2.0], dtype=torch.bfloat16)}
    b = {"w": torch.tensor([3.0, 4.0], dtype=torch.bfloat16)}
    out = average_state_dicts(a, b, 0.5)
    assert out["w"].dtype == torch.bfloat16
    expected = (0.5 * a["w"].float() + 0.5 * b["w"].float()).to(torch.bfloat16)
    assert torch.equal(out["w"], expected)


def test_average_checkpoints_writes_complete_model(tmp_path):
    dir_a = _write_ckpt(
        str(tmp_path / "A"),
        {"layer.weight": torch.ones(4, 4), "layer.bias": torch.zeros(4)},
        sidecars={"vocab.json": '{"a": 1}', "merges.txt": "a b"},
    )
    dir_b = _write_ckpt(
        str(tmp_path / "B"),
        {"layer.weight": torch.full((4, 4), 3.0), "layer.bias": torch.ones(4)},
    )
    out = str(tmp_path / "C")
    average_checkpoints(dir_a, dir_b, out, alpha=0.25)

    state, metadata = load_weight_tensors(out)
    assert metadata == {"format": "pt"}
    expected_w = 0.25 * torch.ones(4, 4) + 0.75 * torch.full((4, 4), 3.0)
    expected_b = 0.25 * torch.zeros(4) + 0.75 * torch.ones(4)
    assert torch.allclose(state["layer.weight"], expected_w)
    assert torch.allclose(state["layer.bias"], expected_b)

    copied = set(os.listdir(out))
    assert "model.safetensors" in copied
    assert "config.json" in copied
    assert "generation_config.json" in copied
    assert "tokenizer_config.json" in copied
    assert "preprocessor_config.json" in copied
    assert "vocab.json" in copied
    assert "merges.txt" in copied
    # Trainer artifacts from A must not leak into C.
    assert "optimizer.pt" not in copied
    assert "trainer_state.json" not in copied


def test_copy_sidecar_skips_weights_and_trainer(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "config.json").write_text("{}")
    (src / "model.safetensors").write_bytes(b"x")
    (src / "optimizer.pt").write_bytes(b"y")
    (src / "rng_state.pth").write_bytes(b"z")
    copied = copy_sidecar_files(str(src), str(dst))
    assert copied == ["config.json"]
    assert os.listdir(dst) == ["config.json"]


def test_sharded_checkpoint_load(tmp_path):
    ckpt = tmp_path / "sharded"
    ckpt.mkdir()
    save_file({"a.weight": torch.ones(2)}, str(ckpt / "model-00001-of-00002.safetensors"))
    save_file({"b.weight": torch.zeros(3)}, str(ckpt / "model-00002-of-00002.safetensors"))
    index = {
        "weight_map": {
            "a.weight": "model-00001-of-00002.safetensors",
            "b.weight": "model-00002-of-00002.safetensors",
        }
    }
    (ckpt / "model.safetensors.index.json").write_text(json.dumps(index))
    files = list_safetensor_files(str(ckpt))
    assert len(files) == 2
    state, _ = load_weight_tensors(str(ckpt))
    assert set(state) == {"a.weight", "b.weight"}


def test_rejects_output_equal_to_input(tmp_path):
    dir_a = _write_ckpt(str(tmp_path / "A"), {"w": torch.ones(2)})
    dir_b = _write_ckpt(str(tmp_path / "B"), {"w": torch.zeros(2)})
    try:
        average_checkpoints(dir_a, dir_b, dir_a, alpha=0.5)
        assert False, "output == model_a should raise"
    except ValueError as exc:
        assert "new directory" in str(exc)


def test_rejects_architecture_mismatch(tmp_path):
    dir_a = _write_ckpt(str(tmp_path / "A"), {"w": torch.ones(2)})
    dir_b = _write_ckpt(
        str(tmp_path / "B"),
        {"w": torch.zeros(2)},
        config={"architectures": ["OtherModel"], "model_type": "qwen3_asr"},
    )
    try:
        average_checkpoints(dir_a, dir_b, str(tmp_path / "C"), alpha=0.5)
        assert False, "architecture mismatch should raise"
    except ValueError as exc:
        assert "architectures" in str(exc)


def test_fill_tied_weights_aliases_omitted_lm_head():
    embed = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    filled = fill_tied_weights({"thinker.model.embed_tokens.weight": embed})
    assert filled["thinker.lm_head.weight"] is embed


def test_average_checkpoints_tied_lm_head_keeps_a_layout(tmp_path):
    embed_a = torch.ones(2, 3)
    embed_b = torch.full((2, 3), 3.0)
    dir_a = _write_ckpt(
        str(tmp_path / "A"),
        {"thinker.model.embed_tokens.weight": embed_a, "other.weight": torch.zeros(2)},
    )
    dir_b = _write_ckpt(
        str(tmp_path / "B"),
        {
            "thinker.model.embed_tokens.weight": embed_b,
            "thinker.lm_head.weight": embed_b.clone(),
            "other.weight": torch.ones(2),
        },
    )
    out = str(tmp_path / "C")
    average_checkpoints(dir_a, dir_b, out, alpha=0.25)
    state, _ = load_weight_tensors(out)
    assert "thinker.lm_head.weight" not in state
    expected = 0.25 * embed_a + 0.75 * embed_b
    assert torch.allclose(state["thinker.model.embed_tokens.weight"], expected)
    assert torch.allclose(state["other.weight"], 0.25 * torch.zeros(2) + 0.75 * torch.ones(2))


def test_cli_average(tmp_path):
    dir_a = _write_ckpt(str(tmp_path / "A"), {"w": torch.tensor([0.0, 10.0])})
    dir_b = _write_ckpt(str(tmp_path / "B"), {"w": torch.tensor([10.0, 0.0])})
    out = str(tmp_path / "C")
    cmd = [
        sys.executable,
        os.path.join("finetuning", "qwen3_asr_average.py"),
        "--model_a", dir_a,
        "--model_b", dir_b,
        "--alpha", "0.4",
        "--output", out,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "C = 0.4 * A + 0.6 * B" in proc.stdout
    state, _ = load_weight_tensors(out)
    expected = 0.4 * torch.tensor([0.0, 10.0]) + 0.6 * torch.tensor([10.0, 0.0])
    assert torch.allclose(state["w"], expected)
