import os
import subprocess
import sys
import tempfile

import torch

from finetuning.qwen3_asr_opd import _discounted_future_sum, _validate_args, _save_opd_checkpoint


def test_discounted_future_sum_recurrence():
    # out[t] = value[t] + gamma * out[t+1]  (reverse cumulative discounted sum)
    values = torch.tensor([[1.0, 2.0, 3.0]])
    # out[t] = value[t] + gamma*out[t+1]: out2=3, out1=2+0.5*3=3.5, out0=1+0.5*3.5=2.75
    out = _discounted_future_sum(values, gamma=0.5)
    assert torch.allclose(out, torch.tensor([[2.75, 3.5, 3.0]]), atol=1e-6)
    # gamma == 0 is a no-op (per-token advantage)
    assert torch.allclose(_discounted_future_sum(values, gamma=0.0), values)


def _namespace(**kw):
    ns = argparse_ns()
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def argparse_ns():
    import argparse

    return argparse.Namespace()


def test_validate_allows_single_generation_but_rejects_bad_discount():
    ok = _namespace(
        batch_size=1, num_generations=1, grad_acc=1, max_new_tokens=512,
        temperature=1.0, kl_discount=0.0,
    )
    _validate_args(ok)  # num_generations==1 is fine for OPD (no group normalization)

    bad = _namespace(
        batch_size=1, num_generations=0, grad_acc=1, max_new_tokens=512,
        temperature=1.0, kl_discount=0.0,
    )
    try:
        _validate_args(bad)
        assert False, "num_generations<1 should raise"
    except ValueError:
        pass


def test_reverse_kl_policy_gradient_direction():
    """loss = -(student_logp - teacher_logp).detach() * student_logp, summed over mask.

    Gradient wrt student_logp at the fixed sample must equal -reverse_kl, so the
    update raises logp where the teacher is more confident (reverse_kl<0) and
    lowers it where the student over-commits (reverse_kl>0) — the OPD correction.
    """
    teacher_logps = torch.tensor([[-1.0, -2.0, -0.5]])  # constant
    student_logps = torch.tensor([[-3.0, -1.0, -3.0]], requires_grad=True)
    mask = torch.tensor([[1.0, 1.0, 0.0]])  # 3rd token masked out
    reverse_kl = (student_logps.detach() - teacher_logps)
    advantage = -1.0 * reverse_kl  # kl_coef = 1
    per_token = -advantage * student_logps
    loss = (per_token * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    loss.mean().backward()

    grad = student_logps.grad
    # masked token gets zero gradient
    assert grad[0, 2].item() == 0.0
    # d loss / d student_logp = -advantage*mask/n_masked = reverse_kl*mask/n_masked
    n_masked = mask.sum()
    expected = (reverse_kl * mask) / n_masked
    assert torch.allclose(grad, expected, atol=1e-6)
    # token 0: student_logp(-3) < teacher(-1) -> reverse_kl<0 -> grad<0 -> ascent raises logp (pull UP to teacher)
    assert grad[0, 0].item() < 0.0
    # token 1: student over-confident (student -1 > teacher -2) -> reverse_kl>0 -> grad>0 -> descent lowers logp
    assert grad[0, 1].item() > 0.0


def test_opd_dry_run_without_cuda(tmp_path):
    """--dry_run loads the dataset + prints freeze resolution without touching CUDA."""
    # point at a real, accessible audio file from gigaspeech if present, else skip
    gigaspeech = os.path.join("data", "gigaspeech", "train.jsonl")
    if not os.path.isfile(gigaspeech):
        import pytest

        pytest.skip("gigaspeech train.jsonl not available")
    with open(gigaspeech) as f:
        first = f.readline().strip()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(first + "\n")

    cmd = [
        sys.executable, os.path.join("finetuning", "qwen3_asr_opd.py"),
        "--student_path", "does/not/matter", "--teacher_path", "does/not/matter",
        "--train_file", str(manifest), "--output_dir", str(tmp_path / "out"),
        "--dry_run", "1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "[data] rows=1" in out
    assert "frozen=encoder,aligner; trainable=llm" in out
    assert "reverse-KL" in out


def test_save_checkpoint_uses_opd_state_filename(tmp_path, monkeypatch):
    """_save_opd_checkpoint writes opd_trainer_state.pt (not grpo_trainer_state.pt)."""
    calls = {}

    class FakeModel:
        class generation_config:  # noqa: N801
            do_sample = False

        def save_pretrained(self, d, safe_serialization=True):
            calls["model_dir"] = d

        def parameters(self):
            return []

    class FakeProcessor:
        def save_pretrained(self, d):
            pass

    class FakeOpt:
        def state_dict(self):
            return {}

    # torch.save is patched to just record the path and not serialize the fakes
    def fake_save(obj, path):
        calls["state_path"] = path

    monkeypatch.setattr("finetuning.qwen3_asr_opd.torch.save", fake_save)
    path = _save_opd_checkpoint(FakeModel(), FakeProcessor(), FakeOpt(), str(tmp_path), 42, 0)
    assert path.endswith("checkpoint-42")
    # do_sample was forced True in-place on the fake's generation_config
    assert FakeModel.generation_config.do_sample is True
    assert calls["state_path"].endswith("opd_trainer_state.pt")
    assert "grpo" not in calls["state_path"]
