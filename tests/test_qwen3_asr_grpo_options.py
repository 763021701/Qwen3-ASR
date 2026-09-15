"""GRPO regression scenarios; no model downloads or CUDA required by these tests."""
from types import SimpleNamespace

import pytest
import torch

from finetuning import qwen3_asr_grpo as grpo


def arguments(monkeypatch, *extra):
    monkeypatch.setattr("sys.argv", ["grpo", "--model_path", "model", "--train_file", "train",
                                     "--output_dir", "out", *extra])
    args = grpo.parse_args()
    grpo._validate_args(args)
    return args


@pytest.mark.parametrize("mode", ["wer", "wer_cer", "cer", "weighted_cer_loop"])
def test_reward_batch_contract(monkeypatch, mode):
    args = arguments(monkeypatch, "--reward_mode", mode)
    rewards, stats = grpo._reward_batch(args, ["hello"] * 4,
                                       ["language English<asr_text>hello"] * 4)
    assert rewards.shape == (4,)
    assert torch.equal(rewards, torch.zeros(4))
    assert not rewards.requires_grad
    assert (stats is not None) == (mode == "weighted_cer_loop")


def test_greedy_advantages_keep_absolute_difference_and_detach():
    rewards = torch.tensor([-0.3, -0.3, -0.3, 0., -0.1, -0.2], requires_grad=True)
    anchors = torch.tensor([-0.1, -0.1], requires_grad=True)
    advantages = grpo.greedy_advantages(rewards, anchors, 3)
    torch.testing.assert_close(advantages, torch.tensor([-0.2, -0.2, -0.2, 0.1, 0., -0.1]))
    assert not advantages.requires_grad
    assert grpo.greedy_advantages(torch.tensor([0.]), torch.tensor([0.]), 1).item() == 0


@pytest.mark.parametrize("extra", [
    ["--num_generations", "1"], ["--epochs", "0"], ["--log_steps", "0"],
    ["--temperature", "nan"], ["--temperature", "inf"], ["--temperature_end", "0"],
    ["--temperature_schedule", "linear"], ["--beta", "-1"], ["--epsilon", "1"],
    ["--freeze_modules", "encoder,aligner,llm"], ["--top_p", "0.95"], ["--top_k", "50"],
    ["--temperature_schedule", "linear", "--temperature_anneal_steps", "10",
     "--temperature_end", "2"],
])
def test_invalid_options(monkeypatch, extra):
    with pytest.raises(ValueError):
        arguments(monkeypatch, *extra)


def test_mode_defaults_and_single_greedy(monkeypatch):
    args = arguments(monkeypatch, "--advantage_mode", "greedy", "--num_generations", "1")
    assert (args.top_p, args.top_k) == (1., 0)
    args = arguments(monkeypatch, "--policy_probability_mode", "legacy")
    assert (args.top_p, args.top_k) == (0.95, 50)


def test_temperature_endpoints(monkeypatch):
    args = arguments(monkeypatch, "--temperature_schedule", "linear", "--temperature", "1",
                     "--temperature_end", "0.1", "--temperature_anneal_steps", "10")
    assert [grpo.temperature_at_step(args, u) for u in [0, 5, 10, 20]] == pytest.approx([1, .55, .1, .1])
    args.temperature_schedule = "constant"
    assert grpo.temperature_at_step(args, 100) == 1


def test_scores_temperature_exact_kl_and_chunk_backward():
    policy = torch.tensor([.6, .4]).log().repeat(1, 35, 1).requires_grad_()
    reference = torch.tensor([.5, .5]).log().repeat(1, 35, 1).requires_grad_()
    ids = torch.ones(1, 35, dtype=torch.long)
    selected, raw, kl = grpo._token_scores(policy, ids, .5, reference)
    torch.testing.assert_close(selected.exp(), torch.full((1, 35), .16 / .52))
    torch.testing.assert_close(raw.exp(), torch.full((1, 35), .4))
    expected = .6 * torch.log(torch.tensor(1.2)) + .4 * torch.log(torch.tensor(.8))
    torch.testing.assert_close(kl, expected.expand(1, 35))
    (selected.mean() + kl.mean()).backward()
    assert policy.grad is not None and torch.isfinite(policy.grad).all()
    assert reference.grad is None
    identity, identity_raw, zero_kl = grpo._token_scores(policy.detach(), ids, 1., policy.detach())
    torch.testing.assert_close(identity, identity_raw)
    torch.testing.assert_close(zero_kl, torch.zeros_like(zero_kl))
    old = grpo._token_scores(policy.detach(), ids, .5)[0]
    torch.testing.assert_close((selected.detach() - old).exp(), torch.ones_like(old))


def test_repeat_penalty_requires_actual_repetition():
    assert grpo.loop_penalty_ratio("abcdefgh", "ijklmnop") == 0
    assert grpo.loop_penalty_ratio("abcdabcd", "abcd", 4, 4) > 0
    assert grpo.loop_penalty_ratio("abcdabcd", "abcdabcd", 4, 4) == 0


def test_tail_gradients_average_actual_batches():
    weight = torch.nn.Parameter(torch.tensor(1.))
    optimizer = torch.optim.SGD([weight], lr=.1)
    (weight * 2).backward()
    (weight * 4).backward()
    grpo._optimizer_update(optimizer, [weight], 2, 100.)
    assert weight.item() == pytest.approx(.7)
    assert weight.grad is None


class FakeThinker(torch.nn.Module):
    def __init__(self, value=1.):
        super().__init__()
        self.config = SimpleNamespace(audio_token_id=3)
        self.embedding = torch.nn.Embedding(4, 2)
        self.audio_weight = torch.nn.Parameter(torch.tensor(value))
        self.audio_calls = 0
        self.generation_calls = []

    def get_input_embeddings(self):
        return self.embedding

    def get_audio_features(self, inputs, feature_attention_mask):
        self.audio_calls += 1
        return inputs * self.audio_weight

    def generate(self, input_ids, inputs_embeds, generation_config, **kwargs):
        self.generation_calls.append((inputs_embeds.detach().clone(), kwargs))
        return torch.cat([input_ids, torch.ones(input_ids.shape[0], 1, dtype=torch.long)], dim=1)

    def forward(self, inputs_embeds, **kwargs):
        self.last_embeds = inputs_embeds
        # Prefix/audio affects the subsequent token distribution.
        context = inputs_embeds.cumsum(1).sum(-1, keepdim=True)
        return SimpleNamespace(logits=context * torch.arange(4.))


def fake_prefix():
    return {"input_ids": torch.tensor([[0, 3]]), "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "input_features": torch.ones(1, 2), "feature_attention_mask": torch.ones(1, 1)}


def test_shared_rollout_reference_audio_and_live_gradients():
    policy, reference = FakeThinker(1.).eval(), FakeThinker(2.).eval()
    processor = SimpleNamespace(batch_decode=lambda ids, **kw: ["hello"] * len(ids))
    prefix = fake_prefix()
    config = SimpleNamespace(eos_token_id=[1])
    with torch.no_grad():
        shared = policy.get_audio_features(prefix["input_features"], prefix["feature_attention_mask"])
    rollout = grpo.rollout_groups(policy, processor, prefix, 2, config, audio_features=shared,
                                  use_model_defaults=False)
    grpo.rollout_groups(policy, processor, prefix, 1, config, audio_features=shared)
    assert policy.audio_calls == 1
    torch.testing.assert_close(policy.generation_calls[0][0][0], policy.generation_calls[1][0][0])
    assert policy.generation_calls[0][1]["use_model_defaults"] is False
    assert "use_model_defaults" not in policy.generation_calls[1][1]
    raw_audio = (prefix["input_features"], prefix["feature_attention_mask"])
    with torch.no_grad():
        grpo._completion_logits(reference, rollout, raw_audio)
    assert reference.audio_calls == 1
    torch.testing.assert_close(reference.last_embeds[:, 1], torch.full((2, 2), 2.))
    grpo.completion_logps(policy, rollout, raw_audio).sum().backward()
    assert policy.audio_weight.grad is not None and policy.audio_weight.grad.abs() > 0
    assert reference.audio_weight.grad is None
    # Legacy callers still get a raw selected-token tensor and can encode on demand.
    assert grpo.completion_logps(policy, rollout).shape == (2, 1)
    grpo.rollout_groups(policy, processor, prefix, 1, config)
    assert policy.audio_calls == 3


def test_training_loop_accumulation_save_and_schedule(monkeypatch, tmp_path):
    """Exercise real orchestration with tiny CPU models; stop mid-window."""
    args = arguments(monkeypatch, "--advantage_mode", "greedy", "--num_generations", "2",
                     "--grad_acc", "2", "--max_steps", "3", "--save_steps", "1",
                     "--temperature_schedule", "linear", "--temperature", "1",
                     "--temperature_anneal_steps", "2")
    args.output_dir = str(tmp_path)
    monkeypatch.setattr(grpo, "parse_args", lambda: args)
    monkeypatch.setattr(grpo, "JsonlDataset", lambda *a, **kw: [{"audio": "fake", "text": "hello"}] * 4)
    cpu = torch.device("cpu")
    class TorchProxy:
        cuda = SimpleNamespace(is_available=lambda: True, get_device_capability=lambda *a: (8, 0))
        device = staticmethod(lambda *a: cpu)

        def __getattr__(self, name):
            return getattr(torch, name)

    monkeypatch.setattr(grpo, "torch", TorchProxy())
    monkeypatch.setattr(grpo, "seed_everything", lambda seed: None)
    monkeypatch.setattr(grpo, "_prepare_prefix_inputs", lambda *a: fake_prefix())
    processor = SimpleNamespace(batch_decode=lambda ids, **kw: ["hello"] * len(ids))
    models = []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.thinker = FakeThinker()
            self.thinker.audio_tower = torch.nn.Identity()
            self.generation_config = SimpleNamespace(eos_token_id=[1], pad_token_id=1)

    def load(*a, **kw):
        model = Model()
        models.append(model)
        return SimpleNamespace(model=model, processor=processor)

    monkeypatch.setattr(grpo.Qwen3ASRModel, "from_pretrained", load)
    monkeypatch.setattr(grpo, "_check_reference_compatibility", lambda *a: None)
    monkeypatch.setattr(grpo, "set_part_freeze", lambda *a: None)
    monkeypatch.setattr(grpo, "count_part_parameters", lambda *a: {p: (1, 1) for p in grpo.PARTS})
    saved, temperatures = [], []
    original_rollout = grpo.rollout_groups

    def rollout(*a, **kw):
        if a[4].do_sample:
            temperatures.append(a[4].temperature)
        return original_rollout(*a, **kw)

    def save(model, processor, optimizer, output_dir, step, epoch, training_state):
        assert all(p.grad is None for p in model.parameters())
        saved.append((step, training_state.copy()))
        return str(tmp_path / f"checkpoint-{step}")

    monkeypatch.setattr(grpo, "rollout_groups", rollout)
    monkeypatch.setattr(grpo, "_save_checkpoint", save)
    grpo.main()
    assert temperatures == pytest.approx([1., 1., .55])
    assert [item[0] for item in saved] == [2, 3]
    assert [item[1]["optimizer_step"] for item in saved] == [1, 2]
    assert saved[-1][1]["temperature"] == pytest.approx(.1)
    assert all(not m.training for m in models)
    assert models[1].thinker.audio_calls == 3
    assert models[0].thinker.audio_calls == 3


def test_reference_compatibility_rejects_different_processing():
    def wrapper(scale=1, vocab=None):
        processor = SimpleNamespace(
            tokenizer=SimpleNamespace(get_vocab=lambda: vocab or {"a": 0}, special_tokens_map={}),
            feature_extractor=SimpleNamespace(to_dict=lambda: {"sampling_rate": scale}),
            chat_template="same",
        )
        return SimpleNamespace(processor=processor, model=SimpleNamespace(
            thinker=SimpleNamespace(config=SimpleNamespace(audio_token_id=3))))

    grpo._check_reference_compatibility(wrapper(), wrapper())
    with pytest.raises(ValueError, match="audio preprocessing"):
        grpo._check_reference_compatibility(wrapper(), wrapper(scale=2))
    with pytest.raises(ValueError, match="vocabularies"):
        grpo._check_reference_compatibility(wrapper(), wrapper(vocab={"b": 0}))


def test_checkpoint_records_optimizer_schedule_state(monkeypatch, tmp_path):
    written = {}
    monkeypatch.setattr(grpo.torch, "save", lambda state, path: written.update(state=state, path=path))
    model = SimpleNamespace(generation_config=SimpleNamespace(do_sample=True),
                            save_pretrained=lambda *a, **kw: None)
    processor = SimpleNamespace(save_pretrained=lambda *a: None)
    optimizer = SimpleNamespace(state_dict=lambda: {"state": {}})
    state = {"optimizer_step": 2, "temperature": .1, "temperature_schedule": "linear",
             "temperature_start": 1., "temperature_end": .1, "temperature_anneal_steps": 2}
    grpo._save_checkpoint(model, processor, optimizer, str(tmp_path), 3, 0, training_state=state)
    assert written["state"]["global_step"] == 3
    assert all(written["state"][key] == value for key, value in state.items())
