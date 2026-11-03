"""M3 layer probe (DESIGN.md §12.1): one AuT forward feeds an LN+Linear CTC probe per layer; dev MER per layer.

  python -m aut_ctc.tools.probe_layers --config aut_ctc/configs/base.yaml aut_ctc/configs/probe.yaml --exp_dir exp/probe

Decision rule: keep layer_ids=(-1,) if ln_post is best or within 3% (relative) of the best probe, otherwise use a
LayerMix of the last 8 layers + ln_post.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data import Collator, ManifestDataset, WeightedBucketBatchSampler, duration_batches
from ..decode import ctc_greedy, hyps_to_text
from ..evaluate import score
from ..frontend import AuTFrontend
from ..losses import ctc_loss
from ..tokenizer import load_tokenizer
from ..train import load_config, warmup_cosine


@torch.no_grad()
def probe_dev_mer(frontend, probes, tokenizer, dataset, max_batch_seconds) -> dict:
    probes.eval()
    batches = duration_batches(range(len(dataset)), dataset.durations, max_batch_seconds)
    refs, hyps = [], {name: [] for name in probes}
    for batch in DataLoader(dataset, batch_sampler=batches, collate_fn=Collator(frontend.fe), num_workers=2):
        feats, lens = frontend.encode(batch["input_features"], batch["mel_lens"])
        refs += batch["texts"]
        for (name, probe), f in zip(probes.items(), feats):
            hyps[name] += hyps_to_text(tokenizer, ctc_greedy(probe(f.float()).float(), lens))
    probes.train()
    return {name: score(refs, h)["mer"] for name, h in hyps.items()}


def main():
    p = argparse.ArgumentParser("AuT-CTC M3 layer probe")
    p.add_argument("--config", nargs="+", required=True)
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--set", nargs="*", default=[])
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    pcfg, dcfg, acfg = cfg["probe"], cfg["data"], cfg["aut"]
    exp = Path(args.exp_dir)
    exp.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(int(cfg.get("seed", 0)))
    torch.manual_seed(int(cfg.get("seed", 0)))

    tokenizer = load_tokenizer(cfg["tokenizer"]["sp_model"])
    layer_ids = tuple(pcfg["layer_ids"])
    frontend = AuTFrontend(acfg["model_dir"], device, getattr(torch, acfg.get("dtype", "bfloat16")), layer_ids,
                           acfg.get("attn_implementation", "sdpa"))
    in_dim = frontend.aut.config.d_model
    probes = nn.ModuleDict({str(i): nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, tokenizer.vocab_size))
                            for i in layer_ids}).to(device)
    style = cfg["tokenizer"].get("style", "woitn")
    train_set = ManifestDataset(dcfg["train_manifests"], tokenizer, style, dcfg.get("min_duration", 0.3),
                                dcfg.get("max_duration", 30.0), dcfg.get("drop_unk", True))
    dev_sets = [(Path(m).stem, ManifestDataset([m], tokenizer, style, 0.0, float("inf"), False, check_feasible=False))
                for m in dcfg.get("dev_manifests") or []]
    weights = dcfg.get("sampling_weights") or [1.0] * len(dcfg["train_manifests"])
    sampler = WeightedBucketBatchSampler(train_set.durations, train_set.sources, weights, dcfg["max_batch_seconds"],
                                         rng.randrange(2**31))
    loader = DataLoader(train_set, batch_sampler=sampler, collate_fn=Collator(frontend.fe),
                        num_workers=int(dcfg.get("num_workers", 4)))
    steps, peak = int(pcfg["steps"]), float(pcfg["lr"])
    optimizer = torch.optim.AdamW(probes.parameters(), lr=peak, betas=(0.9, 0.98), weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(int(pcfg.get("warmup_steps", 500)), steps,
                                                                           peak, peak * 0.01))
    bf16 = device.type == "cuda"
    t0 = time.time()
    for step, batch in zip(range(1, steps + 1), loader):
        feats, lens = frontend.encode(batch["input_features"], batch["mel_lens"])
        targets, target_lens = batch["targets"].to(device), batch["target_lens"].to(device)
        losses = []
        for probe, f in zip(probes.values(), feats):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                logits = probe(f.float())
            losses.append(ctc_loss(logits, lens, targets, target_lens))
        optimizer.zero_grad(set_to_none=True)
        sum(losses).backward()
        optimizer.step()
        scheduler.step()
        if step % 100 == 0:
            print(json.dumps({"step": step, **{f"loss_{k}": round(l.item(), 3) for k, l in zip(probes, losses)},
                              "sec": round(time.time() - t0, 1)}), flush=True)

    report = {"layer_ids": list(layer_ids), "steps": steps, "dev": {}}
    for name, ds in dev_sets:
        report["dev"][name] = probe_dev_mer(frontend, probes, tokenizer, ds, cfg.get("eval", {}).get("max_batch_seconds", 600))
    mean = {k: sum(d[k] for d in report["dev"].values()) / len(report["dev"]) for k in probes} if dev_sets else {}
    if mean:
        best = min(mean.values())
        use_ln_post = "-1" in mean and (mean["-1"] <= best * 1.03)
        report["mean_mer"] = mean
        report["decision"] = {"layer_ids": [-1] if use_ln_post else list(range(16, 24)) + [-1],
                              "rule": "ln_post best or within 3% relative of the best probe -> [-1]; else last 8 layers + ln_post"}
    (exp / "probe_report.json").write_text(json.dumps(report, indent=1))
    torch.save(probes.state_dict(), exp / "probes.pt")
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
