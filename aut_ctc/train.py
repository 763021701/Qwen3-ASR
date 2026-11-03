"""AuT-CTC training (DESIGN.md §9): frozen bf16 AuT, trainable decoder, dynamic-window streaming training.

  python -m aut_ctc.train --config aut_ctc/configs/base.yaml --exp_dir exp/base
  torchrun --nproc_per_node 4 -m aut_ctc.train --config aut_ctc/configs/base.yaml --exp_dir exp/base
  # M5: streaming fine-tune from the M4 offline checkpoint (stream.yaml is merged over base.yaml)
  python -m aut_ctc.train --config aut_ctc/configs/base.yaml aut_ctc/configs/stream.yaml --exp_dir exp/stream
Overrides: --set data.max_batch_seconds=400 optim.lr=5e-4
An existing <exp_dir>/last.pt is resumed automatically.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .augment import WaveAugment
from .data import Collator, ManifestDataset, WeightedBucketBatchSampler
from .evaluate import decode_dataset, eval_dataset, score
from .frontend import AUT_FRAMES_PER_CHUNK, AUT_MEL_PER_CHUNK, AuTFrontend
from .losses import total_loss
from .model import AuTCTCDecoder, CTCDecoderConfig, load_decoder, save_decoder
from .tokenizer import load_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]


def deep_update(base: dict, new: dict) -> dict:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(paths, overrides=()) -> dict:
    """YAML files merged in order, then dotted overrides like ``optim.lr=5e-4``."""
    cfg = {}
    for path in paths:
        deep_update(cfg, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    for item in overrides:
        key, value = item.split("=", 1)
        *parents, leaf = key.split(".")
        node = cfg
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = yaml.safe_load(value)
    return cfg


def sample_stream_config(rng: random.Random, p_stream=0.5, windows=(100, 200, 400), left_choices=(2, 4, 8, -1)):
    """Dynamic-window training (DESIGN §9.3): (n_window_infer, chunk_frames, left_chunks) for one batch.
    Offline: 8 s AuT windows + full decoder attention. Streaming: AuT window W and decoder chunks of the same W."""
    if rng.random() >= p_stream:
        return 800, 0, -1
    w = rng.choice(windows)
    return w, AUT_FRAMES_PER_CHUNK * w // AUT_MEL_PER_CHUNK, rng.choice(left_choices)


def param_groups(model, weight_decay: float):
    """No weight decay for LayerNorm, biases, embeddings and the LayerMix weights (DESIGN §9.2)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 or name.startswith("query_embed") else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def warmup_cosine(warmup: int, total: int, peak: float, min_lr: float):
    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return (min_lr + (peak - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))) / peak

    return factor


def git_revision() -> str:
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                                        text=True).strip()
        return rev + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class Logger:
    def __init__(self, path: Path, enabled: bool):
        self.path, self.enabled = path, enabled

    def __call__(self, record: dict):
        if not self.enabled:
            return
        line = json.dumps(record, ensure_ascii=False)
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def update_best(exp: Path, best: list, step: int, mer: float, decoder, keep: int = 3) -> None:
    """Keep the ``keep`` checkpoints with the lowest mean dev MER (best.json lists them, best first)."""
    best.append([mer, step])
    best.sort()
    if [mer, step] in best[:keep]:
        save_decoder(exp / f"ckpt-{step}.pt", decoder, step=step, dev_mer=mer)
    for _, s in best[keep:]:
        (exp / f"ckpt-{s}.pt").unlink(missing_ok=True)
    del best[keep:]
    (exp / "best.json").write_text(json.dumps([{"mer": m, "step": s, "path": f"ckpt-{s}.pt"} for m, s in best], indent=1))


def parse_args():
    p = argparse.ArgumentParser("AuT-CTC training")
    p.add_argument("--config", nargs="+", required=True, help="YAML files, merged in order")
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--set", nargs="*", default=[], help="overrides, e.g. optim.lr=5e-4")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    exp = Path(args.exp_dir)
    world, rank, local_rank = (int(os.environ.get(k, d)) for k, d in (("WORLD_SIZE", 1), ("RANK", 0), ("LOCAL_RANK", 0)))
    if world > 1:
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=1))  # rank 0 evaluates while others wait
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    tcfg, acfg, ocfg, dcfg = cfg["tokenizer"], cfg["aut"], cfg["optim"], cfg["data"]
    scfg, ecfg = cfg.get("streaming", {}), cfg.get("eval", {})
    log = Logger(exp / "train_log.jsonl", rank == 0)

    # ---------------------------------------------------------------- tokenizer, AuT, decoder
    vocab_json = exp / "vocab.json"
    tokenizer = load_tokenizer(tcfg["sp_model"], vocab_json if vocab_json.exists() else None)
    frontend = AuTFrontend(acfg["model_dir"], device, getattr(torch, acfg.get("dtype", "bfloat16")),
                           acfg.get("layer_ids", [-1]), acfg.get("attn_implementation", "sdpa"))
    decoder = AuTCTCDecoder(CTCDecoderConfig(**cfg["decoder"], layer_ids=frontend.layer_ids,
                                             vocab_size=tokenizer.vocab_size)).to(device)
    seed = int(cfg.get("seed", 0))
    if rank == 0:
        exp.mkdir(parents=True, exist_ok=True)
        if not vocab_json.exists():
            tokenizer.save_vocab(vocab_json, tcfg["sp_model"])
        (exp / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        meta = {"git": git_revision(), "aut_model_dir": frontend.model_dir, "aut_fingerprint": frontend.fingerprint,
                "seed": seed, "world_size": world, "torch": torch.__version__,
                "decoder_params": sum(p.numel() for p in decoder.parameters()), "vocab_size": tokenizer.vocab_size}
        (exp / "meta.json").write_text(json.dumps(meta, indent=1))
        log({"event": "start", **meta})

    # ---------------------------------------------------------------- data
    style = tcfg.get("style", "woitn")
    aug_cfg = dict(dcfg.get("augment") or {})
    specaug = aug_cfg.pop("specaug", None)
    train_set = ManifestDataset(dcfg["train_manifests"], tokenizer, style, dcfg.get("min_duration", 0.3),
                                dcfg.get("max_duration", 30.0), dcfg.get("drop_unk", True), decoder.cfg.upsample,
                                decoder.cfg.num_queries, augment=WaveAugment(**aug_cfg) if aug_cfg else None)
    weights = dcfg.get("sampling_weights") or [1.0] * len(dcfg["train_manifests"])
    log({"event": "data", "train_utts": len(train_set), "train_hours": round(sum(train_set.durations) / 3600, 2),
         "filter": dict(train_set.stats)})
    dev_sets = [(Path(m).stem, eval_dataset(m, tokenizer, decoder.cfg, style))
                for m in (dcfg.get("dev_manifests") or [] if rank == 0 else [])]

    # ---------------------------------------------------------------- optimization
    num_batches = WeightedBucketBatchSampler(train_set.durations, train_set.sources, weights,
                                             dcfg["max_batch_seconds"]).num_batches()
    total_steps = int(ocfg.get("total_steps") or math.ceil(float(ocfg.get("epochs", 10)) * num_batches / world))
    peak = float(ocfg["lr"])
    optimizer = torch.optim.AdamW(param_groups(decoder, float(ocfg.get("weight_decay", 0.01))), lr=peak,
                                  betas=tuple(ocfg.get("betas", (0.9, 0.98))), eps=float(ocfg.get("eps", 1e-8)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(int(ocfg.get("warmup_steps", 5000)), total_steps, peak, float(ocfg.get("min_lr", 1e-5))))
    step, best = 0, []
    if (exp / "last.pt").exists():
        ckpt = torch.load(exp / "last.pt", map_location="cpu", weights_only=False)
        decoder.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        step, best = ckpt["step"], ckpt["best"]
        log({"event": "resume", "step": step})
    elif cfg.get("init_from"):
        decoder.load_state_dict(load_decoder(cfg["init_from"])[0].state_dict())
        log({"event": "init_from", "path": cfg["init_from"]})
    model = DDP(decoder, device_ids=[local_rank]) if world > 1 else decoder

    # A resumed run continues with fresh but deterministic random streams.
    run_seed = seed + 1000 * rank + 1_000_003 * step
    rng = random.Random(run_seed)
    torch.manual_seed(run_seed)
    np.random.seed(run_seed % 2**32)
    sampler = WeightedBucketBatchSampler(train_set.durations, train_set.sources, weights, dcfg["max_batch_seconds"],
                                         run_seed)
    workers = int(dcfg.get("num_workers", 4))
    loader = DataLoader(train_set, batch_sampler=sampler, collate_fn=Collator(frontend.fe, specaug),
                        num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=workers > 0)

    bf16 = bool(ocfg.get("bf16", True)) and device.type == "cuda"
    log_every, eval_every = int(cfg.get("log", {}).get("every_steps", 50)), int(ecfg.get("every_steps", 2000))
    stream_kw = dict(p_stream=float(scfg.get("p_stream", 0.0)), windows=scfg.get("windows", [100, 200, 400]),
                     left_choices=scfg.get("left_chunks", [2, 4, 8, -1]))
    inter_w, grad_clip = float(cfg.get("loss", {}).get("inter_ctc_weight", 0.3)), float(ocfg.get("grad_clip", 5.0))
    log({"event": "train", "total_steps": total_steps, "batches_per_pass": num_batches, "start_step": step})

    # ---------------------------------------------------------------- loop
    decoder.train()
    batches, t0, audio_sec = iter(loader), time.time(), 0.0
    while step < total_steps:
        batch = next(batches)
        step += 1
        n_window, chunk_frames, left_chunks = sample_stream_config(rng, **stream_kw)
        feats, lens = frontend.encode(batch["input_features"], batch["mel_lens"], n_window)
        targets, target_lens = batch["targets"].to(device), batch["target_lens"].to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            out = model(feats, lens, chunk_frames=chunk_frames, left_chunks=left_chunks)
        loss, final, inter = total_loss(out, targets, target_lens, inter_w)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        audio_sec += float(batch["mel_lens"].sum()) / 100

        if step % log_every == 0:
            with torch.no_grad():
                valid = torch.arange(out["logits"].size(1), device=device)[None] < out["lens"][:, None]
                blank = (out["logits"].argmax(-1) == 0)[valid].float().mean().item()
            elapsed = time.time() - t0
            log({"step": step, "loss": round(loss.item(), 4), "ctc": round(final.item(), 4),
                 "inter": round(inter.item(), 4) if inter is not None else None, "lr": scheduler.get_last_lr()[0],
                 "grad_norm": round(float(grad_norm), 3), "blank": round(blank, 3), "window": n_window,
                 "left_chunks": left_chunks if chunk_frames else None, "rtf_x": round(audio_sec / elapsed, 1),
                 "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else None})
            t0, audio_sec = time.time(), 0.0

        if step % eval_every == 0 or step == total_steps:
            if rank == 0:
                dev = {}
                for name, ds in dev_sets:
                    res = decode_dataset(frontend, decoder, tokenizer, ds, ecfg.get("max_batch_seconds", 600),
                                         local_frames=int(ecfg.get("local_frames", 0)), bf16=bf16)
                    dev[name] = score([r["ref"] for r in res], [r["hyp"] for r in res])["mer"]
                if dev:
                    update_best(exp, best, step, sum(dev.values()) / len(dev), decoder)
                save_decoder(exp / "last.pt", decoder, optimizer=optimizer.state_dict(),
                             scheduler=scheduler.state_dict(), step=step, best=best)
                log({"event": "eval", "step": step, "dev_mer": dev, "best": best[:1]})
            if world > 1:
                dist.barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
