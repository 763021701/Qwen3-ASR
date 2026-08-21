#!/usr/bin/env python3
"""Rollout loop test: measure how often the model falls into repetition loops.

Runs N sampled rollouts (plus one greedy decode) per audio and flags outputs
that loop: a word 5-gram repeated >= 3 times, a char 20-gram (non-overlapping)
repeated >= 3 times, or generation that hits --max_new_tokens without EOS.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import transformers
from transformers import GenerationConfig
from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import parse_asr_output

_WS_RE = re.compile(r"\s+")


class TokenPenaltyLogitsProcessor:
    """Standard frequency/presence penalty semantics.

    transformers >= 4.5x removed native frequency_penalty/presence_penalty
    from the generation loop (GenerationConfig still stores them but they are
    never applied), so this processor implements the classic behavior:
    scores -= frequency_penalty * occurrences + presence_penalty * (occurred).
    """

    def __init__(self, frequency_penalty: float, presence_penalty: float, vocab_size: int):
        self.fp = float(frequency_penalty)
        self.pp = float(presence_penalty)
        self.vocab = int(vocab_size)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.fp == 0.0 and self.pp == 0.0:
            return scores
        out = scores.clone().float()
        for row in range(input_ids.shape[0]):
            values, counts = torch.unique(input_ids[row], return_counts=True)
            if self.fp:
                penalty = torch.zeros(self.vocab, dtype=torch.float32, device=scores.device)
                penalty[values] = counts.float() * self.fp
                out[row] -= penalty
            if self.pp:
                present = torch.zeros(self.vocab, dtype=torch.float32, device=scores.device)
                present[values] = 1.0
                out[row] -= present * self.pp
        return out.to(scores.dtype)


def detect_repetition(text: str) -> dict:
    words = _WS_RE.split(text.strip())
    n_word = 5
    word_repeats = 0
    top_word_gram = ""
    if len(words) >= n_word * 2:
        counts = Counter(
            tuple(words[i : i + n_word]) for i in range(len(words) - n_word + 1)
        )
        gram, word_repeats = counts.most_common(1)[0]
        top_word_gram = " ".join(gram)

    n_char = 20
    char_repeats = 0
    top_char_gram = ""
    if len(text) >= n_char * 2:
        counts = Counter(text[i : i + n_char] for i in range(0, len(text) - n_char + 1, n_char))
        gram, char_repeats = counts.most_common(1)[0]
        top_char_gram = gram

    return {
        "word5_repeats": word_repeats,
        "top_word5": top_word_gram,
        "char20_repeats": char_repeats,
        "top_char20": top_char_gram,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--audio_dir", required=True)
    p.add_argument("--num_rollouts", type=int, default=8, help="Sampled rollouts per audio.")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--frequency_penalty", type=float, default=0.0,
                   help="HF generation frequency_penalty applied to every rollout.")
    p.add_argument("--presence_penalty", type=float, default=0.0,
                   help="HF generation presence_penalty applied to every rollout.")
    p.add_argument("--greedy", type=int, default=1, help="Also run one greedy decode per audio.")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    audio_paths = sorted(Path(args.audio_dir).rglob("*.wav"))
    if not audio_paths:
        raise SystemExit(f"No wav files under {args.audio_dir}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rollout_loop_test.jsonl"

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    print(f"Loading model from {args.model!r} ...", flush=True)
    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map="cuda:0",
        max_inference_batch_size=1,
        max_new_tokens=args.max_new_tokens,
    )
    hf_model = model.model
    tokenizer = model.processor.tokenizer
    # The outer Qwen3ASR generate() override does not read the outer
    # generation_config; generation parameters must be set on the thinker.
    thinker = hf_model.thinker

    greedy_cfg = GenerationConfig(
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
    )
    sample_cfg = GenerationConfig(
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
    )
    parts = []
    if args.frequency_penalty:
        parts.append(f"fp{args.frequency_penalty:g}")
    if args.presence_penalty:
        parts.append(f"pp{args.presence_penalty:g}")
    pen_label = "_".join(parts) + "_" if parts else ""

    if pen_label:
        # transformers 4.5x silently ignores frequency/presence_penalty, so
        # inject the classic penalty as a custom logits processor. The outer
        # Qwen3ASR generate() forwards extra kwargs to thinker.generate.
        cfg_obj = thinker.config
        vocab_size = (
            getattr(getattr(cfg_obj, "text_config", None), "vocab_size", None)
            or getattr(cfg_obj, "vocab_size", None)
            or len(tokenizer)
        )
        _orig_thinker_generate = thinker.generate

        def _thinker_generate_with_penalty(*args_, **kwargs_):
            kwargs_["logits_processor"] = [
                TokenPenaltyLogitsProcessor(args.frequency_penalty, args.presence_penalty, vocab_size)
            ]
            return _orig_thinker_generate(*args_, **kwargs_)

        thinker.generate = _thinker_generate_with_penalty
        print(
            f"[penalty] HF {transformers.__version__} does not apply frequency/presence penalties "
            f"natively; injecting custom logits processor (vocab={vocab_size}).",
            flush=True,
        )

    records = []
    done_keys = set()
    if out_path.is_file():
        for line in out_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done_keys.add((rec["audio"], rec["rollout"]))

    t_start = time.time()
    for audio_path in audio_paths:
        modes = ([f"{pen_label}greedy"] if args.greedy else []) + [
            f"{pen_label}sample-{i}" for i in range(args.num_rollouts)
        ]
        for mode in modes:
            key = (str(audio_path), mode)
            if key in done_keys:
                continue
            is_greedy = mode.endswith("greedy")
            cfg = greedy_cfg if is_greedy else sample_cfg
            thinker.generation_config = cfg
            hf_model.generation_config = cfg
            t0 = time.time()
            outs = model.transcribe(audio=[str(audio_path)], language=None, return_time_stamps=False)
            raw_out = getattr(outs[0], "text", "") or ""
            _, hyp = parse_asr_output(raw_out, user_language=None)
            hyp = hyp or ""
            tokens = len(tokenizer.encode(hyp, add_special_tokens=False)) if hyp else 0
            rep = detect_repetition(hyp)
            hit_max = tokens >= args.max_new_tokens
            loop = hit_max or rep["word5_repeats"] >= 3 or rep["char20_repeats"] >= 3
            rec = {
                "audio": str(audio_path),
                "rollout": mode,
                "frequency_penalty": args.frequency_penalty,
                "presence_penalty": args.presence_penalty,
                "temperature": None if is_greedy else args.temperature,
                "top_p": None if is_greedy else args.top_p,
                "hyp_tokens": tokens,
                "hyp_words": len(hyp.split()),
                "hit_max_tokens": hit_max,
                "loop": loop,
                **rep,
                "hypothesis": hyp,
                "elapsed_sec": round(time.time() - t0, 1),
            }
            records.append(rec)
            with out_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(
                f"[{len(records)}/{len(audio_paths) * len(modes)}] {audio_path.name} {mode}: "
                f"tokens={tokens} loop={loop} w5={rep['word5_repeats']} c20={rep['char20_repeats']} "
                f"({rec['elapsed_sec']}s)",
                flush=True,
            )

    all_recs = []
    for line in out_path.read_text().splitlines():
        if line.strip():
            all_recs.append(json.loads(line))

    total = len(all_recs)
    loops = [r for r in all_recs if r["loop"]]
    hit_max = [r for r in all_recs if r["hit_max_tokens"]]
    per_audio = {}
    for r in all_recs:
        per_audio.setdefault(r["audio"], []).append(r)

    summary = {
        "model": args.model,
        "audio_dir": args.audio_dir,
        "num_audios": len(audio_paths),
        "rollouts_per_audio": len(per_audio.get(audio_paths[0], [])) if per_audio else 0,
        "total_rollouts": total,
        "loop_rollouts": len(loops),
        "loop_rate": round(len(loops) / total, 4) if total else None,
        "hit_max_token_rollouts": len(hit_max),
        "loop_rate_hit_max_only": round(len(hit_max) / total, 4) if total else None,
        "per_audio": {
            os.path.basename(os.path.dirname(a)) + "/" + os.path.basename(a): {
                "rollouts": len(rs),
                "loops": sum(1 for r in rs if r["loop"]),
                "loop_rate": round(sum(1 for r in rs if r["loop"]) / len(rs), 4),
                "max_tokens": max(r["hyp_tokens"] for r in rs),
            }
            for a, rs in sorted(per_audio.items())
        },
        "looped_rollouts": [
            {
                "audio": r["audio"],
                "rollout": r["rollout"],
                "hyp_tokens": r["hyp_tokens"],
                "hit_max_tokens": r["hit_max_tokens"],
                "top_word5": r["top_word5"],
                "word5_repeats": r["word5_repeats"],
                "top_char20": r["top_char20"],
                "char20_repeats": r["char20_repeats"],
            }
            for r in loops
        ],
    }
    summary_path = out_dir / "rollout_loop_test_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"\nTotal time: {(time.time() - t_start) / 60:.1f} min")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {out_path} and {summary_path}")


if __name__ == "__main__":
    main()
