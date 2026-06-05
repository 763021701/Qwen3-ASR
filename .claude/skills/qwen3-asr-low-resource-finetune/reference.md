# Qwen3-ASR Fine-Tuning — Reference & Troubleshooting

## Language List Authority

`qwen_asr/inference/utils.py` → `SUPPORTED_LANGUAGES` (currently ~30 languages).

Inference-side `validate_language` / `validate_language_spec` reject language atoms not in `SUPPORTED_LANGUAGES`. Training data `language {Name}` or `language {Name,Name,...}` — each atom must match. For code-switched labels, `normalize_language_spec` reorders atoms by `SUPPORTED_LANGUAGES` definition order.

## Transcript Normalization (`docs/normalize_label.md`)

**Authoritative manual**: `docs/normalize_label.md`. Read before writing any JSONL-producing script.

**Division of labor**:

| Step | Owner | Notes |
|------|-------|-------|
| Transcript body normalization | Conversion script (per manual) | Punctuation, whitespace, numbers, etc. — before `<asr_text>` tag |
| `language` prefix | `normalize_language_spec` / `format_label.py` | Only normalizes language label, does not touch transcript body |
| Training collator | `finetuning/qwen3_asr_sft.py` | Full `text` as target, no secondary normalization |
| Eval scoring | `evaluation/*/eval_*_jsonl.py` | Uses `masr_eval_pkg` for normalization + metrics (ChineseNormalizer, UyghurNormalizer, compute_wer/cer/mer). See `masr_eval_pkg` for language-specific normalization details |

**Language → manual section**: `English` → ENGLISH; `Chinese` → MANDARIN; `Cantonese` → CANTONESE. For code-switched labels, confirm per-utterance strategy with user. For `SUPPORTED_LANGUAGES` entries not covered in manual: ask user, at minimum follow GLOBAL RULES.

**Script self-check**:
- [ ] Read `docs/normalize_label.md` fully
- [ ] Same normalize function for train/dev/test
- [ ] Post-normalization empty strings are discarded and counted in report
- [ ] Number, script, apostrophe policies are reproducible with clear defaults via argparse or config
- [ ] Final `text` = `language {spec}<asr_text>{normalized_body}`

Note: `tools/convert_to_qwen3_asr_jsonl.py`'s `normalize_target_text` only does `strip` + prefix assembly. SwitchLingua source has `_clean_switchlingua_transcript` (whitespace only). New corpora must not copy "strip only" as full normalization.

## Training Script & Corpus Decoupling

`finetuning/qwen3_asr_sft.py` uses each sample's entire `text` as `target` (predicted after chat prefix). Do NOT put bare transcripts without `language ...<asr_text>` prefix in JSONL (unless user explicitly changes collator logic — not the default).

## Training Modes

### Full Fine-Tune (default) — `finetuning/qwen3_asr_sft.py`
All model parameters trainable. Best for large datasets (≥100h). Checkpoint size: full model (~7GB safetensors). Supports `--freeze_audio_tower 1` to freeze encoder.

### Freeze Audio Tower (`--freeze_audio_tower 1`)
Available in the full-finetune script. Freeze `model.thinker.audio_tower` params; train only LLM-side (text processing) parameters. The `KeepAudioTowerFrozenCallback` re-applies freeze on `on_step_begin` / `on_evaluate` to guard against `model.train()` resetting `training` mode on the audio tower submodule. Fewer trainable params than full fine-tune.

### LoRA — `finetuning/qwen3_asr_sft_lora.py`
Add low-rank adapters via PEFT library to specific module groups controlled by `--lora_scope`. Freezes all base model params; only LoRA adapter weights are trainable (~1% of params). Checkpoints contain only adapter weights (~5-20MB).

**Per-module learning rates**: `CastFloatInputsTrainer.create_optimizer()` groups params by `_lora_group(name)` into `encoder` / `aligner` / `llm` / `other` buckets:

| Group | Controlled by | Matched params |
|-------|--------------|----------------|
| `encoder` | `--lr_encoder` | LoRA params in `audio_tower.layers.*` (q/k/v/out_proj, fc1/fc2) |
| `aligner` | `--lr_aligner` | LoRA params in `audio_tower.conv_out, proj1, proj2` |
| `llm` | `--lr_llm` | LoRA params in `model.layers.*` (q/k/v/o/gate/up/down_proj) |
| `other` | `--lr` (main) | Non-LoRA params, or LoRA params that don't match above patterns |

### LoRA Merge — `tools/merge_lora.py`
Merges a trained LoRA adapter into the base model, producing a standalone checkpoint:

```bash
python tools/merge_lora.py \
  --base_model Qwen/Qwen3-ASR-1.7B \
  --adapter outputs/lora_run/checkpoint-XXX \
  --output outputs/lora_merged
```

The merged output can be loaded directly with `Qwen3ASRModel.from_pretrained()` for inference.

### 3-Stage Progressive LoRA (Mega-ASR A2S-SFT strategy)

All stages use `finetuning/qwen3_asr_sft_lora.py`.

1. **Stage 1 (`encoder_aligner`)**: Adapt speech encoder + audio-text aligner to target acoustic domain.
2. **Stage 2 (`llm`)**: Merge stage-1 adapter via `--merge_lora_into_base_from`, then train LLM LoRA for semantic recovery under degraded conditions.
3. **Stage 3 (`all`)**: Merge stage-2 adapter, then jointly train encoder + aligner + LLM LoRA.

Each stage can also be run independently from base model (without `--merge_lora_into_base_from`).

After training, merge the final adapter for inference:
```bash
python tools/merge_lora.py --base_model Qwen/Qwen3-ASR-1.7B --adapter outputs/stage3/checkpoint-XXX --output outputs/merged
```

## Tokenizer Verification

- `tools/verify_tokenizer_cv_ug.py`: reads `sentence` column from TSV for round-trip. For new corpora, adapt to read `text` from generated JSONL, or check stripped transcript body.
- If many UNK tokens: verify Unicode normalization matches training; consider larger checkpoint; confirm with user whether subword segmentation is acceptable.

## Common Issues

| Symptom | Diagnosis |
|---------|-----------|
| `FileNotFoundError` for audio | Check `audio` is absolute path; verify file exists on training node |
| Loss not decreasing / NaN | Spot-check `text` for incorrect prefix, extra spaces, wrong `language` name |
| OOM | Reduce `batch_size`, increase `grad_acc`; consider filtering very long audio |
| Train/eval mismatch | Eval JSONL `text` format must match training; eval scripts that strip prefix must be consistent with `evaluation/cantonese/eval_cantonese_asr_jsonl.py` etc. |
| `audioread.exceptions.NoBackendError` / m4a fails | Install ffmpeg: `conda install -c conda-forge ffmpeg` (or ensure ffmpeg on PATH). `PySoundFile failed. Trying audioread` warning is normal for m4a — works with ffmpeg. |
| `Can't load feature extractor` / `preprocessor_config.json` | `Qwen3ASRModel.from_pretrained(local_checkpoint)` needs Processor files matching the base model. If checkpoint only has `tokenizer*`, `config.json`, `model.safetensors`: copy `preprocessor_config.json` (and `chat_template.json` if missing) from the `--model_path` base directory into the `checkpoint-*` directory. |
| `Cannot use apply_chat_template` | Copy `chat_template.json` from base model into checkpoint directory. |
| Cantonese eval exits: `Install OpenCC first` | `pip install opencc-python-reimplemented`, or use `--hanzi_script_norm off` if the script supports it. |
| Cantonese eval exits: `Install cn2an first` | `pip install cn2an` (needed for number normalization in scoring) |
| `pip: bad interpreter` | Shebang broken in this env; use `python -m pip install <pkg>` instead |
| LoRA: `ModuleNotFoundError: No module named 'peft'` | `pip install peft` |
| LoRA: `adapter_model.safetensors` not found on resume | LoRA checkpoints only contain adapter weights. Use `--resume_from` pointing to the checkpoint dir with `adapter_model.safetensors`. |
| LoRA: `Do not use --merge_lora_into_base_from with --resume` | Merging a previous adapter and resuming training are mutually exclusive. Remove `--resume`/`--resume_from` when using `--merge_lora_into_base_from`. |
| LoRA: zero trainable `encoder`/`aligner`/`llm` params shown | Normal if `--lora_scope` excludes those modules (e.g. `llm` scope will show 0 encoder/aligner params). Only the optimizer groups with params are printed. |
| LoRA: `unrecognized arguments: --lora_scope` on full-finetune script | LoRA-only args (`--lora_scope`, `--lora_r`, `--lora_alpha`, `--lr_encoder`, etc.) are only in `finetuning/qwen3_asr_sft_lora.py`. Use the correct script. |
| Full-finetune: `unrecognized arguments: --freeze_audio_tower` on LoRA script | `--freeze_audio_tower` is only in `finetuning/qwen3_asr_sft.py`. LoRA already freezes all base params via PEFT. |
| LoRA inference: `Can't load feature extractor` on merged model | `merge_lora.py` saves the processor from the loaded wrapper. If loading fails, re-run merge with `--base_model` pointing to a valid HF model id or cached path. |

## Checkpoint & Eval Loading (Qwen3-ASR)

**Full-finetune checkpoints** (`finetuning/qwen3_asr_sft.py`): The `MakeEveryCheckpointInferableCallback` copies processor/tokenizer files from the base model on each save, so checkpoints should be directly loadable with `Qwen3ASRModel.from_pretrained()`. If loading fails with `Can't load feature extractor`, copy `preprocessor_config.json` and `chat_template.json` from the base model into the checkpoint directory.

**LoRA checkpoints** (`finetuning/qwen3_asr_sft_lora.py`): Contain only adapter weights + config. Cannot be loaded directly with `Qwen3ASRModel.from_pretrained()`. Two options for inference:
1. `tools/_eval_lora_ckpt.py` — loads base model + adapter at inference time
2. `tools/merge_lora.py` — merges adapter into base model, producing a standalone checkpoint

**Long-term fix** (optional): call `processor.save_pretrained(output_dir)` on checkpoint save, or merge with base files.

## Existing Conversion Scripts (Reference)

- `tools/convert_to_qwen3_asr_jsonl.py` — general-purpose training/finetuning data converter; called by pipeline `prepare` stage
- `evaluation/chinese/wsc/prepare_wsc_eval_qwen3.py` — WSC / Sichuanhua eval
- `evaluation/cantonese/wsyue_asr/prepare_wsyue_asr_eval_qwen3.py` — Cantonese wsyue eval

## Eval Scripts

Per-language/per-task eval scripts under `evaluation/<lang>/eval_*_jsonl.py` with baselines in `evaluation/<lang>/baselines/`. See `evaluation/README.md` for common invocation patterns. For new languages, reuse the eval script closest in inference interface and change `--language` or JSONL path.
