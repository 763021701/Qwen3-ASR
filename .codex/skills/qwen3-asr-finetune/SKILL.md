---
name: qwen3-asr-finetune
description: End-to-end Qwen3-ASR finetuning pipeline automation for this repo. Use when preparing ASR training data, validating Qwen3-ASR JSONL manifests, launching SFT, selecting checkpoints, or evaluating a finetuned checkpoint.
---

# Qwen3-ASR Finetuning Pipeline

Use this skill when the user asks to fine-tune Qwen3-ASR, add a new speech
dataset, prepare train/dev/test JSONL, validate manifests, run SFT, or evaluate
a final checkpoint.

## Default Workflow

1. Inspect the dataset layout and identify source type:
   - Common Voice TSV + `clips/`
   - Kaldi `wav.scp` + `text`
   - FunASR messages JSONL
   - Custom source requiring a small prepare script
2. Create or update a pipeline config under `configs/`.
3. Run dry-run first:
   `python tools/qwen3_asr_pipeline.py --config CONFIG.yaml --stage all --dry_run 1`
4. Run data preparation:
   `python tools/qwen3_asr_pipeline.py --config CONFIG.yaml --stage prepare`
5. Run validation:
   `python tools/qwen3_asr_pipeline.py --config CONFIG.yaml --stage validate`
6. Run training:
   `python tools/qwen3_asr_pipeline.py --config CONFIG.yaml --stage train`
7. Run evaluation:
   `python tools/qwen3_asr_pipeline.py --config CONFIG.yaml --stage eval`

## Manifest Contract

Each JSONL line must contain:

```json
{"audio": "/absolute/path.wav", "text": "language Uyghur<asr_text>transcript"}
```

The language label must match `SUPPORTED_LANGUAGES` in
`qwen_asr/inference/utils.py`.

## Important Defaults

- Prefer the pipeline CLI over hand-written command sequences.
- Keep `finetuning/qwen3_asr_sft.py` unchanged unless the user explicitly asks
  for training behavior that the current script cannot express.
- Use `--dry_run 1` before expensive training.
- Treat validator failures as blockers.
- If evaluation has no explicit model path, use the latest numeric
  `checkpoint-*` under `training.output_dir`.

## Reference Files

- Pipeline docs: `docs/finetune_pipeline.md`
- Example config: `configs/examples/uyghur_common_voice_pipeline.yaml`
- Pipeline CLI: `tools/qwen3_asr_pipeline.py`
- Validator CLI: `tools/validate_qwen3_asr_jsonl.py`

