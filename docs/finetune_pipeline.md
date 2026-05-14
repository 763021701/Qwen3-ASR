# Qwen3-ASR Finetuning Pipeline

This repo provides a thin pipeline wrapper around the existing data conversion,
finetuning, and evaluation scripts. The wrapper is intentionally small: it does
not replace `finetuning/qwen3_asr_sft.py`; it validates manifests and runs the
existing scripts in a reproducible order.

## Quick Start

Create a config from `configs/examples/uyghur_common_voice_pipeline.yaml`, then
run from the repository root:

```bash
python tools/qwen3_asr_pipeline.py --config configs/examples/uyghur_common_voice_pipeline.yaml --stage all
```

Stages can be run independently:

```bash
python tools/qwen3_asr_pipeline.py --config CONFIG.yaml --stage prepare
python tools/qwen3_asr_pipeline.py --config CONFIG.yaml --stage validate
python tools/qwen3_asr_pipeline.py --config CONFIG.yaml --stage train
python tools/qwen3_asr_pipeline.py --config CONFIG.yaml --stage eval
```

Use dry-run to inspect commands without launching training or evaluation:

```bash
python tools/qwen3_asr_pipeline.py --config CONFIG.yaml --stage all --dry_run 1
```

## Config Contract

The config has four sections:

- `dataset`: source type, language, source files, output manifest directory.
- `training`: arguments passed to `finetuning/qwen3_asr_sft.py`.
- `evaluation`: test script and inference/scoring options.
- `runtime`: launcher controls such as `python` vs `torchrun`.

Supported `dataset.source_type` values in v1:

- `common_voice`: uses TSV files and `clips_dir`.
- `kaldi`: uses `wav.scp` plus matching text files.
- `funasr_jsonl`: uses FunASR-style message JSONL.

The prepared manifests are:

- `train.jsonl`
- `dev.jsonl`
- `test.jsonl`

If only one source is provided through `dataset.source`, the pipeline converts it
to `all.jsonl` and splits it with `train_ratio`, `dev_ratio`, and `split_seed`.

Each line must contain:

```json
{"audio": "/absolute/audio.wav", "text": "language Uyghur<asr_text>transcript"}
```

## Validation

Run the validator directly when debugging data issues:

```bash
python tools/validate_qwen3_asr_jsonl.py \
  --jsonl data/uyghur/common_voice_pipeline/train.jsonl \
  --language Uyghur \
  --check_audio 1 \
  --output_report outputs/ug_common_voice_sft/validation/train_manifest_validation.json
```

Validation checks JSON syntax, required fields, audio existence, label format,
and whether the language is in `qwen_asr/inference/utils.py`.

## Utility Scripts

Training and data-preparation utilities live under repository-level `tools/`.
This keeps pre-training helpers separate from `evaluation/`, which is reserved
for eval drivers, baselines, and evaluation-output analysis.

- `tools/qwen3_asr_pipeline.py` — end-to-end prepare / validate / train / eval wrapper.
- `tools/convert_to_qwen3_asr_jsonl.py` — convert Common Voice, Kaldi, or FunASR-style data into Qwen3-ASR finetuning JSONL.
- `tools/validate_qwen3_asr_jsonl.py` — validate manifest schema, audio paths, and language tags.
- `tools/verify_tokenizer_cv_ug.py` — tokenizer round-trip smoke test for Common Voice Uyghur TSV text.

## Outputs

Training writes checkpoints under `training.output_dir`. Evaluation defaults to
the numerically latest `checkpoint-*` under that directory and writes
predictions to `training.output_dir/eval/predictions.jsonl`.
