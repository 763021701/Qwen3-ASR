---
name: qwen3-asr-low-resource-finetune
description: End-to-end Qwen3-ASR fine-tuning for new corpora or low-resource languages — jsonl manifest contract, conversion scripts, tokenizer checks, optional online audio augmentation, training and eval. Invoke when the user adds a new speech dataset, adapts a new language, asks for ASR SFT data prep, label normalization, data augmentation, or mentions Qwen3-ASR jsonl / prepare_* / verify_tokenizer / normalize_label.
---

# Qwen3-ASR Low-Resource Language & New Corpus Fine-Tuning

## Overview

Complete workflow for adding a new dataset or low-resource language to Qwen3-ASR: raw data exploration → pipeline config → JSONL generation → validation → training → evaluation.

**Unified entry point:**

```bash
python tools/qwen3_asr_pipeline.py --config CONFIG.yaml --stage all
```

**Dry-run first:**

```bash
python tools/qwen3_asr_pipeline.py --config CONFIG.yaml --stage all --dry_run 1
```

## Mandatory Pre-Flight Confirmation

Before writing any config, conversion script, or launching training, you MUST ask the user when any parameter is uncertain. Do not silently choose. At minimum, cover:

- **Data split**: Does train/dev/test already exist? If splitting is needed, confirm ratios and `split_seed`.
- **Language label**: If the language is not in `SUPPORTED_LANGUAGES` (in `qwen_asr/inference/utils.py`), confirm whether to use a neighboring label or extend the list.
- **Transcript normalization**: Strategy for numbers, punctuation, simplified/traditional Chinese, case, code-switched text.
- **Training budget**: Model checkpoint, epochs, batch/grad_acc, learning rate, checkpoint frequency, resume.
- **Online augmentation**: Whether to enable `training.augment`. If yes, confirm SpeedPerturbation, AddNoise, SpecAugment probabilities and intensity. Default recommendation: start without augmentation; enable for low-resource or noisy domain shift.
- **Other**: Tokenizer spot-check, `max_samples` smoke test, dry-run, eval scripts/metrics, stage-only runs.

If the user has no preference, explicitly state the conservative defaults you will use and show the key parameters before writing config or launching.

## Data Contract (Mandatory)

Training/validation files are **JSONL**, one JSON object per line:

| Field | Required | Description |
|-------|----------|-------------|
| `audio` | Yes | Locally readable audio path (absolute path recommended); collator resamples via librosa to `--sr` (default **16000**) |
| `text` | Yes | **Full** supervised string, format: `language {LanguageSpec}<asr_text>{transcript}` — no extra newlines breaking this pattern. `{LanguageSpec}` is a **single** language from `SUPPORTED_LANGUAGES`, or **comma-separated** code-switched labels (e.g. `Chinese,English`). Spaces around commas are optional. |
| `prompt` | No | If present, used as chat prefix; usually omitted for ASR SFT |

`{LanguageSpec}` rules:
- Each comma-separated atom must exist in `SUPPORTED_LANGUAGES` in `qwen_asr/inference/utils.py`.
- Composite names like `Chinese,English` are NOT added to `SUPPORTED_LANGUAGES` as a single entry.
- `normalize_language_spec` reorders multiple atoms by `SUPPORTED_LANGUAGES` definition order (so `English,Chinese` normalizes to `Chinese,English`).
- If a corpus language is **not** in the list: ask the user whether to use the nearest existing label or extend `SUPPORTED_LANGUAGES` (and sync with inference/eval scripts).

**Reference implementations** for prefix construction:
- `evaluation/chinese/wsc/prepare_wsc_eval_qwen3.py` (WSC-Eval / Sichuanhua; `_TEXT_PREFIX`)
- `evaluation/cantonese/wsyue_asr/prepare_wsyue_asr_eval_qwen3.py`

**Existing JSONL example**: `data/uyghur/common_voice/ug_train_qwen3.jsonl`

## Transcript Normalization (Read Before Writing JSONL Scripts)

**Before writing or extending any JSONL-generating script**, read the authoritative manual at `docs/normalize_label.md`.

`normalize_target_text` / `format_label.py` only prepend `language ...<asr_text>` — they do NOT strip punctuation or clean the transcript body per language. Normalization must happen **before** prefix assembly.

| `SUPPORTED_LANGUAGES` label | Manual section |
|-----------------------------|----------------|
| `English` | ENGLISH |
| `Chinese` | MANDARIN |
| `Cantonese` | CANTONESE |
| `Chinese,English` (code-switched) | Apply per-utterance/word-level rules from each section, or confirm unified strategy with user |
| Languages not in manual (e.g. `Uyghur`) | Confirm with user: follow GLOBAL RULES + neighboring language rules, or define custom |

**GLOBAL RULES** (from manual):
- Same rules for train/dev/test
- Unicode normalization
- Remove invisible/control characters and non-speech annotations
- Default: remove punctuation (unless user explicitly wants punctuation prediction)
- Normalize whitespace
- Discard samples with empty post-normalization transcript
- Consistent number policy across all datasets

**Implementation options** (choose one, document in script):
1. Add `--normalize_transcript` / `--label_locale` flags to `tools/convert_to_qwen3_asr_jsonl.py`
2. Create `tools/normalize_transcript.py` for `prepare_*` scripts to import
3. Inline in dedicated `tools/prepare_*.py`, matching manual rules; report `skipped_empty_after_normalize` in `--report_json`

After normalization, use `qwen3_asr_supervised_text()` from `scripts/format_label.py` to build the final `text` field.

## Workflow Checklist (Execute in Order)

```
- [ ] 0. Ask user to confirm all uncertain parameters (split, normalize, budget, augment)
- [ ] 1. Explore raw data: directory structure, metadata format, audio paths/extensions
- [ ] 2. Select language label from SUPPORTED_LANGUAGES; confirm with user if missing
- [ ] 2b. Read docs/normalize_label.md; decide transcript normalization strategy
- [ ] 3. Write/update configs/*.yaml and conversion script; normalize transcripts before writing JSONL
- [ ] 4. Validate: python tools/qwen3_asr_pipeline.py --stage validate
- [ ] 5. Tokenizer: sample-encode final text strings, check for UNK tokens and decode round-trip
- [ ] 6. Train: python tools/qwen3_asr_pipeline.py --config CONFIG.yaml --stage train
- [ ] 7. Evaluate: python tools/qwen3_asr_pipeline.py --config CONFIG.yaml --stage eval
```

## Writing Conversion Scripts for New Datasets

Prefer extending `tools/convert_to_qwen3_asr_jsonl.py` (called by `tools/qwen3_asr_pipeline.py --stage prepare`). Only create `evaluation/<lang>/<dataset>/prepare_*.py` for eval-specific scripts with download/subset/dedup logic.

Requirements:
1. **Read `docs/normalize_label.md` first** — never assume "strip + prefix" is sufficient.
2. Use `argparse` with at minimum: `--output_jsonl`, `--dataset_dir`, `--language`, `--max_samples` (0 = all). If the language has a manual chapter, add matching flags (e.g. `--number_policy`, `--hanzi_script`).
3. Write JSONL line by line: `json.dumps({"audio": abs_path, "text": supervised}, ensure_ascii=False)`, file in **UTF-8**.
4. Pipeline: `raw_transcript` → **normalize per manual** → `qwen3_asr_supervised_text(language, normalized)`.
5. Use `os.path.abspath` or pathlib for audio paths to avoid cwd-dependent failures.
6. Log/count missing audio, empty pre-normalization transcripts, and empty post-normalization strings. Never silently emit bad samples.
7. Train/dev/test share the **same** normalization implementation (manual GLOBAL RULES).

Do NOT modify `finetuning/qwen3_asr_sft.py` for language-specific logic — language info belongs only in the data `text` field. Only change the training script for LoRA/multi-GPU etc.

## Training Quick Reference

- **Script**: `finetuning/qwen3_asr_sft.py`
- **Data**: `load_dataset("json", data_files=...)` — requires `audio` + `text`
- **Key args**: `--model_path`, `--train_file`, `--eval_file`, `--output_dir`, `--sr`, `--batch_size`, `--grad_acc`, `--lr`, `--epochs`, `--save_steps`, `--resume` / `--resume_from`
- **Online augmentation** (default off, train collator only): `--augment 1` to enable; `--augment_prob`, `--speed_prob`, `--speed_factors`, `--noise_prob`, `--noise_snr_min`, `--noise_snr_max`, `--specaug_prob`, `--specaug_time_mask_param`, `--specaug_freq_mask_param`, `--specaug_num_time_masks`, `--specaug_num_freq_masks`. In pipeline YAML, write under `training:`.

**Augmentation strategy**:
- Start with `augment: 0` for baseline.
- For low-resource or limited speaker/acoustic coverage: try `augment: 1`, `speed_factors: "0.9,1.0,1.1"`, moderate `noise_prob`, low-to-moderate `specaug_prob`.
- AddNoise currently synthesizes white noise; for real background noise, confirm noise data path, sample rate, licensing, and mixing approach with user first.

## Additional Resources

- **Transcript normalization manual** (mandatory reading): `docs/normalize_label.md`; summary in `reference.md` "标签规范化".
- **Troubleshooting, environment, checkpoint eval loading**: maintained in `reference.md`.
- **CLI helper for building `text` field**: `scripts/format_label.py` (does NOT normalize — normalize first, then pass with `--transcript`):
  ```bash
  python .claude/skills/qwen3-asr-low-resource-finetune/scripts/format_label.py --language English --transcript "..."
  ```
