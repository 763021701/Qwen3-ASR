# Evaluation layout

Scripts and baselines are grouped by **language**, then by **dataset** or role. Shared utilities live in `evaluation/tools/`.

Run commands from the **repository root** so `qwen_asr` imports resolve.

## `tools/` (language-agnostic)

| Script | Purpose |
|--------|---------|
| `tools/convert_to_qwen3_asr_jsonl.py` | Build Qwen3-ASR training jsonl from wav.scp / other sources |
| `tools/extract_ref_hyp_csv.py` | Extract reference / hypothesis columns to CSV |
| `tools/verify_tokenizer_cv_ug.py` | Tokenizer round-trip check (example: Common Voice TSV) |

## `chinese/`

| Path | Purpose |
|------|---------|
| `chinese/wsc/prepare_wsc_eval_qwen3.py` | Build jsonl for WSC-Eval-ASR (**Sichuan Mandarin / 四川话**, not Cantonese) |

Manifest: `data/chinese/wsc/wsc_eval_qwen3.jsonl`.

## `cantonese/`

| Path | Purpose |
|------|---------|
| `cantonese/eval_cantonese_asr_jsonl.py` | Qwen3-ASR checkpoint eval (CER-oriented scoring; also used with `--language Chinese` for Sichuan-style manifests) |
| `cantonese/wsyue_asr/prepare_wsyue_asr_eval_qwen3.py` | Build jsonl for WSYue-ASR-eval (Cantonese) |
| `cantonese/baselines/` | Third-party / baseline eval drivers (Paraformer, Doubao, Omnilingual, FireRed, etc.) |

Default manifests for Cantonese live under `data/cantonese/<dataset>/`.

## `uyghur/`

| Path | Purpose |
|------|---------|
| `uyghur/eval_uyghur_asr_jsonl.py` | Qwen3-ASR checkpoint eval (WER-oriented scoring) |
| `uyghur/uls/convert_umsc_uas_to_uls_jsonl.py` | UAS → ULS jsonl conversion (umsc) |
| `uyghur/uls/verify_umsc_roundtrip_jsonl.py` | umsc round-trip check on jsonl |
| `uyghur/baselines/` | Whisper HF, Omnilingual, etc. |

Manifests live under `data/uyghur/common_voice/` and `data/uyghur/thuyg20/`.

## Example commands

Prefer the pipeline wrapper for end-to-end finetuning and eval:

```bash
python tools/qwen3_asr_pipeline.py \
  --config configs/examples/uyghur_common_voice_pipeline.yaml \
  --stage eval
```

Use the task scripts directly when comparing checkpoints, datasets, or baselines.
Replace `MODEL_OR_CHECKPOINT` with a local checkpoint or Hugging Face model path.

### Uyghur

Qwen3-ASR checkpoint on Common Voice:

```bash
python evaluation/uyghur/eval_uyghur_asr_jsonl.py \
  --jsonl data/uyghur/common_voice/ug_test_qwen3.jsonl \
  --model MODEL_OR_CHECKPOINT \
  --language Uyghur \
  --max_samples 2000 \
  --batch_size 16 \
  --output_predictions outputs/qwen3_asr_sft_ug_norm/ug_test_qwen3_predictions.jsonl
```

Omnilingual baseline on THUYG-20:

```bash
python evaluation/uyghur/baselines/eval_uyghur_asr_omnilingual_jsonl.py \
  --jsonl data/uyghur/thuyg20/thuyg20_test_qwen3.jsonl \
  --model_card omniASR_LLM_1B_v2 \
  --omnilingual_lang uig_Arab \
  --max_samples 2000 \
  --batch_size 4 \
  --output_predictions outputs/omniASR_LLM_1B_v2/thuyg20_predictions.jsonl
```

### Cantonese and WSC

Qwen3-ASR checkpoint on Cantonese manifests:

```bash
python evaluation/cantonese/eval_cantonese_asr_jsonl.py \
  --jsonl data/cantonese/common_voice_yue/cv_yue_test_qwen3.jsonl \
  --model MODEL_OR_CHECKPOINT \
  --batch_size 16 \
  --output_predictions outputs/qwen3_asr/cantonese_eval/predictions.jsonl

python evaluation/cantonese/eval_cantonese_asr_jsonl.py \
  --jsonl data/cantonese/wsyue_asr/wsyue_asr_eval_qwen3.jsonl \
  --model MODEL_OR_CHECKPOINT \
  --batch_size 16 \
  --output_predictions outputs/qwen3_asr/wsyue_eval/predictions.jsonl

python evaluation/cantonese/eval_cantonese_asr_jsonl.py \
  --jsonl data/cantonese/common_voice_hk/cv_hk_test_qwen3.jsonl \
  --model MODEL_OR_CHECKPOINT \
  --batch_size 16 \
  --output_predictions outputs/qwen3_asr/cv_hk/predictions.jsonl
```

Qwen3-ASR on WSC-Eval-ASR with Chinese inference and simplified scoring:

```bash
python evaluation/cantonese/eval_cantonese_asr_jsonl.py \
  --jsonl data/chinese/wsc/wsc_eval_qwen3.jsonl \
  --model MODEL_OR_CHECKPOINT \
  --language Chinese \
  --hanzi_script_norm to_simplified \
  --batch_size 16 \
  --output_predictions outputs/qwen3_asr/wsc_eval/predictions.jsonl
```

### Baselines

Doubao streaming ASR examples require API credentials from the local project
environment before running the command.

```bash
source .prjenv
python evaluation/cantonese/baselines/eval_cantonese_asr_doubao_streaming_jsonl.py \
  --jsonl data/chinese/wsc/wsc_eval_qwen3.jsonl \
  --pace_ms 200 \
  --concurrency 20 \
  --hanzi_script_norm to_simplified \
  --output_predictions outputs/doubao_asr/wsc_eval/predictions.jsonl

source .prjenv
python evaluation/cantonese/baselines/eval_cantonese_asr_doubao_streaming_jsonl.py \
  --jsonl data/cantonese/common_voice_hk/cv_hk_test_qwen3.jsonl \
  --pace_ms 200 \
  --concurrency 20 \
  --hanzi_script_norm to_traditional \
  --output_predictions outputs/doubao_asr/cv_hk_eval/predictions.jsonl
```

Omnilingual and Paraformer examples:

```bash
python evaluation/cantonese/baselines/eval_cantonese_asr_omnilingual_jsonl.py \
  --jsonl data/cantonese/common_voice_hk/cv_hk_test_qwen3.jsonl \
  --model_card omniASR_LLM_7B_v2 \
  --omnilingual_lang yue_Hant \
  --batch_size 2 \
  --output_predictions outputs/omniASR_LLM_7B_v2/cv_hk_predictions.jsonl

python evaluation/cantonese/baselines/eval_cantonese_asr_paraformer_jsonl.py \
  --jsonl data/chinese/wsc/wsc_eval_qwen3.jsonl \
  --model dengcunqin/speech_paraformer-large_asr_nat-chuan-16k-common-vocab8404-pytorch \
  --batch_size 16 \
  --eval_label "Sichuan Paraformer" \
  --hanzi_script_norm to_simplified \
  --output_predictions outputs/paraformer/wsc_eval/predictions.jsonl

python evaluation/cantonese/baselines/eval_cantonese_asr_paraformer_jsonl.py \
  --jsonl data/cantonese/common_voice_hk/cv_hk_test_qwen3.jsonl \
  --model dengcunqin/speech_seaco_paraformer_large_asr_nat-zh-cantonese-en-16k-common-vocab11666-pytorch \
  --batch_size 16 \
  --eval_label "Yue Paraformer" \
  --hanzi_script_norm to_traditional \
  --output_predictions outputs/paraformer/cv_hk_eval/predictions.jsonl
```

## See also

- `data/README.md` — manifest locations
- `docs/finetune_pipeline.md` — pipeline wrapper usage
