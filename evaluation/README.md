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

## See also

- `data/README.md` — manifest locations
- `quick_eval.sh` — commented example invocations
