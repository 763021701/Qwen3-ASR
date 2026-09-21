# Archive

Expired / off-mainline recipes.

Keep using `configs/`, `data/`, and `outputs/` at the repo root for the current POC mainline:

- plus0821 SFT (`checkpoint-3900`)
- GRPO loop v1 / v2
- long-audio eval
- hotword context probe (diagnostic)

Do not re-run archived recipes as the competitive line. YAML files still use original `data/` and `outputs/` paths from when they were live.

## 2026-09-14

- `configs/` — balanced multilingual, medical TTS, pathology-en, freeze / RAFT / oversample / 2voice-clean ablations
- `data/` — matching manifests (Uyghur, GigaSpeech, SwitchLingua, etc.)
- `outputs/` — matching runs, failed/empty POC dirs, off-mainline evals

## 2026-09-21

Moved concluded, non-competitive POC SFT/eval after plus0821 became the working checkpoint:

- Historical SFT: TCGA 1to3 / v2, real-only, raw+denoised, LLM-syn, 2voice, fullvoice without 0821, giga stage1, noise-ext
- Negative prompt-hotword continue-SFT from 3900 (`poc_sft3900_real_hotword_prompt_ft`; weights already deleted)
- Soup interpolations with base, GRPO smoke/screen/baseline logs
- Matching `data/` manifests and eval directories
