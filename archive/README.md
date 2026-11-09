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

## 2026-10-21

- Radiology-only oracle SFT (1 epoch + 5 epochs), diagnostic for whether target-domain
  data teaches the radiology scene. Configs, manifests, checkpoints and eval results
  moved from the repo root; `tools/prepare_radiology_oracle.py` stays in `tools/`.
  Results: `outputs/poc_radiology_oracle_5ep/RESULTS.md` and
  `outputs/poc_radiology_oracle_5ep/epoch_summary.json`.

## 2026-10-22

Conservative output cleanup (~116GB freed): removed redundant checkpoints only, keeping
one model per experiment and all optimizer states. Deletions: yue run 1300/1900/4557,
radiology oracle 5ep 76/152/228/304, radiology oracle 1ep weights, historical 2voice
checkpoint, old Uyghur/OPD checkpoints, LoRA merged_best and non-selected adapters,
and the LoRA preflight directory.

## 2026-11-09

Terminated the training-time auxiliary CTC head (negative result; see
`docs/ctc_auxiliary.md` and `docs/ctc_diagnostic_report.md`).

- `ctc_auxiliary/` — implementation, unit test and diagnostic tools
  (`finetuning/ctc_auxiliary.py`, `tests/test_ctc_auxiliary.py`, `tools/`).
- `configs/` — `poc_radiology_mix_m1_ctc_{1ep,v2_025ep,v3_cov_025ep,v3_cov_1ep}.yaml`.
- `data/` — `ctc_diag_32.jsonl` (32-clip diagnostic set).
- `outputs/` — `ctc_diagnostic/`, all `poc_radiology_mix_m1_ctc_*` runs, and the
  CTC training nohup logs.
- The tracked SFT/pipeline CTC hooks were reverted. The concurrent AuT-only
  (frozen-LLM) control config, outputs and doc were deleted. Docs stay under
  `docs/` as the negative-result record.
- Archived code keeps its original imports and paths; it is not runnable in
  place without restoring the `finetuning/` / `tools/` layout.
