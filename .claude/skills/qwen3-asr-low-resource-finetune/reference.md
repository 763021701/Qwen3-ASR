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
| Eval scoring | `evaluation/*/eval_*_jsonl.py` | May have independent `normalize_for_scoring`; training JSONL should follow manual regardless |

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

## Checkpoint & Eval Loading (Qwen3-ASR)

Checkpoints saved by `finetuning/qwen3_asr_sft.py` via Transformers Trainer typically contain only weights + tokenizer — NOT Whisper-family feature extractor / chat template. `qwen_asr.inference.qwen3_asr.Qwen3ASRModel.from_pretrained(ckpt)` calls `AutoProcessor.from_pretrained(ckpt)` internally, so missing files surface during eval or standalone inference.

**Workaround** (verified): For each `checkpoint-*` to evaluate, copy from the training `--model_path` base (e.g. `Qwen/Qwen3-ASR-1.7B` HF cache snapshot):
- `preprocessor_config.json`
- `chat_template.json`

Place them alongside `model.safetensors` in the checkpoint directory.

**Long-term fix** (optional): call `processor.save_pretrained(output_dir)` on checkpoint save, or merge with base files.

## Existing Conversion Scripts (Reference)

- `tools/convert_to_qwen3_asr_jsonl.py` — general-purpose training/finetuning data converter; called by pipeline `prepare` stage
- `evaluation/chinese/wsc/prepare_wsc_eval_qwen3.py` — WSC / Sichuanhua eval
- `evaluation/cantonese/wsyue_asr/prepare_wsyue_asr_eval_qwen3.py` — Cantonese wsyue eval

## Eval Scripts

Per-language/per-task eval scripts under `evaluation/<lang>/eval_*_jsonl.py` with baselines in `evaluation/<lang>/baselines/`. See `evaluation/README.md` for common invocation patterns. For new languages, reuse the eval script closest in inference interface and change `--language` or JSONL path.
