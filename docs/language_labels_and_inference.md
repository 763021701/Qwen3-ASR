# Language Labels and Inference

Concise reference for Qwen3-ASR **training labels** vs **inference `language`**. Misalignment here is a common cause of **nospeech hallucination** after SFT.

## Training labels

Each jsonl `text` field is the full supervised target (including the prefix). Loss is computed on everything after the chat template, e.g.:

```json
{"audio": "speech.wav", "text": "language English<asr_text>the patient was admitted"}
{"audio": "nonspeech.wav", "text": "language None<asr_text>"}
```

| Sample type | Correct `text` | Wrong (causes problems) |
|-------------|------------------|---------------------------|
| Speech with known language | `language English<asr_text>{transcript}` | `language None<asr_text>{transcript}` if you need LID |
| Nospeech / silence / noise-only | `language None<asr_text>` (empty after tag) | `language English<asr_text>` with empty body |
| Unknown language, has speech | `language None<asr_text>{transcript}` | — |

Notes:

- `<asr_text>` must be followed by **nothing** for nospeech (no spaces, no placeholder text).
- Official finetuning: `language None` on a sample does **not** teach language ID from that prefix.
- `tools/convert_to_qwen3_asr_jsonl.py` rejects empty transcripts; nospeech rows must be written directly into jsonl.

## Inference: three `language` modes

| Call | Prompt suffix | Model generates | Use when |
|------|---------------|-----------------|----------|
| Omit `language` / `language=None` / `language=""` | *(none)* | Full string, e.g. `language English<asr_text>...` | **Auto LID + ASR** (official default) |
| `language="English"` (etc.) | `language English<asr_text>` | Transcript only | Language is known |
| `language="None"` (**string**) | `language None<asr_text>` | Transcript only | Force None prefix; **not** auto-detect |

Python `None` and the string `"None"` are **different**. Auto mode = do not pass `language`. String `"None"` = forced prompt prefix.

Parsing (auto mode): `language None<asr_text>` with empty body → `("", "")` (nospeech). See `qwen_asr/inference/utils.py` → `parse_asr_output`.

## Train / infer alignment

| Goal | Training | Inference |
|------|----------|-----------|
| Auto language detection | Explicit `language English/Chinese/...<asr_text>{text}` on speech | Omit `language` |
| Nospeech rejection | `language None<asr_text>` on silence segments | Omit `language` (same as LID) |
| Fixed-language ASR | `language English<asr_text>{text}` | `language="English"` |

Do not train silence as `language English<asr_text>` and expect base-model nospeech behavior at inference.

## Pitfalls in this repo

1. **`tools/prepare_poc_train.py`** hardcodes `ASR_PREFIX = "language English<asr_text>"` for all sources, including silence. Silence rows become `language English<asr_text>` with an empty body — use `language None<asr_text>` instead.
2. **Eval scripts** (e.g. `evaluation/chinese_english/eval_chinese_english_asr_jsonl.py`) default to `--language None` (string) = **forced None mode**, not auto-detect. For LID + nospeech eval matching training, pass an empty language or omit the flag if the script maps that to Python `None`.
3. **`--curriculum`** (swap labels to `language None<asr_text>{text}`) is a fork extension, not upstream official SFT. It does not replace explicit language labels for teaching auto LID.

## Quick checklist

- [ ] Speech: correct atomic language in the label (`English`, `Chinese`, …).
- [ ] Nospeech: `language None<asr_text>` only; include dedicated nospeech audio in train mix.
- [ ] Inference for LID + nospeech: do not pass `language` (not the string `"None"`).
- [ ] Eval flags match the inference mode you deploy.

See also: `finetuning/README.md`, `docs/finetune_pipeline.md`.
