# Qwen3-ASR CTC Decoder Implementation

This document summarizes the added CTC branch for Qwen3-ASR. The goal is to produce a coarse ASR result from the Qwen3-ASR audio tower while keeping the original LLM-based recognition path unchanged.

## Design

The CTC branch is an auxiliary path attached to the Qwen3-ASR audio tower. It does not replace the normal Qwen3-ASR decoder.

```text
input_features
  -> Qwen3-ASR audio_tower
       -> ln_post hidden states, audio d_model dim
       -> CTC decoder
       -> CTC head
       -> greedy CTC text / optional timestamps

normal path remains:
input_features
  -> audio_tower proj1/proj2
  -> LLM audio embeddings
  -> Qwen3-ASR text generation
```

The branch uses the audio hidden states after `ln_post` and before `proj1/proj2`. This keeps the CTC input closer to acoustic encoder features instead of using the final LLM-oriented audio embeddings.

## Architecture

The CTC decoder follows the Fun-ASR-Nano style at a smaller integration boundary:

- Input: Qwen3-ASR audio hidden states; dimension is inferred from `audio_config.d_model` (`1024` for the cached Qwen3-ASR-1.7B model used in smoke training).
- Projection: `Linear(input_dim -> 2048) -> ReLU -> Linear(2048 -> 512)`.
- Context modeling: 5 Transformer-style self-attention blocks.
- CTC head: `Linear(512 -> 60515)` followed by `log_softmax`.
- Blank id: `60514`.
- Time step: `0.08s`, matching 12.5Hz.

The CTC vocabulary and tokenizer are Fun-ASR/SenseVoice compatible. Training uses `multilingual.tiktoken` through `SenseVoiceTokenizer`.

## Implementation

The implementation is intentionally opt-in.

- `Qwen3ASRAudioEncoder.forward(..., return_ctc_hidden=True)` returns the pre-projection CTC hidden states through `BaseModelOutput.hidden_states`.
- `Qwen3ASRCTCDecoder` and `Qwen3ASRCTCHead` implement the auxiliary decoder and CTC projection.
- `Qwen3ASRThinkerForConditionalGeneration.enable_ctc()` creates the CTC modules and stores `ctc_config`.
- `get_ctc_logits()` runs audio tower hidden extraction, CTC decoder, and CTC head.
- `ctc_loss()` computes standard PyTorch CTC loss.
- `decode_ctc_logits()` performs greedy CTC decoding: argmax, collapse repeats, remove blank, decode ids.
- `generate_ctc()` is a convenience wrapper for CTC-only inference.

Default model behavior is unchanged because `ctc_config` defaults to `{"enabled": false}`.

## Training

The existing finetuning script supports CTC-only training:

```bash
python finetuning/qwen3_asr_sft.py \
  --model_path /path/to/qwen3-asr \
  --train_file train.jsonl \
  --output_dir ./qwen3-asr-ctc-out \
  --train_ctc_only 1 \
  --ctc_vocab_path /path/to/multilingual.tiktoken \
  --funasr_path ../FunASR
```

When `--train_ctc_only 1` is set:

- Qwen3-ASR audio tower and LLM parameters are frozen.
- Only `ctc_decoder` and `ctc_head` are trainable.
- The collator encodes dataset `text` with `SenseVoiceTokenizer` and emits `ctc_labels` plus `ctc_label_lengths`.
- The model forward path returns CTC loss directly and skips LLM loss.

The tokenizer loader first tries the SenseVoice tokenizer source file from `--funasr_path`/`FUNASR_PATH`. If that file is unavailable, it falls back to the normal FunASR import.

## Output

CTC inference returns dictionaries shaped like:

```python
{
    "ctc_token_ids": [...],
    "ctc_text": "...",
    "ctc_timestamps": [
        {"token": "...", "start_time": 0.0, "end_time": 0.08},
        ...,
    ],
}
```

Timestamps are coarse and derived from CTC frame indices using `frame_index * 0.08` seconds.

## Validation

The implementation was checked with:

- Python compile checks for the modified modeling, config, and finetuning files.
- A small CTC decoder/head smoke test covering tensor shapes and finite CTC loss.
- Direct loading of Fun-ASR `multilingual.tiktoken` through the SenseVoice tokenizer source.
- A 16-sample CTC-only smoke training run on the cached Qwen3-ASR-1.7B model completed 16/16 steps with finite loss.
- `git diff --check` for whitespace issues.

Known environment note: full finetuning still requires the original script dependencies such as `datasets`. The lightweight tokenizer fallback avoids requiring the full FunASR dependency stack just to build CTC labels.
