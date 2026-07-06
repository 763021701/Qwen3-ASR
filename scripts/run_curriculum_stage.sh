#!/usr/bin/env bash
# Launch one curriculum CTC training stage. Reused for stages 1-4.
#   Usage: bash scripts/run_curriculum_stage.sh <stage> <model_path> <bucket_bs> [extra args...]
# stage 1: model_path = base Qwen3-ASR snapshot (fresh CTC).
# stage 2-4: model_path = previous stage's final checkpoint (CTC weights reloaded).
set -e
cd "$(dirname "$0")/.."
STAGE="$1"; MODEL_PATH="$2"; BUCKET_BS="$3"; shift 3
VOCAB=/root/.cache/modelscope/hub/models/FunAudioLLM/Fun-ASR-Nano-2512/multilingual.tiktoken
OUT="outputs/curriculum_ctc/stage${STAGE}"
mkdir -p "$OUT"
echo "[run_curriculum_stage] stage=${STAGE} model_path=${MODEL_PATH} bucket_bs=${BUCKET_BS}"
PYTHONPATH=. HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  python finetuning/qwen3_asr_sft.py \
  --train_ctc_only 1 --model_path "$MODEL_PATH" \
  --train_file "data/curriculum/stage${STAGE}.jsonl" \
  --output_dir "$OUT" \
  --epochs 1 --batch_size 32 --grad_acc 1 --lr 2e-5 \
  --warmup_ratio 0.02 --lr_scheduler_type linear \
  --curriculum_bucket 1 --bucket_bs "$BUCKET_BS" \
  --save_steps 2000 --save_total_limit 3 --log_steps 50 \
  --num_workers 4 --pin_memory 1 --persistent_workers 1 --prefetch_factor 2 \
  --ctc_vocab_path "$VOCAB" "$@" \
  > "$OUT/train.log" 2>&1
echo "stage${STAGE} exited with code $?"
