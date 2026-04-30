#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/project/Qwen3-ASR

python3 finetuning/qwen3_asr_sft.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --train_file /root/autodl-tmp/project/developing/Qwen3-ASR/data/uyghur/common_voice/ug_train_qwen3_uls.jsonl \
  --eval_file /root/autodl-tmp/project/developing/Qwen3-ASR/data/uyghur/common_voice/ug_dev_qwen3_uls.jsonl \
  --output_dir /root/autodl-tmp/project/developing/Qwen3-ASR/outputs/qwen3_asr_sft_uls \
  --batch_size 4 \
  --grad_acc 8 \
  --lr 2e-5 \
  --epochs 3 \
  --save_steps 300 \
  --save_total_limit 3 \
  --log_steps 20 \
  --sr 16000
