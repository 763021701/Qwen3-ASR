#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/project/Qwen3-ASR

python3 finetuning/qwen3_asr_sft.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --train_file /root/autodl-tmp/project/Qwen3-ASR/data/cv_ug_tr_zhCN_train_qwen3.jsonl \
  --eval_file /root/autodl-tmp/project/Qwen3-ASR/data/ug_dev_qwen3.jsonl \
  --output_dir /root/autodl-tmp/project/Qwen3-ASR/outputs/qwen3_asr_sft_ug3 \
  --batch_size 4 \
  --grad_acc 8 \
  --lr 2e-5 \
  --epochs 3 \
  --save_steps 300 \
  --save_total_limit 3 \
  --log_steps 20 \
  --sr 16000
