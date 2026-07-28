#!/usr/bin/env bash
# Robust bilingual (Uyghur + Mandarin) Qwen3-ASR full-parameter fine-tune.
#
# Data:   Uyghur Common Voice (205h, explicit 'language Uyghur') +
#         MAGICDATA Mandarin downsampled to ~200h (explicit 'language Chinese').
# Aug:    speed perturbation [0.8,1.6] + add-noise (synthetic) + SpecAugment.
# Curriculum: epoch 0 explicit language labels; epoch>=1 derives 'language None'.
# Selection: keep the 3 checkpoints with lowest dev macro-CER (Uyghur+Mandarin),
#         evaluated on a 400-utt subset with deterministic dev speed perturbation.
set -euo pipefail
cd /root/autodl-tmp/workspace/project/Qwen3-ASR-FT

MODEL="/root/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5"
OUT="outputs/ug_zh_robust"
TRAIN="data/combined_ug_zh/train.jsonl"
DEV="data/combined_ug_zh/dev.jsonl"

mkdir -p "$OUT/logs"

# shellcheck disable=SC2086
/root/autodl-tmp/miniconda3/envs/qwen3_asr_ft/bin/python -u finetuning/qwen3_asr_sft.py \
  --model_path "$MODEL" \
  --train_file "$TRAIN" \
  --eval_file "$DEV" \
  --output_dir "$OUT" \
  --batch_size 4 --grad_acc 8 --lr 2e-5 --epochs 3 \
  --save_steps 3000 --log_steps 20 \
  --sr 16000 --warmup_ratio 0.02 --lr_scheduler_type linear \
  --num_workers 12 --pin_memory 1 --persistent_workers 1 --prefetch_factor 4 \
  --resume 1 \
  --augment 1 --augment_prob 1.0 \
  --speed_prob 0.5 --speed_factor_min 0.8 --speed_factor_max 1.6 \
  --noise_prob 0.3 --noise_snr_min 5.0 --noise_snr_max 20.0 \
  --specaug_prob 0.5 --specaug_time_mask_param 50 --specaug_freq_mask_param 27 \
  --specaug_num_time_masks 2 --specaug_num_freq_masks 2 \
  --curriculum 1 --curriculum_switch_epoch 1.0 \
  --save_best_metric wer --save_best_total_limit 3 \
  --wer_eval_samples 400 --wer_batch_size 4 \
  --eval_speed_aug 1 --eval_speed_min 0.8 --eval_speed_max 1.6 \
  2>&1 | tee "$OUT/logs/train.log"
