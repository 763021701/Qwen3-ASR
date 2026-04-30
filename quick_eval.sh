#!/usr/bin/env bash

# python evaluation/uyghur/eval_uyghur_asr_jsonl.py \
#     --jsonl data/uyghur/common_voice/ug_test_qwen3.jsonl \
#     --model outputs/qwen3_asr_sft_ug_norm/checkpoint-11934 \
#     --language Uyghur \
#     --max_samples 2000 \
#     --batch_size 16 \
#     --output_predictions outputs/qwen3_asr_sft_ug_norm/ug_test_qwen3_predictions.jsonl

# python evaluation/uyghur/baselines/eval_uyghur_asr_omnilingual_jsonl.py \
#   --jsonl data/uyghur/thuyg20/thuyg20_test_qwen3.jsonl \
#   --model_card omniASR_LLM_1B_v2 \
#   --omnilingual_lang uig_Arab \
#   --max_samples 2000 \
#   --batch_size 4 \
#   --output_predictions outputs/omniASR_LLM_1B_v2/thuyg20_predictions.jsonl


# python evaluation/cantonese/eval_cantonese_asr_jsonl.py \
#   --jsonl data/cantonese/common_voice_yue/cv_yue_test_qwen3.jsonl \
#   --model /root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5 \
#   --batch_size 16 \
#   --output_predictions outputs/qwen3_asr/cantonese_eval/predictions.jsonl

# python evaluation/cantonese/eval_cantonese_asr_jsonl.py \
#   --jsonl data/cantonese/wsyue_asr/wsyue_asr_eval_qwen3.jsonl \
#   --model /root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5 \
#   --batch_size 16 \
#   --output_predictions outputs/qwen3_asr/wsyue_eval/predictions.jsonl

# python evaluation/cantonese/eval_cantonese_asr_jsonl.py \
#   --jsonl data/cantonese/common_voice_hk/cv_hk_test_qwen3.jsonl \
#   --model /root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5 \
#   --batch_size 16 \
#   --output_predictions outputs/qwen3_asr/cv_hk/predictions.jsonl

# 豆包 ASR 流式，四川话
# source .prjenv
# python evaluation/cantonese/baselines/eval_cantonese_asr_doubao_streaming_jsonl.py \
#   --jsonl data/chinese/wsc/wsc_eval_qwen3.jsonl \
#   --pace_ms 200 \
#   --concurrency 20 \
#   --hanzi_script_norm to_simplified \
#   --output_predictions outputs/doubao_asr/wsc_eval/predictions.jsonl

# 豆包 ASR 流式，粤语
# source .prjenv
# python evaluation/cantonese/baselines/eval_cantonese_asr_doubao_streaming_jsonl.py \
#   --jsonl data/cantonese/common_voice_hk/cv_hk_test_qwen3.jsonl \
#   --pace_ms 200 \
#   --concurrency 20 \
#   --hanzi_script_norm to_traditional \
#   --output_predictions outputs/doubao_asr/cv_hk_eval/predictions.jsonl

# python evaluation/cantonese/baselines/eval_cantonese_asr_omnilingual_jsonl.py \
#   --jsonl data/cantonese/common_voice_hk/cv_hk_test_qwen3.jsonl \
#   --model_card omniASR_LLM_7B_v2 \
#   --omnilingual_lang yue_Hant \
#   --batch_size 2 \
#   --output_predictions outputs/omniASR_LLM_7B_v2/cv_hk_predictions.jsonl

# python evaluation/cantonese/eval_cantonese_asr_jsonl.py \
#   --jsonl data/chinese/wsc/wsc_eval_qwen3.jsonl \
#   --model /root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5 \
#   --language Chinese \
#   --hanzi_script_norm to_simplified \
#   --batch_size 16 \
#   --output_predictions outputs/qwen3_asr/wsc_eval/predictions.jsonl


# Paraformer, 四川话
# python evaluation/cantonese/baselines/eval_cantonese_asr_paraformer_jsonl.py \
#   --jsonl data/chinese/wsc/wsc_eval_qwen3.jsonl \
#   --model dengcunqin/speech_paraformer-large_asr_nat-chuan-16k-common-vocab8404-pytorch \
#   --batch_size 16 \
#   --eval_label "Sichuan Paraformer" \
#   --hanzi_script_norm to_simplified \
#   --output_predictions outputs/paraformer/wsc_eval/predictions.jsonl

# Paraformer, 粤语
python evaluation/cantonese/baselines/eval_cantonese_asr_paraformer_jsonl.py \
  --jsonl data/cantonese/common_voice_hk/cv_hk_test_qwen3.jsonl \
  --model dengcunqin/speech_seaco_paraformer_large_asr_nat-zh-cantonese-en-16k-common-vocab11666-pytorch \
  --batch_size 16 \
  --eval_label "Yue Paraformer" \
  --hanzi_script_norm to_traditional \
  --output_predictions outputs/paraformer/cv_hk_eval/predictions.jsonl
