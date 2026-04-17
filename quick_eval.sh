  #!/usr/bin/env bash
  
  # python finetuning/eval_uyghur_asr_jsonl.py \
  #   --jsonl data/thuyg20_test_qwen3.jsonl \
  #   --model outputs/qwen3_asr_sft_ug2/checkpoint-19400 \
  #   --language Uyghur \
  #   --max_samples 2000 \
  #   --batch_size 16 \
  #   --output_predictions outputs/qwen3_asr_sft_ug2/checkpoint-1940/thuyg20_predictions.jsonl

  python baselines/eval_uyghur_asr_omnilingual_jsonl.py \
    --jsonl data/thuyg20_test_qwen3.jsonl \
    --model_card omniASR_LLM_1B_v2 \
    --omnilingual_lang uig_Arab \
    --max_samples 2000 \
    --batch_size 4 \
    --output_predictions outputs/omniASR_LLM_1B_v2/thuyg20_predictions.jsonl