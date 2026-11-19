# Robust ASR

本分支在 Qwen3-ASR-1.7B 上做鲁棒性 LoRA 训练。目标是加声学退化之后，中文、英文、粤语、中文方言相对基座不退化，并用同一套数据比较训练方案。当前协议是 1000 小时混料，500 小时是同一次运行的检查点，不是另一份数据。

基座模型仍是上游 `qwen_asr`。本分支不维护官方推理、Demo 和部署文档。

## 数据约定

- **train**：各数据集的 train，只做训练。
- **dev**：各数据集的 dev，只做开发集。不放进训练混合。
- **评测**：训练结束后再跑。评测集固定为 HM-ASR-Benchmark，直接调用 `/root/autodl-tmp/workspace/project/hmasr_benchmark`。

```bash
cd /root/autodl-tmp/workspace/project/hmasr_benchmark
PYTHONPATH=src python -m asr_benchmark run \
  --model-path /path/to/checkpoint \
  --output-dir outputs/run_name
```

dev 的路径不完全同名，这些都不要拿去训练：`dev/`、WenetSpeech 的 `DEV/`、`webdataset/gigaspeech-dev-00000.tar`、`kespeech-dev_phase1/2-*.tar`、AliMeeting 的 `near/dev` 与 `far/dev`。

语料在 ModelScope 的 `huangpppu/*`。吴语 `Internet-Dialect-Wu` 目前没有 dev。

## Pipeline

一次训练由两份 YAML 驱动：

- `configs/robust_mix_1000h.yaml`：模型、LoRA、步数、batch。入口是 `tools/qwen3_asr_pipeline.py --stage train`，实际跑 `finetuning/qwen3_asr_sft_lora.py`。
- `configs/data_configs/robust_mix_1000h.yaml`：数据源、小时配额、缓存和增强。

数据流：

1. `finetuning/data_pipeline` 只索引各源的 **train**。ModelScope shard 按需下载到有上限的本地缓存。
2. `weight_mode: hours` 按小时配额混合。标签写成 `language {Name}<asr_text>{text}`。
3. 开仿真时，`finetuning/augmentation/service.py` 用独立进程读干净音频、做退化、写入有上限的 spool，训练器再从 spool 读。Speed 在仿真进程里做，SpecAugment 在 collator 里做。
4. 仿真场景固定 10 个：`noise`、`far_field`、`obstructed`、`echo_reverb`、`recording`、`distortion`、`dropout`、`far_field+noise`、`echo_reverb+noise`、`obstructed+noise`。`clean_prob: 0.25`。`alimeeting_far` 和 `ami_sdm` 本身是远场，不做仿真。

1000 小时配置：LoRA all，r=16，alpha=32，lr 2e-5，有效 batch 32，`max_steps=26800`。`save_steps=13400` 留下约 500 小时的检查点。

扩到万小时时不要把这套权重乘 10。会议、吴语、AMI、MAGICDATA 重口音大约一个 epoch 封顶，多出来的小时只加到仍有余量的源。

## 环境

Python 使用 Conda 环境 `qwen3_asr_ctc_smoke`。

```bash
conda run -n qwen3_asr_ctc_smoke python tools/qwen3_asr_pipeline.py \
  --config configs/robust_mix_1000h.yaml \
  --stage train
```

这份配置没有 JSONL prepare，不要跑 `--stage all`。
