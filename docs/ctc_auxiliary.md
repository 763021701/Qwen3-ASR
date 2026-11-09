# Qwen3-ASR 辅助 CTC head（AuT 声学约束）

> 2026-11-09：训练期 CTC 路线终止。实现、测试、诊断脚本与配置已归档到
> `archive/ctc_auxiliary/` 和 `archive/configs/`；本文保留作为负结果记录，
> 下文中的代码/配置路径为历史路径。

## 目的

极端域偏移下，基座模型本身 WER 很高；只用目标域/合成目标域做 CE 微调时，LLM
容易用语言先验补全不确定的发音，表现为模板化插入、循环和整段幻觉。本方案在
audio tower（AuT）输出上接一个 CTC head，用同一段音频的**实际转写**做帧级监督，
使 AuT 必须保留与文字对应的时序声学信息，而不是把不确定性全部交给 LLM。

CTC 是辅助目标：最终推理仍由 LLM 输出，CTC head 只改变训练时的梯度。

## 结构

```
audio -> AuT (conv + 24 layers + proj2) -> 2048-d frames
                                          ├─ CTC head -> CTC loss (+ coverage)
                                          └─ LLM (audio placeholder embeddings) -> CE loss
loss = CE + ctc_loss_weight * (CTC + coverage_weight * coverage_penalty)
```

- **Tokenizer**：复用 SenseVoiceSmall 的
  `chn_jpn_yue_eng_ko_spectok.bpe.model`（25,055 pieces，中/英/粤/日/韩），但
  **CTC 输出词表只保留训练转写中实际出现的 piece**（M1 上 2,712 个 token /
  2,713 类，blank=0），映射保存在 `ctc_vocab.json`。
- **Blank**：`0`，与 SenseVoice 的 CTC 约定一致；SentencePiece 的 `<unk>` 也是 0，
  因此编码时丢弃 id 0，避免把 unk 当作 blank 目标。
- **Head**：`LayerNorm(2048) -> Linear(2048,512) -> GELU -> LayerNorm(512) ->
  Linear(512,V)`，fp32（M1 上约 2.45M 参数），学习率单独设置。
- **Coverage penalty**：`relu(target_len - 非blank帧数)^2` 的加权项，防止 CTC
  head 塌缩成“几乎全 blank + 单个高频 token”的退化解。
- **位置**：`thinker.audio_tower` 的最终输出（即 LLM 实际看到的 2048 维帧），
  因此 CTC 与 CE 约束同一个表示；aligner 只是逐帧 MLP，无法替 encoder 修复时序。
- **帧率**：AuT 输出约 13 帧/秒（76.9 ms/帧）。实测训练集约 0.24 tokens/帧，
  最大 0.63；speed 1.2x 后仍 < 0.8，不会出现 target 长于帧数。

## CTC 目标文本

只用 `<asr_text>` 后的转写，并做声学归一化（`normalize_ctc_text`）：

- NFKC、小写；
- 连字符/破折号转空格（`x-ray` -> `x ray`）；
- 去掉标点，保留撇号和数字间的小数点；
- 折叠空白；CJK 字符和数字原样保留（不转阿拉伯数字）。

大小写和标点无法从波形观察到，保留它们只会让 CTC head 学语言先验，和本方案的
目的相反。该归一化只影响 CTC 目标，不影响 LLM 的 CE 标签。

## 已观察到的塌缩与对策

2026-10-29 首轮（全 25,055 词表 + 线性 head，M1，λ=1.0）在 checkpoint-300 时
CTC-only WER 99.7%，解码为 92 帧中 91 帧 blank、1 帧 `the`；600 步为 88.6%，
仍是高频词拼接。全词表 + 线性 head 会落到“空白 + 单 token”的退化解。

v2 改为训练集 2,712-token 子集 + 瓶颈 MLP，150 步 CTC-only WER 92.7%，仍以
高频词为主；v3 加 coverage penalty（`ctc_coverage_weight: 0.01`）后，300 步
（0.25 epoch）CTC-only WER **83.0% / CER 71.0%**，解码开始出现
“八七 八五五”“lymph / measuring up to 零點五六 cm / fracture / compared
previous”等内容 token，不再是单 token 塌缩。coverage 项把 head 从空白+单 token
的退化解里拉出来，CTC loss 再塑形 token 序列。

完整 1 epoch v3（λ=1、coverage 0.01 **未 detach**）的 LLM 结果并未受益：
holdout19 ckpt-600/900/1200 = 54.68/57.97/56.96%（baseline 54.18%），
radiology95 = 48.88/53.46/60.24%（baseline 38.15%），POC53 =
19.18/16.59/18.47%（baseline 15.88%）。ckpt-1200 出现
“Full medical history is available.”、“Please stop next slide.” 等循环；
holdout19 插入率仍 ~16%（baseline 15.8%），而 CTC-only 头插入率仅 3.9–5.2%。
说明 AuT 已有声学证据、CTC 头是保守解码器，**瓶颈在 LLM 的语言先验解码**；
训练期 CTC 单独不能降低 LLM 插入。下一步优先推理侧 CTC 约束/重排；若继续
训练期方案，先预训练 head，再以 coverage=0（或 detached）、小 λ 联合训练。

## 用法

配置（`configs/poc_radiology_mix_m1_ctc_1ep.yaml`）：

```yaml
training:
  ctc_loss_weight: 1.0
  ctc_coverage_weight: 0.01
  lr_ctc: 1e-3
  ctc_tokenizer: /path/to/chn_jpn_yue_eng_ko_spectok.bpe.model
```

命令行等价参数：

```bash
--ctc_loss_weight 1.0 \
--ctc_coverage_weight 0.01 \
--lr_ctc 1e-3 \
--ctc_tokenizer /path/to/chn_jpn_yue_eng_ko_spectok.bpe.model
```

仅在 full fine-tuning（`use_lora: 0`）中实现；pipeline 会对 LoRA + CTC 直接报错。
`ctc_loss_weight` 默认 0，完全保持原有 CE-only 配方。启用后训练日志会同时打印
`loss_ce` 和 `loss_ctc`；`eval_loss` 仍只含 CE，便于和旧实验横比。

## 诊断：CTC-only 解码

```bash
python tools/eval_ctc_encoder.py \
  --model outputs/<run>/checkpoint-600 \
  --ctc_tokenizer /path/to/chn_jpn_yue_eng_ko_spectok.bpe.model \
  --eval_jsonl data/poc_radiology_oracle/holdout.jsonl \
  --output_dir outputs/<run>/ctc_eval_holdout
```

该工具只跑 AuT + CTC head，不看 LLM，回答两个问题：

- CTC-only WER/CER 是否随训练下降：AuT 是否真的保留了声学信息；
- 若 CTC-only 明显改善而 LLM WER 不变：瓶颈在 LLM 不使用 encoder 信息，
  需要更大 CTC 权重或推理侧约束。

## 实现文件

- `finetuning/ctc_auxiliary.py`：tokenizer 包装、归一化、CTCHead、AuT forward hook、
  CTC loss 与 greedy decode。
- `finetuning/qwen3_asr_sft.py`：collator 生成 `ctc_targets`；`CTCAuxiliaryTrainer`
  在 CE 之外加 CTC；`--ctc_*` 参数与 `ctc_config.json`。
- `tools/qwen3_asr_pipeline.py`：配置键到 CLI 的映射。
- `tools/eval_ctc_encoder.py`：CTC-only 诊断解码。
- `tests/test_ctc_auxiliary.py`：归一化、tokenizer、loss、greedy decode 单测。

## 注意

- 每个 checkpoint 会多出 `thinker.ctc_head.*`（fp32，M1 上约 10 MB）和对应的
  optimizer state，以及 `ctc_vocab.json` / `ctc_config.json` 两个 sidecar；
  WER 推理会提示这些 key unexpected，属正常现象。
- 若 `--freeze_modules encoder`，CTC 只会训练 head，起不到约束 encoder 的作用。
- GPU 被其他任务占用时，CTC head + CTCLoss 会增加显存和单步时间，建议独占 GPU
  做正式对照。