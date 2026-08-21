# Qwen3-ASR POC 微调实验报告

更新日期：2026-08-21

## 1. 范围与结论

本文总结当前仓库中与 TCGA/病理英文目标域 POC 直接相关的 Qwen3-ASR 实验，覆盖：

- SFT 数据配比、标签括号清洗、合成数据筛选和 v2 合成集实验；
- GRPO rollout 数据挖掘、奖励函数和真实域-only 后训练实验；
- 每个实验对应的配置、训练输出、评测结果和保留的最佳 checkpoint。

不包含 Uyghur、GigaSpeech、通用英文/中文医疗评测等独立实验。评测主要使用 53 条 POC `dev/test` 音频；若 summary 显示 184 条，则是 GRPO 训练期间使用的另一份扩展评测清单，不能与 53 条结果直接横向比较。

当前最重要的结论：

1. 2026-08-21 的 真实域 + LLM 合成 + 粤语号码/度量（2 音色）SFT 在 53 条 POC 评测上得到 `9.76%` WER、`7.39%` CER（checkpoint-250），是当前统一归一化下的最佳结果；它同时是使用 test 作为 dev/选点集的结果，不能视为独立测试结果。
2. 注意归一化口径：此前报告记录的 `9.17%`（真实域 raw/denoised SFT）使用旧归一化；同一模型按当前归一化重测为 `15.63%`–`17.74%`（checkpoint-75/125），两套数字不可直接横比。2026-08-20 轮最佳 checkpoint-275 从当时的 `10.24%` 按当前归一化重测为 `10.12%`。
3. 初始 SFT 达到约 `13.40%` WER（旧归一化），是早期的目标域基线。
4. 真实域重复采样可以改善训练损失，但重复同一批 896 条真实录音不能增加声学多样性。
5. 清洗 catastrophic synthetic 数据后，SFT 的 53 条评测 WER 为 `14.76%`，优于 v2 合成集 4 倍/`2e-5` 的 `17.62%`。
6. 此前 v2 合成集 5 倍真实域、`1e-5` 学习率的 SFT 得到最佳 dev loss `0.59243`，评测 WER `15.71%`。
7. 第一轮 GRPO（全量 mined 数据、`8` generations、`5e-7`）在 `checkpoint-30000` 达到 `13.16%` WER，基本保持 SFT 能力；后续高学习率或纯真实域 GRPO 没有稳定改善。
8. 真实域-only、`-CER` 的 GRPO 在选定 checkpoint 上为 `15.55%` WER、`10.46%` CER，说明 CER 目标可以改善字符级相似度，但没有改善 WER。

## 2. 统一评测约定

评测脚本为 [`evaluation/english_medical/eval_english_medical_asr_jsonl.py`](../evaluation/english_medical/eval_english_medical_asr_jsonl.py)。默认设置为：

- `language=None`、`batch_size=4`、`max_new_tokens=512`；
- NFKC、中文数字转阿拉伯数字、`×/乘 -> x`、连续单字母序列合并（如 `m m -> mm`、`V A B -> VAB`）、小写、去 ASCII 标点、合并空白；
- 报告中的 WER 是 corpus WER，除非特别说明，不是独立于 dev 的测试集结果。

自 2026-08-21 起，医疗归一化函数统一实现在轻量共享模块 [`evaluation/english_medical/text_normalization.py`](../evaluation/english_medical/text_normalization.py)（`normalize_english`），数据采样、训练期 CER/WER 选点和最终评测共用；评测脚本保留同名函数以兼容既有导入。上表"当前归一化"指该共享模块的实现，与此前若干轮实验使用的旧实现存在口径差异（如数字/单位合并规则），跨轮比较需按同一实现重测。

历史评测有两个需要注意的偏差：

- 53 条 `dev` 同时被用于早停和部分结果选择，不能视为严格独立测试；
- 标签中出现文字 `period`、`26 SS 12099` 与 `26SS12099`、`SN 1` 与 `SN1` 等格式差异，可能把语义正确的输出计为 WER。当前评测已统一数字、乘号、单位和逐字母缩写；`period`、编号粘连等结构化规则仍未合并。

## 3. 数据演化

| 数据清单 | 条数 | 来源构成 | 时长 | 用途 |
|---|---:|---|---:|---|
| [`train.jsonl`](../data/poc_train_real_tcga_1to3_5silence_none/train.jsonl) | 25,687 | TCGA 合成 19,928；真实域 4,138；静音 1,621 | 38.17h / 12.72h / 2.68h | 初始 SFT |
| [`train_real_only.jsonl`](../data/poc_train_real_tcga_1to3_5silence_none/train_real_only.jsonl) | 896 | 真实域 896 条独立录音 | 2.75h | real-only SFT/对照 |
| [`train_filtered_tcga_catastrophic_keep_best015.jsonl`](../data/poc_train_real_tcga_1to3_5silence_none/train_filtered_tcga_catastrophic_keep_best015.jsonl) | 22,947 | 合成 17,197；真实域 4,129；静音 1,621 | 33.25h / 12.70h / 2.68h | 清洗 catastrophic 后 SFT |
| [`train_filtered_tcga_catastrophic_keep_best015_real_full_synth20h.jsonl`](../data/poc_train_real_tcga_1to3_5silence_none/train_filtered_tcga_catastrophic_keep_best015_real_full_synth20h.jsonl) | 11,370 | 合成 9,899；真实域 896；静音 575 | 19.05h / 2.75h / 0.95h | 不重复真实域、合成约 20h |
| [`v2 real4x train.jsonl`](../data/poc_train_tcga_v2_42h_real4x/train.jsonl) | 20,396 | v2 合成 16,812；真实域 3,584（4x） | 41.98h / 11.01h | v2 合成集 SFT |
| [`v2 real5x train.jsonl`](../data/poc_train_tcga_v2_42h_real5x/train.jsonl) | 21,292 | v2 合成 16,812；真实域 4,480（5x） | 41.98h / 13.77h | v2 合成集 SFT |
| [`train.jsonl`](../data/poc_train_real_llm_syn_full_testdev/train.jsonl) | 4,525 | 真实域 2,620（raw/denoised 各 1,310）；LLM 文本 TTS 合成 1,905 | 5.297h / 3.64h | LLM 合成 + CER 选点 SFT（2026-08-20） |
| [`train.jsonl`](../data/poc_train_real_llm_syn_cantonese_num_measure_2voice/train.jsonl) | 8,017 | 真实域 2,620；LLM 合成 1,905；粤语号码 1,686；粤语度量 1,806 | 5.296h / 3.628h / 0.718h / 1.003h | 最新 SFT（2026-08-21） |

真实域 5 倍采样后的 4,480 条并不是 4,480 条独立录音，而是 896 条录音的重复权重。它能改变优化目标的权重，不能补充 `SN1/SN2/non-SN`、`scrape cytology` 等短语的真实声学覆盖。

## 4. SFT 实验

| 实验 | 配置 | 训练数据 | 最佳 dev loss | 评测结果 | 输出 |
|---|---|---|---:|---:|---|
| 初始 SFT | [`poc_train_real_tcga_1to3_5silence_none_sft_lr2e5_3ep.yaml`](../configs/poc_train_real_tcga_1to3_5silence_none_sft_lr2e5_3ep.yaml) | 25,687 条；`lr=2e-5`；增强；3 epoch | 记录的最佳 dev 为 `0.53572`（checkpoint-300；模型目录已不在输出中） | checkpoint-500：`13.40%` WER | [`output`](../outputs/poc_train_real_tcga_1to3_5silence_none_sft_lr2e5_3ep)；[`baseline summary`](../outputs/poc_train_real_tcga_1to3_5silence_none_grpo_g8_lr5e7_1ep/eval/checkpoint-0_poc_test_summary.txt) |
| 标签括号清洗 SFT | [`poc_train_real_tcga_1to3_5silence_none_sft_lr2e5_bracketnorm_earlystop.yaml`](../configs/poc_train_real_tcga_1to3_5silence_none_sft_lr2e5_bracketnorm_earlystop.yaml) | 同初始数据；`strip_target_brackets=1`；早停 | 记录的最佳 dev 为 `0.52442`（checkpoint-500；模型目录已不在输出中） | checkpoint-200：`14.95%`；checkpoint-500：`16.15%` | [`output`](../outputs/poc_train_real_tcga_1to3_5silence_none_sft_lr2e5_bracketnorm_earlystop) |
| 真实域-only SFT | [`poc_train_real_only_sft_lr2e5_earlystop.yaml`](../configs/poc_train_real_only_sft_lr2e5_earlystop.yaml) | 896 条真实录音；无增强；`lr=2e-5` | `1.08986`，checkpoint-25 | `19.05%` WER | [`output`](../outputs/poc_train_real_only_sft_lr2e5_earlystop) |
| 清洗 catastrophic 后 SFT | 配置文件未保留；训练参数可由 output 中的 `trainer_args.bin` 和 checkpoint-400 还原 | 22,947 条；删除合成坏样本；`lr=2e-5` | `0.56703`，checkpoint-400 | `14.76%` WER；15/53 exact | [`output`](../outputs/poc_train_real_tcga_1to3_5silence_none_sft_filtered_tcga_catastrophic_keep_best015_lr2e5_earlystop) |
| 真实域不重复、合成约 20h | [`poc_train_real_tcga_1to3_5silence_none_sft_real_full_synth20h_lr2e5_earlystop.yaml`](../configs/poc_train_real_tcga_1to3_5silence_none_sft_real_full_synth20h_lr2e5_earlystop.yaml) | 11,370 条；真实域 896 条原始数量；合成约 20h | `0.61965`，checkpoint-300 | `31.19%` WER（53 条 dev） | [`output`](../outputs/poc_train_real_tcga_1to3_5silence_none_sft_real_full_synth20h_lr2e5_earlystop) |
| v2 合成、真实域 4x | [`poc_train_tcga_v2_42h_real4x_sft_lr2e5_earlystop.yaml`](../configs/poc_train_tcga_v2_42h_real4x_sft_lr2e5_earlystop.yaml) | v2 合成 41.98h；真实域 4x；`lr=2e-5` | `0.63298`，checkpoint-200 | `17.62%` WER；12/53 exact | [`output`](../outputs/poc_train_tcga_v2_42h_real4x_sft_lr2e5_earlystop) |
| v2 合成、真实域 5x | [`poc_train_tcga_v2_42h_real5x_sft_lr1e5_earlystop.yaml`](../configs/poc_train_tcga_v2_42h_real5x_sft_lr1e5_earlystop.yaml) | v2 合成 41.98h；真实域 5x；`lr=1e-5` | `0.59243`，checkpoint-500 | `15.71%` WER；9/53 exact | [`output`](../outputs/poc_train_tcga_v2_42h_real5x_sft_lr1e5_earlystop)；[`summary`](../outputs/poc_train_tcga_v2_42h_real5x_sft_lr1e5_earlystop/eval/checkpoint-500_poc_test_summary.txt) |
| 真实域 raw + denoised SFT | [`poc_train_real_raw_denoised_sft_lr2e5_3ep.yaml`](../configs/poc_train_real_raw_denoised_sft_lr2e5_3ep.yaml) | raw 1363 条 + denoised 1363 条；各使用一次；标签为 `language None<asr_text>` | `0.612891`，checkpoint-125 | `9.17%` WER；16/53 exact | [`output`](../outputs/poc_train_real_raw_denoised_sft_lr2e5_3ep)；[`predictions`](../outputs/poc_train_real_raw_denoised_sft_lr2e5_3ep/eval_checkpoint-125/predictions.jsonl) |
| 真实域 + LLM 合成、CER 选点 SFT | [`poc_train_real_llm_syn_full_testdev_cer_sft_lr2e5_3ep.yaml`](../configs/poc_train_real_llm_syn_full_testdev_cer_sft_lr2e5_3ep.yaml) | 真实域 2,620 + LLM 合成 1,905；`use_test_as_dev=1`；`save_best_metric=cer` 保留最低 5 个 | CER 最佳 `0.0791`，checkpoint-275 | 当时归一化 `10.24%` WER；按当前归一化重测 `10.12%` WER、25/53 exact | [`output`](../outputs/poc_train_real_llm_syn_full_testdev_cer_sft_lr2e5_3ep)；[`summary`](../outputs/poc_train_real_llm_syn_full_testdev_cer_sft_lr2e5_3ep/eval/checkpoint-275_poc_test_summary.txt) |
| 真实域 + LLM 合成 + 粤语号码/度量（2 音色）SFT | [`poc_train_real_llm_syn_cantonese_num_measure_2voice_full_testdev_cer_sft_lr2e5_3ep.yaml`](../configs/poc_train_real_llm_syn_cantonese_num_measure_2voice_full_testdev_cer_sft_lr2e5_3ep.yaml) | 上行数据 + 粤语号码 1,686 + 粤语度量 1,806（每归一化文本 ≤2 音色）；`save_best_metric=cer`；`wer_max_new_tokens=512` | CER 最佳 `0.0739`，checkpoint-250 | `9.76%` WER、`7.39%` CER；22/53 exact | [`output`](../outputs/poc_train_real_llm_syn_cantonese_num_measure_2voice_full_testdev_cer_sft_lr2e5_3ep)；[`predictions`](../outputs/poc_train_real_llm_syn_cantonese_num_measure_2voice_full_testdev_cer_sft_lr2e5_3ep/eval/checkpoint-250_poc_test_predictions.jsonl) |

### 最新真实域 raw/denoised SFT

数据划分：raw 和 denoised 各 1,363 条、各约 2.818 小时，严格配对后各使用一次；train 2,108 条（raw/denoised 各 1,054），dev 618 条，test 53 条；按 23 个病例组划分，`dev_fraction=0.2`、seed=42。

- 标签统一为 `language None<asr_text>...`；raw 样本 `noise_aug=0`，denoised 样本 `noise_aug=1`。
- 在线增强：`augment_prob=1.0`；速度增强概率 0.5，速度因子 0.8--1.2；denoised 才允许噪声增强，概率 0.5、SNR 5--20 dB；SpecAugment 概率 0.5。
- `nospeech_augment=1`，概率 0.3，前后静音 0.5--3.0 秒，并允许语音 + 静音 + 语音拼接。
- 训练参数：Qwen3-ASR-1.7B，`batch_size=2`、`grad_acc=16`（有效 batch 32），`lr=2e-5`，linear scheduler，`warmup_ratio=0.02`，3 epoch。
- 每 25 steps 保存/评估，以 dev `eval_loss` 早停（patience=5），只保留最佳 5 个 checkpoint。
- 最佳 dev loss 为 `0.612891`（checkpoint-125）；使用 `language=None`、512 tokens 在 53 条 POC 测试上评估，归一化后 WER 为 `9.17%`，exact accuracy 为 16/53。

### SFT 阶段判断

- `strip_target_brackets=1` 改善了 dev loss，但没有稳定改善 WER，说明括号规范化解决的是标签/格式噪声，不是主要声学瓶颈。
- 只训练 896 条真实录音导致明显过拟合，WER 为 `19.05%`。
- 清洗 catastrophic synthetic 后的 `14.76%` 是较好的旧 SFT 结果。
- v2 合成集规模增加后，真实域 5x + 较低学习率优于 4x + `2e-5`，但仍未超过初始 SFT，说明合成数据的文本覆盖不等于目标域声学覆盖。

### 真实域 + LLM 合成 + 粤语号码/度量 SFT（2026-08-20/21）

两轮实验都在 Qwen3-ASR-1.7B 基座重训 3 epoch（batch 2 × grad_acc 16、`lr=2e-5`、linear、warmup 0.02、save/eval 25 steps），训练参数与增强（speed 0.8–1.2、noise SNR 5–20 dB、SpecAug、nospeech 0.3）保持一致。

- 2026-08-20 轮：真实域 raw/denoised 各 1,310 条 + LLM 文本 TTS 合成 1,905 条（`raw/POC_train/llm_text_syn/sentences.jsonl` 与 TTS metadata CSV 按文本连接；该 jsonl 已重排，位置对齐假设不再成立）。
- 2026-08-21 轮：在上轮基础上加入两类粤语短数据源，并做音色下采样：
  - `cantonese_specimen_number`：4,230 条输入、5 个音色，按归一化文本分组（843 组），每组保留累计样本数最少的至多 2 个音色（平局按 source/index/audio_path 确定性打破），保留 1,686 条、0.718h；
  - `cantonese_measure`：4,108 条输入，908 组，同规则保留 1,806 条、1.003h；
  - 两类标签统一为 `language None<asr_text>` + `original` 字段（英文医疗文本），`aug=1`、`noise_aug=1`，独立 `sampling_source` 与 `segment_id`；
  - 新数据占训练集样本数 43.6%、时长 16.2%；不过滤与测试集相同的号码/度量值（它们是合成短语音原语，不含测试音频或完整测试句）。

两轮均使用 53 条 POC test 同时作为 dev 和 checkpoint 选择集（`use_test_as_dev=1`、`save_best_metric=cer`，仅保留最低 CER 的 5 个 checkpoint），因此下表的 dev/选择指标与 test 指标相同，**不能视为独立测试结果**。

2026-08-21 轮按当前归一化在 53 条 POC test 上的结果（batch 2、512 tokens；基线为旧轮最佳 checkpoint-275 的同配置重测）：

| 模型 | WER | CER | S/D/I | exact | 标本号子集(4条) WER/CER | 度量子集(24条) WER/CER |
|---|---:|---:|---|---:|---|---|
| 旧轮 checkpoint-275（基线） | 10.12% | 7.65% | 57/10/19 | 25/53 | 10.7% / 3.0% | 7.2% / 7.0% |
| checkpoint-200 | 11.29% | 8.01% | 58/15/23 | 20/53 | 3.6% / 0.7% | 8.6% / 7.4% |
| checkpoint-225 | 11.65% | 7.96% | 59/22/18 | 22/53 | 10.7% / 2.2% | 10.2% / 7.7% |
| **checkpoint-250（本轮最佳）** | **9.76%** | **7.39%** | 51/14/18 | 22/53 | 10.7% / 2.2% | 7.9% / 7.9% |
| checkpoint-575 | 10.00% | 7.91% | 50/12/23 | 24/53 | 10.7% / 2.2% | 7.0% / 7.8% |
| checkpoint-600 | 9.76% | 7.75% | 49/13/21 | 23/53 | 10.7% / 2.2% | 7.4% / 7.9% |

子集定义（按归一化参考文本匹配）：标本号 `\d{1,4} ss \d{3,6}`（4 条）；度量 `\d+(\.\d+)? (c m|m m|gram)`（24 条）。子集样本很少，指标波动大，只作方向性参考。

实现与产物：

- 数据准备扩展了 [`tools/prepare_real_raw_denoised_sft.py`](../tools/prepare_real_raw_denoised_sft.py)（`additional_synthetic_csvs` 多 CSV 接入 + 双音色确定性下采样 + 报告）；pipeline 通过 [`tools/qwen3_asr_pipeline.py`](../tools/qwen3_asr_pipeline.py) 展开 `dataset.additional_synthetic_csvs` / `dataset.additional_synthetic_max_per_text`。
- 医疗归一化函数提取为轻量共享模块 [`evaluation/english_medical/text_normalization.py`](../evaluation/english_medical/text_normalization.py)，数据采样、训练期 CER 和最终评测共用同一实现；`eval_english_medical_asr_jsonl.py` 保持原导入接口。
- 训练脚本新增 `--wer_max_new_tokens`（显式传给 ASR wrapper，本实验 512）；每次训练期 CER 评测把原始及归一化预测落盘到 `generation_eval/step-*_predictions.jsonl`，用于追踪循环输出。
- 本轮训练 step-450 的 dev CER 一度飙升到 `0.5864`，`generation_eval` 定位到单条样本产生 2,518 字符的重复循环（同一 20-gram 重复 302 次），该 checkpoint 未进入保留集；其余评测步无循环。
- 按 WER 选出本轮最佳 checkpoint-250（与 checkpoint-600 同为 9.76%，CER 更低），其余 4 个 checkpoint 暂不删除。

#### Radiology 长音频 rollout 循环测试（2026-08-21）

对 checkpoint-250 在域外放射科口述长音频（`/root/autodl-tmp/workspace/dataset/Test_Samples/samples/radiology/wav`，10 条、21 分钟、41–323s，含粤英混合）做 rollout 循环测试（[`tools/rollout_loop_test.py`](../tools/rollout_loop_test.py)），`max_new_tokens=512`。循环判定：词 5-gram 重复 ≥3 次、字符 20-gram 非重叠重复 ≥3 次，或顶到 token 上限。

**重要口径说明**：第一轮 90 次 rollout（每条 1 贪心 + 8 "采样"）实际全部是贪心解码——`Qwen3ASRForConditionalGeneration` 覆写的 `generate`（`modeling_qwen3_asr.py:1325`）不读外层 `generation_config`，`temperature/top_p/penalty` 均未生效（已验证 8 条 "采样" 输出逐字节相同）。因此第一轮数字应读作**贪心解码**结果；采样循环率在修复后另行实测（见下）。

- **贪心解码循环率 60%（54/90，即 10 条音频 × 10 次贪心）**。
- 循环在音频层面是确定性的：6 条音频全部 rollout 都循环（10/10），4 条 0 次循环，不存在部分循环的音频。
- 所有循环输出都顶到 512 token 上限（解码后 508–509 tokens，未发 EOS），重复段占输出中位 95%，典型短语如 `fresh. The left axillary sentinel`、`side is rendered as a` 重复数十次。
- 非循环的 4 条音频输出长度稳定（33/228/255/500 tokens），但该域外集合整体质量也差：出现大段粤语输出（如 `佢就一層嚟嘅`）和过短输出（1435R 仅 33 tokens）。
- 对比：POC 病理域内 53 条 test 的 30 余次训练期评测 + 终测中仅 step-450 出现过 1 次循环（约 0.1% 量级）。循环风险高度集中在域外/语言混合长音频，域内短语音基本不触发。

**真正采样 + repetition penalty 复测（仅 1854O，贪心 + 8 采样/组）**：

修复配置注入位置（设到 `thinker.generation_config`）后复测。另注意 transformers 4.57 已移除生成循环对 `frequency_penalty`/`presence_penalty` 的原生支持（`GenerationConfig` 字段仍在但静默忽略），脚本改为注入标准语义的自定义 `LogitsProcessor` 实现。

| 配置 | 周期循环 | 输出 token 范围 | 备注 |
|---|---:|---|---|
| 无 penalty（真采样） | 5/9 (56%) | 4–509 | 比贪心 9/9 明显下降但仍高；出现 1 次 4-token 过早终止 |
| presence_penalty=1.5 | 0/9 (0%) | 309–439 | 循环全消；尾部仍有粤语退化片段 |
| frequency_penalty=1.5 | 0/9 (0%) | 208–365 | 循环全消；贪心输出为最连贯的完整转写（307 tokens） |
| frequency_penalty=5.0（仅贪心） | 0/1 | 509 | 无周期重复但仍 rambling 到 token 上限（非周期退化） |

结论：penalty 对**周期性重复循环**有效（fp/pp=1.5 均归零），fp=1.5 下贪心转写质量最好；但它不能恢复域外长音频上的正常终止与连贯性（fp=5.0 贪心仍顶格 rambling，pp=1.5 尾部退化）。penalty 是推理期缓解手段，不改变模型在该域上的整体能力。

产物：第一轮 [`rollout_radiology/`](../outputs/poc_train_real_llm_syn_cantonese_num_measure_2voice_full_testdev_cer_sft_lr2e5_3ep/rollout_radiology/rollout_loop_test.jsonl)（实际为贪心 ×10/音频）；复测 [`rollout_radiology_v3/`](../outputs/poc_train_real_llm_syn_cantonese_num_measure_2voice_full_testdev_cer_sft_lr2e5_3ep/rollout_radiology_v3/rollout_loop_test.jsonl)。

## 5. GRPO 数据挖掘

挖掘实现为 [`tools/mine_qwen3_asr_grpo_data.py`](../tools/mine_qwen3_asr_grpo_data.py)，产物在 [`grpo_g16_mining`](../outputs/poc_train_real_tcga_1to3_5silence_none_grpo_g16_mining)。每个候选音频使用同一 prompt/audio evidence 生成 16 条 rollout，使用 ground-truth text 计算 WER，并根据 mean/best/std/worst 分桶。

完整挖掘报告 [`mining_report.jsonl`](../outputs/poc_train_real_tcga_1to3_5silence_none_grpo_g16_mining/mining_report.jsonl) 共 25,667 条：

| 类别 | 数量 | 解释 |
|---|---:|---|
| Easy | 11,293 | mean WER 低、方差低 |
| Mixed | 8,562 | 混合/中等难度，不能简单归入单一稳定性类别 |
| Catastrophic hallucination | 3,758 | 至少部分 rollout 完全偏离 |
| Unstable | 1,699 | 方差高、结果不稳定 |
| Recoverable | 285 | mean WER 高但 best WER 低，最适合 GRPO |
| Consistently wrong | 70 | best WER 高且方差低，GRPO 信号弱 |

按音频路径区分，真实域有 4,118 条，合成域有 21,549 条。最终用于全量 GRPO 的去重清单为 [`train_recoverable_unstable_catastrophic_mixed_dedup.jsonl`](../outputs/poc_train_real_tcga_1to3_5silence_none_grpo_g16_mining/train_recoverable_unstable_catastrophic_mixed_dedup.jsonl)，训练日志记录为 12,196 条。

挖掘结果验证了：Easy 占多数是正常现象；Mixed 是边界样本和轻度不稳定样本的集合；合成音频的 catastrophic 类别显著多于真实域，不能未经筛选直接作为 RL 信号。

## 6. GRPO 实验

GRPO 实现为 [`finetuning/qwen3_asr_grpo.py`](../finetuning/qwen3_asr_grpo.py)。实现中 audio tower/aligner 冻结，仅更新 LLM 和 `lm_head`；每个 rollout group 共享同一音频编码，目标是优化同一音频证据下的文本生成。

| 实验 | 训练元数据/配置 | 数据与奖励 | 最佳/最终结果 | 输出 |
|---|---|---|---|---|
| 全量 mined GRPO | 独立 config 未保留；[`train.log`](../outputs/poc_train_real_tcga_1to3_5silence_none_grpo_g8_lr5e7_1ep/train.log)；实现默认参数可见脚本 | 40,871 条；G=8；`lr=5e-7`；初始 SFT checkpoint-500；负 WER 奖励；冻结 audio tower/aligner | checkpoint-30000：`13.16%`；checkpoint-40871：`13.52%` | [`output`](../outputs/poc_train_real_tcga_1to3_5silence_none_grpo_g8_lr5e7_1ep)；[`comparison`](../outputs/poc_train_real_tcga_1to3_5silence_none_grpo_g8_lr5e7_1ep/eval/checkpoint_comparison.json) |
| mined 子集 GRPO | [`run_config.json`](../outputs/poc_train_real_tcga_1to3_5silence_none_grpo_g16_cer025_lr2e5_beta004_1000step/run_config.json) | 12,196 条；G=16；`lr=2e-5`；`beta=0.04`；奖励 `-(WER+0.25*CER)` | dev 选 checkpoint-600：25.80%；53 条 test：17.46% | [`output`](../outputs/poc_train_real_tcga_1to3_5silence_none_grpo_g16_cer025_lr2e5_beta004_1000step)；[`selection`](../outputs/poc_train_real_tcga_1to3_5silence_none_grpo_g16_cer025_lr2e5_beta004_1000step/eval/selected_checkpoint.json) |
| 真实域-only GRPO，WER+CER | [`run_config.json`](../outputs/poc_train_real_tcga_1to3_5silence_none_grpo_g16_realonly_lr2e6_beta004_1ep/run_config.json) | 525 条重点真实域样本；G=16；`lr=2e-6`；`beta=0.04`；`-(WER+0.25*CER)` | dev 选 checkpoint-400；test `15.19%`，与 SFT checkpoint-200 的 `14.95%` 相比没有提升 | [`output`](../outputs/poc_train_real_tcga_1to3_5silence_none_grpo_g16_realonly_lr2e6_beta004_1ep)；[`selection`](../outputs/poc_train_real_tcga_1to3_5silence_none_grpo_g16_realonly_lr2e6_beta004_1ep/eval/selected_checkpoint.json) |
| 真实域-only GRPO，CER-only | [`run_config.json`](../outputs/poc_train_real_tcga_1to3_5silence_none_grpo_g16_realonly_ceronly_lr2e6_beta004_1ep/run_config.json) | 525 条；G=16；`lr=2e-6`；`beta=0.04`；奖励 `-CER` | checkpoint-500；test `15.55%` WER、`10.46%` CER | [`output`](../outputs/poc_train_real_tcga_1to3_5silence_none_grpo_g16_realonly_ceronly_lr2e6_beta004_1ep)；[`summary`](../outputs/poc_train_real_tcga_1to3_5silence_none_grpo_g16_realonly_ceronly_lr2e6_beta004_1ep/eval/selected_checkpoint_test_summary.txt) |
| 早期 real-only GRPO | 独立 config 未保留；[`eval`](../outputs/poc_train_real_only_grpo_g8_cer_lr2e6_beta004_1000step/eval) | G=8；`lr=2e-6`；`beta=0.04`；CER 目标；真实域 | checkpoint-1000：`21.31%` WER | [`output`](../outputs/poc_train_real_only_grpo_g8_cer_lr2e6_beta004_1000step) |

### GRPO 阶段判断

- 第一轮全量 GRPO 的学习率很小，WER 基本保持在 SFT 附近，说明冻结 audio tower 后没有立即破坏模型能力，但优化信号很弱。
- `lr=2e-5`、`beta=0.04` 的 mined 子集实验出现明显退化，说明当前奖励、rollout 分布或训练步数不足以支持这么大的更新幅度。
- 真实域-only 的 `-CER` 比 `-(WER+0.25*CER)` 更符合字符级纠错目标，但没有提升 corpus WER；CER 不能解决 `all -> not`、block 字母和整词替换等结构性错误。

## 7. 主要错误来源

此前 v2 real5x SFT 的逐条预测在 [`checkpoint-500_poc_test_predictions.jsonl`](../outputs/poc_train_tcga_v2_42h_real5x_sft_lr1e5_earlystop/eval/checkpoint-500_poc_test_predictions.jsonl)。主要错误可以归纳为：

1. **目标域缩略语和固定短语缺少真实声学覆盖**：`SN1/SN2/non-SN`、`scrape cytology`、`coronal full slabs` 在真实域训练集中几乎没有对应样本；重复 896 条录音不能补足这一点。
2. **语言模型模板补全**：`coronal full slabs` 被续写为 `No frozen blocks left`，`non-SN` 被续写为 `nonetheless` 或 `nonaneurysmal aneurysm`。
3. **高风险结构词错误**：`all -> not/non`、`block K -> block A`、`frozen -> full-thickness`。这些错误在 WER 中只算普通替换，但对病理报告语义影响很大。
4. **编号和格式评分问题**：`26 SS 12099`、`SN 1`、文字 `period` 等导致额外 WER；正式评测需要另提供结构归一化指标。
5. **音频标签边界/代码切换**：少数音频在标注文本结束后仍有粤语或中英混合语音，模型输出会被 WER 计为插入。

## 8. 清理与保留策略

本次清理只针对报告中、使用同一套 53 条 POC `dev/test` 评测的实验输出。排名不纳入 `combined_three_source_sft_lr2e5` 的 474 条训练切分结果（该目录记录过 `5.71%` WER，但不可与 53 条 POC 结果直接比较）。

- 按归一化后的 corpus WER 排名前三，只保留以下三个实验的完整输出目录；
- 保留全量 mined GRPO 的挖掘报告，作为第二名实验的配套数据产物；
- 删除其余 POC SFT/GRPO 模型、checkpoint 和评测产物；
- 保留所有 `configs/`、`data/`、训练日志、本报告和非 POC 输出；
- 不删除 Uyghur、GigaSpeech、通用评测或模型缓存输出。

保留的三次实验产物为：

- 真实域 raw + denoised SFT：`9.17%` WER，checkpoint-125，目录 `poc_train_real_raw_denoised_sft_lr2e5_3ep`；
- 全量 mined GRPO：`13.16%` WER，checkpoint-30000，目录 `poc_train_real_tcga_1to3_5silence_none_grpo_g8_lr5e7_1ep`；
- 初始 SFT：`13.40%` WER，checkpoint-500，目录 `poc_train_real_tcga_1to3_5silence_none_sft_lr2e5_3ep`。

全量 mined GRPO 的辅助挖掘目录 `poc_train_real_tcga_1to3_5silence_none_grpo_g16_mining` 一并保留；它不是独立 WER 实验。

## 9. 后续建议

1. 在已有数字、单位和缩写归一化基础上，继续评估 `period`、编号分词、`SN1/SN 1` 和英式拼写；同时保留结构化错误指标，不要把 `all/not` 或 block 字母归一化掉。
2. 从真实目标域补充带有 `SN/non-SN`、编号、block、frozen/paraffin、hemithyroidectomy 和 lumpectomy 的短句，优先增加真实声学实例而不是继续重复采样。
3. GRPO 继续使用冻结 audio tower/aligner，但降低更新幅度，先在 525 条真实 hard samples 上做小步数对照，并同时报告 WER、CER、结构词准确率和 catastrophic rate。
4. 对 GRPO rollout 使用有效样本过滤，避免 Easy 和合成 catastrophic 样本稀释或污染组内优势信号。
