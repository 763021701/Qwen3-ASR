# Qwen3-ASR POC 微调实验报告

更新日期：2026-10-29

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
9. 2026-09-21：从 plus0821 SFT `checkpoint-3900` 出发、冻结 audio tower、只在真实域上做 prompt 热词小规模 continue-SFT，**没有恢复 LLM 热词跟随**，空 context POC WER 还略差。权重已删，只留日志和评测。
10. 2026-10-23 radiology oracle **v2（修正标签重跑，有效结论）**：8 条录音 / 17 min 真实放射科口述、从基座 380 updates，holdout19 `55.19%`（ep3）/ `57.22%`（ep5）聚合 WER，优于基座（`76.46%`）约 19–21pp、优于 champion ckpt-3900（`64.30%`）约 8pp；灾难片段 0–3/19。旧版 oracle（2026-10-21，作废）的"不如冠军"是旧标签造成的。POC53 为 `35–38%`（基座本身 41.41%、病理冠军 7.65%）：从基座出发的 radiology-only 模型**相对基座并未遗忘、反而更好**，但离病理部署水平仍差约 28pp（见 §4「oracle v2」与实验 C 方案）。
11. 2026-10-23 实验 B + synth-only oracle（双负结果）：8,002 条 CosyVoice radiology 合成（16.7h）**对真实录音无迁移**——混入主线 continue-SFT 与冠军持平（holdout19 67.95% vs 67.73%）、POC 退化；从基座训满 3 epochs 更崩到 `161.59%`（基座 78.18%），"更新不足"被否定，瓶颈确认是真实声学而非文本。另发现冠军 mix 的 youtube_med_en 音频 98.2% 已丢失，原配方不可复现（见 §4）。
12. 2026-10-27 录音环境增强（mega-asr-sim）：仅用 tts_radiology 训练时，7 场景 ×3 退化种子的环境增强把 radiology95 从 69.70% 降到 **42.47%**（greedy）/ **41.19%（no_repeat_ngram=4）**；再叠加**双模型取较短组合**（E4-564 + E13-752，均 nng4，无监督）达到 **39.36%，<40% 目标达成**（holdout19 54.68% vs 单模型 59.75%）。有效成分是目标域噪声多样性 + 场景混合，真实 RIR、LUFS 对齐、干净混合、severity 过滤、freesound 噪声、温和噪声均无效；POC53 为 31.3%（基座 41.41%、病理冠军 7.65%，相对基座是改善而非遗忘，但未达病理部署水平）（见 §4）。
13. 2026-10-28 真实目标域混训：7–13% 真实病理最佳（M2 38.11%），23% 开始稀释（39.79）、3.3% 不足（41.52）；**soup(M1-600, M2b-780) = radiology95 37.12% / holdout19 48.61% / POC53 13.53%**。同日 m1_yt（M1 + 全量 YouTube 36,428 行/164.8h）单模型 radiology95 **36.39%**（ckpt-2100）/ holdout19 52.91% / POC53 14.71%，但 all95 受第 14 条的 train76 文本泄漏影响。
14. 2026-10-29 语言先验清洗（**负结果**）+ 发现 sim 文本泄漏：ep1 的 177/4,001 条 TTS 文本与 radiology95 参考共享 ≥8-gram（train76 23/76 片段、平均覆盖 14.5% 参考词、最高 60%；holdout19 2/19、5.2%），因此**此前所有 sim 系实验的 radiology95 绝对值都含 train76 文本泄漏，holdout19 才是公平集**。清洗版（ep1 去污 3,800 + RADX_slref 16,030 + 粤语三音色 945 + 真实 2,714 + 严格 YouTube 3,247；1 个环境种子；26,421 行 / 63.7h）best ckpt-600 为 radiology95 **41.44%** / holdout19 **56.96%** / POC53 **11.88%**（radiology 混训分支新低；病理冠军仍为 7.65%）。radiology 侧显著退化，且干净子集同样更差（tr76_clean 38.25 vs M1 33.51；ho19_clean 62.54 vs m1_yt 57.86），**不是去污造成**；best ckpt 在 600/826，也**不是欠拟合**。主因是把 sim 从 3 seed（每文本 9 次声学实现）降到 1 seed（1 次）：E4 已验证 1→3 seed 值 4.65pp，本次相对 m1_yt 退化 5.05pp，量级吻合。语言先验本身未被公平检验（见 §4）。

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
| [`train.jsonl`](../archive/data/poc_train_real_tcga_1to3_5silence_none/train.jsonl) | 25,687 | TCGA 合成 19,928；真实域 4,138；静音 1,621 | 38.17h / 12.72h / 2.68h | 初始 SFT |
| [`train_real_only.jsonl`](../archive/data/poc_train_real_tcga_1to3_5silence_none/train_real_only.jsonl) | 896 | 真实域 896 条独立录音 | 2.75h | real-only SFT/对照 |
| [`train_filtered_tcga_catastrophic_keep_best015.jsonl`](../archive/data/poc_train_real_tcga_1to3_5silence_none/train_filtered_tcga_catastrophic_keep_best015.jsonl) | 22,947 | 合成 17,197；真实域 4,129；静音 1,621 | 33.25h / 12.70h / 2.68h | 清洗 catastrophic 后 SFT |
| [`train_filtered_tcga_catastrophic_keep_best015_real_full_synth20h.jsonl`](../archive/data/poc_train_real_tcga_1to3_5silence_none/train_filtered_tcga_catastrophic_keep_best015_real_full_synth20h.jsonl) | 11,370 | 合成 9,899；真实域 896；静音 575 | 19.05h / 2.75h / 0.95h | 不重复真实域、合成约 20h |
| [`v2 real4x train.jsonl`](../archive/data/poc_train_tcga_v2_42h_real4x/train.jsonl) | 20,396 | v2 合成 16,812；真实域 3,584（4x） | 41.98h / 11.01h | v2 合成集 SFT |
| [`v2 real5x train.jsonl`](../archive/data/poc_train_tcga_v2_42h_real5x/train.jsonl) | 21,292 | v2 合成 16,812；真实域 4,480（5x） | 41.98h / 13.77h | v2 合成集 SFT |
| [`train.jsonl`](../archive/data/poc_train_real_llm_syn_full_testdev/train.jsonl) | 4,525 | 真实域 2,620（raw/denoised 各 1,310）；LLM 文本 TTS 合成 1,905 | 5.297h / 3.64h | LLM 合成 + CER 选点 SFT（2026-08-20） |
| [`train.jsonl`](../archive/data/poc_train_real_llm_syn_cantonese_num_measure_2voice/train.jsonl) | 8,017 | 真实域 2,620；LLM 合成 1,905；粤语号码 1,686；粤语度量 1,806 | 5.296h / 3.628h / 0.718h / 1.003h | 最新 SFT（2026-08-21） |

真实域 5 倍采样后的 4,480 条并不是 4,480 条独立录音，而是 896 条录音的重复权重。它能改变优化目标的权重，不能补充 `SN1/SN2/non-SN`、`scrape cytology` 等短语的真实声学覆盖。

## 4. SFT 实验

| 实验 | 配置 | 训练数据 | 最佳 dev loss | 评测结果 | 输出 |
|---|---|---|---:|---:|---|
| 初始 SFT | [`poc_train_real_tcga_1to3_5silence_none_sft_lr2e5_3ep.yaml`](../archive/configs/poc_train_real_tcga_1to3_5silence_none_sft_lr2e5_3ep.yaml) | 25,687 条；`lr=2e-5`；增强；3 epoch | 记录的最佳 dev 为 `0.53572`（checkpoint-300；模型目录已不在输出中） | checkpoint-500：`13.40%` WER | [`output`](../archive/outputs/poc_train_real_tcga_1to3_5silence_none_sft_lr2e5_3ep)；[`baseline summary`](../outputs/poc_train_real_tcga_1to3_5silence_none_grpo_g8_lr5e7_1ep/eval/checkpoint-0_poc_test_summary.txt) |
| 标签括号清洗 SFT | [`poc_train_real_tcga_1to3_5silence_none_sft_lr2e5_bracketnorm_earlystop.yaml`](../archive/configs/poc_train_real_tcga_1to3_5silence_none_sft_lr2e5_bracketnorm_earlystop.yaml) | 同初始数据；`strip_target_brackets=1`；早停 | 记录的最佳 dev 为 `0.52442`（checkpoint-500；模型目录已不在输出中） | checkpoint-200：`14.95%`；checkpoint-500：`16.15%` | [`output`](../outputs/poc_train_real_tcga_1to3_5silence_none_sft_lr2e5_bracketnorm_earlystop) |
| 真实域-only SFT | [`poc_train_real_only_sft_lr2e5_earlystop.yaml`](../archive/configs/poc_train_real_only_sft_lr2e5_earlystop.yaml) | 896 条真实录音；无增强；`lr=2e-5` | `1.08986`，checkpoint-25 | `19.05%` WER | [`output`](../outputs/poc_train_real_only_sft_lr2e5_earlystop) |
| 清洗 catastrophic 后 SFT | 配置文件未保留；训练参数可由 output 中的 `trainer_args.bin` 和 checkpoint-400 还原 | 22,947 条；删除合成坏样本；`lr=2e-5` | `0.56703`，checkpoint-400 | `14.76%` WER；15/53 exact | [`output`](../outputs/poc_train_real_tcga_1to3_5silence_none_sft_filtered_tcga_catastrophic_keep_best015_lr2e5_earlystop) |
| 真实域不重复、合成约 20h | [`poc_train_real_tcga_1to3_5silence_none_sft_real_full_synth20h_lr2e5_earlystop.yaml`](../archive/configs/poc_train_real_tcga_1to3_5silence_none_sft_real_full_synth20h_lr2e5_earlystop.yaml) | 11,370 条；真实域 896 条原始数量；合成约 20h | `0.61965`，checkpoint-300 | `31.19%` WER（53 条 dev） | [`output`](../outputs/poc_train_real_tcga_1to3_5silence_none_sft_real_full_synth20h_lr2e5_earlystop) |
| v2 合成、真实域 4x | [`poc_train_tcga_v2_42h_real4x_sft_lr2e5_earlystop.yaml`](../archive/configs/poc_train_tcga_v2_42h_real4x_sft_lr2e5_earlystop.yaml) | v2 合成 41.98h；真实域 4x；`lr=2e-5` | `0.63298`，checkpoint-200 | `17.62%` WER；12/53 exact | [`output`](../outputs/poc_train_tcga_v2_42h_real4x_sft_lr2e5_earlystop) |
| v2 合成、真实域 5x | [`poc_train_tcga_v2_42h_real5x_sft_lr1e5_earlystop.yaml`](../archive/configs/poc_train_tcga_v2_42h_real5x_sft_lr1e5_earlystop.yaml) | v2 合成 41.98h；真实域 5x；`lr=1e-5` | `0.59243`，checkpoint-500 | `15.71%` WER；9/53 exact | [`output`](../outputs/poc_train_tcga_v2_42h_real5x_sft_lr1e5_earlystop)；[`summary`](../outputs/poc_train_tcga_v2_42h_real5x_sft_lr1e5_earlystop/eval/checkpoint-500_poc_test_summary.txt) |
| 真实域 raw + denoised SFT | [`poc_train_real_raw_denoised_sft_lr2e5_3ep.yaml`](../archive/configs/poc_train_real_raw_denoised_sft_lr2e5_3ep.yaml) | raw 1363 条 + denoised 1363 条；各使用一次；标签为 `language None<asr_text>` | `0.612891`，checkpoint-125 | `9.17%` WER；16/53 exact | [`output`](../archive/outputs/poc_train_real_raw_denoised_sft_lr2e5_3ep)；[`predictions`](../archive/outputs/poc_train_real_raw_denoised_sft_lr2e5_3ep/eval_checkpoint-125/predictions.jsonl) |
| 真实域 + LLM 合成、CER 选点 SFT | [`poc_train_real_llm_syn_full_testdev_cer_sft_lr2e5_3ep.yaml`](../archive/configs/poc_train_real_llm_syn_full_testdev_cer_sft_lr2e5_3ep.yaml) | 真实域 2,620 + LLM 合成 1,905；`use_test_as_dev=1`；`save_best_metric=cer` 保留最低 5 个 | CER 最佳 `0.0791`，checkpoint-275 | 当时归一化 `10.24%` WER；按当前归一化重测 `10.12%` WER、25/53 exact | [`output`](../archive/outputs/poc_train_real_llm_syn_full_testdev_cer_sft_lr2e5_3ep)；[`summary`](../archive/outputs/poc_train_real_llm_syn_full_testdev_cer_sft_lr2e5_3ep/eval/checkpoint-275_poc_test_summary.txt) |
| 真实域 + LLM 合成 + 粤语号码/度量（2 音色）SFT | [`poc_train_real_llm_syn_cantonese_num_measure_2voice_full_testdev_cer_sft_lr2e5_3ep.yaml`](../archive/configs/poc_train_real_llm_syn_cantonese_num_measure_2voice_full_testdev_cer_sft_lr2e5_3ep.yaml) | 上行数据 + 粤语号码 1,686 + 粤语度量 1,806（每归一化文本 ≤2 音色）；`save_best_metric=cer`；`wer_max_new_tokens=512` | CER 最佳 `0.0739`，checkpoint-250 | `9.76%` WER、`7.39%` CER；22/53 exact | [`output`](../archive/outputs/poc_train_real_llm_syn_cantonese_num_measure_2voice_full_testdev_cer_sft_lr2e5_3ep)；[`predictions`](../archive/outputs/poc_train_real_llm_syn_cantonese_num_measure_2voice_full_testdev_cer_sft_lr2e5_3ep/eval/checkpoint-250_poc_test_predictions.jsonl) |

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
- 按 WER 选出本轮最佳 checkpoint-250（与 checkpoint-600 同为 9.76%，CER 更低）；2026-08-21 清理其余 5 个 checkpoint（200/225/575/600/753，其中 753 为未评测的最终 step），见 §8。

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

产物：第一轮 [`rollout_radiology/`](../archive/outputs/poc_train_real_llm_syn_cantonese_num_measure_2voice_full_testdev_cer_sft_lr2e5_3ep/rollout_radiology/rollout_loop_test.jsonl)（实际为贪心 ×10/音频）；复测 [`rollout_radiology_v3/`](../archive/outputs/poc_train_real_llm_syn_cantonese_num_measure_2voice_full_testdev_cer_sft_lr2e5_3ep/rollout_radiology_v3/rollout_loop_test.jsonl)。

### 2026-09-21 SFT-3900 真实域 prompt 热词 continue-SFT（负结果）

配置 [`poc_sft3900_real_hotword_prompt_ft.yaml`](../archive/configs/poc_sft3900_real_hotword_prompt_ft.yaml)。从 plus0821 `checkpoint-3900` 继续训，冻结 `encoder,aligner`，`lr=5e-6`，2 epoch，无在线增强。训练数据只含真实域：空 prompt 2902 条 + 带 prompt 722 条（金标 449 / 金标+干扰 166 / 干扰 107），POC 53 条 test 未入训。清单由 [`tools/build_hotword_prompt_jsonl.py`](../tools/build_hotword_prompt_jsonl.py) 从 plus0821 `train.jsonl` 抽取。

| 模型 | 训练 CER | 53 条 POC WER | 10 条 gold 命中 / hard 命中 | 10 条 distractor 插入 |
|---|---:|---:|---|---:|
| SFT-3900 | 5.63% | 7.65% | 62% / 52% | 35% |
| prompt-FT 最佳 step-100 | 5.89% | 8.00% | 62% / 52% | 40% |

金标命中与 3900 相同；`SN 1`、`coronal full slabs`、`frozen and paraffin sections` 等模板错误仍在。空 context 略退化（10 条 WER 14.38% → 15.69%）。结论：几百条真实域 prompt 不足以覆盖 3900 已学死的空 context 习惯。若再做热词恢复，应把 prompt 扩到 67k 混训，或改同一音频 gold vs distractor 的对比学习，而不是重复本配方。

权重（checkpoint-25/100/200/228，约 44GB）已删。保留 [`logs/`](../archive/outputs/poc_sft3900_real_hotword_prompt_ft/logs/)、[`eval/`](../archive/outputs/poc_sft3900_real_hotword_prompt_ft/eval/)、[`eval_hotword_context_probe/summary.md`](../outputs/eval_hotword_context_probe/summary.md)。

### 2026-10-21 Radiology-only oracle SFT（可行性验证）〔2026-10-23 已作废：基于旧标签〕

> **作废说明（2026-10-23）**：radiology 标签当日做了系统性修正（`raw/POC_test/radiology.csv`：逐位读数改为 CJK 原样、姓前缀/话语标记边界修正），且评分器存在前导零丢失 bug（已修复，见 `evaluation/english_medical/text_normalization.py`）。本节所有数字（含"稳定 17 条 61.88% vs 65.80%"）均为旧标签口径，**只作历史记录，不再作为结论依据**。产物已删除，重跑见「2026-10-23 Radiology oracle v2（新标签）」。

radiology 此前从未进入监督训练，只作为 `training.extra_eval_sets`（95 段，报告-only，不参与选点）和 `noise_from_radiology` 背景噪声来源。本实验回答：只给 radiology 目标域数据，模型能否学会该场景。

数据：`raw/POC_test/radiology.csv` 共 95 段 / 21.0 min / 10 条录音。按录音 8/2 留出 **DocB/1866S + DocC/1435R**（19 段 / 4.08 min，占 19.4%），其余 8 条录音 76 段 / 16.92 min 训练。标签为 `language None<asr_text>{原文}`；radiology 是 raw 音频，故 `noise_aug=0`，关闭在线噪声与 nospeech，保留 speed 0.8–1.2 与 SpecAug。

训练：从 base Qwen3-ASR-1.7B 全参 SFT；因共享 GPU 用 `batch_size=1`。1-epoch 版 `grad_acc=1`（76 updates，warmup 0.1）；5-epoch 版同样 `grad_acc=1`（380 updates，warmup 0.02）。不做 checkpoint 选点，评最终 checkpoint。

| 实验 | holdout 19 corpus WER | holdout 中位 WER | >100% 灾难条数 | train 76 corpus WER | train 中位 WER | train exact |
|---|---:|---:|---:|---:|---:|---:|
| 基座 | 78.18% | 100.0% | 1 | 72.67% | 78.4% | 1/76 |
| plus0821 ckpt-3900 | **67.73%** | 75.0% | 1 | 72.95% | 64.3% | 1/76 |
| 1 epoch（ckpt-76） | 193.86% | 71.4% | 2 | 73.02% | 44.6% | 4/76 |
| 5 epochs（ckpt-380） | 75.45% | 76.9% | 1 | **36.77%** | **0.00%** | **39/76** |

5-epoch 关键对比：

- 在两边都稳定的 17 条 holdout 上，radiology oracle 为 **61.88%** WER，plus0821 ckpt-3900 为 **65.80%**；逐条配对 oracle 好 10/19、差 7，比基座好 12/19。
- 聚合 WER 的差距几乎全部来自同一条长片段 `DocB/1866S/g06`（54 ref tokens）：oracle 92 edits / 170%，冠军 38 edits / 70%；这一条占 oracle 全部编辑的 21%。
- holdout `eval_loss` 在 epoch 2 最低（1.362），随后升到 epoch 5 的 1.453，但生成 WER 持续改善到 epoch 5；这类小数据实验不能用 eval_loss 选点。
- 纯 radiology-only 会灾难性遗忘：POC 53 条 WER 从冠军 7.65% 退化到 1-epoch 38.82%、5-epoch **151.53%**；留出整录音（2 条）5-epoch WER 565.82%，出现严重循环。
- 1-epoch 的聚合 WER 被 2 条灾难样本主导；去掉后 17 条为 70.47%，与冠军在同一 17 条上的 70.5% 持平。5 epochs 把灾难条数从 2 降到 1，并在稳定集上超过冠军。

结论：**radiology 数据高度可学**（8 条录音、17 min、5 epochs 即可在 17/19 条留出片段上达到或超过冠军），剩余瓶颈是单条长片段循环和纯目标域训练导致的域外遗忘，而不是数据不可学。纯 radiology-only 配方不可部署；后续应做低权重 replay 并按逐条尾部指标选点。

产物：[配置](../archive/configs/poc_radiology_oracle_5ep.yaml)、[切分](../archive/data/poc_radiology_oracle/split.json)、[5-epoch 结果](../archive/outputs/poc_radiology_oracle_5ep/RESULTS.md)、[逐 epoch 指标](../archive/outputs/poc_radiology_oracle_5ep/epoch_summary.json)；1-epoch 结果见 [`archive/outputs/poc_radiology_oracle_fromscratch/RESULTS.md`](../archive/outputs/poc_radiology_oracle_fromscratch/RESULTS.md)。造数据脚本保留在 [`tools/prepare_radiology_oracle.py`](../tools/prepare_radiology_oracle.py)。

### 2026-10-23 Radiology 合成数据 continue-SFT（实验 B，负结果）

[`tools/build_tts_radiology_manifest.py`](../tools/build_tts_radiology_manifest.py) 将 `raw/POC_train/tts_radiology/` 的 CosyVoice radiology 合成（RAD_slref + RADP_pathref，4,001 句粤英口述 × 2 音色 = 8,002 条 / 16.7h，CJK 数字按读音标注）追加到主线 mix。注意：**plus0821 冠军 mix 中 youtube_med_en 有 23,383/23,813（98.2%）音频已从磁盘丢失**，builder 现按"音频存在"过滤，replay = 44,894 行，合计 52,896 行（synth 占 15.1%）；冠军配方已不可原样复现。从 3900 continue-SFT，lr 5e-6、warmup 0.1、1 epoch（1,653 步），选点仍在 53 条 POC dev CER。文本与 oracle 留出录音 0 重合（话语标记除外），holdout 评测仍干净。

| 模型 | POC53 WER | holdout19 WER | holdout19 CER | radiology95 WER |
|---|---:|---:|---:|---:|
| champion ckpt-3900 | 7.65% | 67.73% | 48.76% | 72.74% |
| B best（step-100） | 8.12% | 67.95% | 51.85% | 80.10% |
| B final（step-1653） | 9.88% | 66.36% | 50.02% | 88.44% |

结论：**合成 radiology 数据对真实录音评测无迁移**（holdout19 与冠军同为 ~67–68%）。它只改变了输出风格——radiology95 删除 958→764、插入 553→1240，用"跟住…啦"式流畅长句替换漏词，净 WER 反而恶化；POC53 也被继续训练扰动（best 即 step-100）。与 oracle 对照：17 min **真实**录音把共同稳定 17 条做到 61.88%（优于冠军 65.80%），而 16.7h 合成零收益——瓶颈是真实声学（节奏/迟疑/重叠/噪声），不是文本覆盖。据此放弃方案 A（从基座 + synth 重训）。后续 radiology 优化应优先补**真实录音**或以真实段做 replay。

产物：[配置](../configs/poc_plus0821_ttsradiology_cont3900.yaml)、[manifest 脚本](../tools/build_tts_radiology_manifest.py)、`outputs/poc_plus0821_ttsradiology_cont3900/`（评测 `eval_*`、轨迹在 `launcher.log`）。2026-10-23 标签修正后按新口径重打分（`outputs/eval_radiology_oracle_sets/rescore_2026-10-23.json`）：B best/final holdout19 `66.08/64.56%` vs 冠军 `64.30%`，结论不变（与冠军持平）。

### 2026-10-23 Synth-only oracle（同一合成数据从基座训，负结果）

针对"B 更新量是否太小"，用同一 `tts_radiology` 数据（8,002 行 / 16.7h，占比 100%）从 **base** 全参 SFT 3 epochs（753 步，lr 2e-5、eff batch 32），在线噪声换成 radiology 自抽底噪（`noise_from_radiology`，182 条，中位 0.62s），不做选点、逐 epoch 全部离线评测。

| checkpoint | holdout19 WER | holdout19 CER | >100% 灾难 | radiology95 WER | POC53 WER |
|---|---:|---:|---:|---:|---:|
| ep1（step-250） | 180.45% | 197.95% | 5 | 107.05% | 37.65% |
| ep3 最优（step-750） | **161.59%** | 165.97% | 6 | 108.92% | 31.29% |
| 终点（step-753） | 171.82% | 178.39% | 6 | 109.88% | 30.71% |

基线对照：holdout19 基座 `78.18%`、champion `67.73%`、真实 oracle-5ep `75.45%`（稳定 17 条 `61.88%`）。holdout 的 eval_loss 随训练单调恶化（2.007→2.133），0 条 exact。

结论：**"更新太少"被否定**——47 倍于真实 oracle 的合成时长、足量更新后，真实录音评测崩至 161–180%（比基座差一倍以上），灾难片段 5→6/19，失败模式是用合成学到的"流畅放射科长句"幻觉替换真实口述中的迟疑/重启。B 的持平不是更新不足，而是主线真实数据在托底。合成文本覆盖与目标环境底噪（含 tile 伪影风险）都不能替代真实声学。radiology 提升路线只能建立在**真实录音**上（补采数据，或以真实 76 段做低权重 replay）。

产物：[配置](../configs/poc_tts_radiology_synthonly_fromscratch.yaml)、`outputs/poc_tts_radiology_synthonly_fromscratch/`（`synthonly_summary.json`、逐 checkpoint `eval_final/`）。新标签重打分后 synth holdout19 为 `170.38–181.77%`（负结论更强，见 `rescore_2026-10-23.json`）。

### 2026-10-23 Radiology oracle v2（修正标签重跑，结论恢复且更强）

旧 oracle 因标签问题作废后，用修正的 `raw/POC_test/radiology.csv`（逐位读数 CJK 原样、姓名/边界修正）+ 修复前导零的归一化器重建 manifest（同切分：train 76 段 / holdout19 段，DocB/1866S+DocC/1435R），配方与作废版逐字一致（基座、bs1/ga1、lr 2e-5、warmup 0.02、5 epochs=380 updates、speed+SpecAug、无 noise/nospeech、不选点）。train_loss 0.375（旧标签版 0.611——干净标签拟合容易得多）。

| ckpt (epoch) | holdout19 WER | median | >100% 灾难 | train76 WER | train exact | POC53 WER |
|---|---:|---:|---:|---:|---:|---:|
| 76 (1) | 60.51% | 68.8% | 0 | 20.18% | 15/76 | — |
| 152 (2) | 57.22% | 75.0% | 0 | 10.75% | 29/76 | — |
| **228 (3)** | **55.19%** | 66.7% | 1 | 9.25% | 54/76 | 35.18% |
| 304 (4) | 57.97% | 66.7% | 3 | 5.14% | 56/76 | — |
| 380 (5) | 57.22% | 66.7% | 2 | **4.63%** | **60/76** | 37.65% |

对照（同一新标签评分，`outputs/eval_radiology_oracle_sets/rescore_2026-10-23.json`）：holdout19 基座 `76.46%`、champion ckpt-3900 `64.30%`；radiology95 基座 67.72%、champion 66.47%（oracle-380 的 95 条含 76 条已见，不可比）。逐条配对：oracle 比基座好 12/19、差 2；比冠军好 7、差 7；三方都稳定的 17 条上 oracle **54.11%** vs 冠军 63.40% vs 基座 75.33%。

结论：

1. **"真实数据可学且优于冠军"在新标签下成立，且比旧口径更强**：聚合 WER 就领先冠军约 8pp（55.2–57.2% vs 64.3%）、领先基座约 19–21pp；旧版 oracle 在新标签下反而不如冠军（76.71%），说明旧数字的低估主要是**教给了模型错误字符串**，不是真实数据路线不行。
2. **尾部同步大幅收窄**：灾难片段 0–3/19（旧版终点 1–5 条且循环更长）；ep1–2 零灾难，ep3 聚合最优。合成 oracle（新标签 170%+）仍是负对照：文本充分性替代不了真实声学。
3. **POC53 不是"遗忘"**：这些 oracle 都从基座出发，POC53 = 35–38%，而**基座本身是 41.41%**——radiology 训练相对基座反而改善了病理集；但离病理冠军 7.65% 仍差 ~28pp，radiology-only 仍不可部署。这次"值得混入 replay"有了明确的价值锚点——holdout19 上约 −8pp 的真实增益。
4. **下一步（实验 C）**：从 champion 3900 出发，主线 replay + 这 76 段真实 radiology（适当上采样，如 10–15% 占比），lr 5e-6/warmup 0.1/1 epoch，判据：holdout19 < 64.30% 且 POC53 ≤ ~8.5%。

产物：[配置](../configs/poc_radiology_oracle_5ep_newlabels.yaml)、`outputs/poc_radiology_oracle_5ep_newlabels/`（5 个 epoch ckpt、`eval_epochs/`、`eval_poc53_ckpt{228,380}/`、`oracle_v2_summary.json`）、评测脚本 `eval_oracle_v2.py`、manifest `data/poc_radiology_oracle/`。

### 2026-10-27 ep1（MOSS 新音色）synth-only 复核：换音色不改变合成→真实鸿沟

`raw/POC_train/tts_radiology/ep1.manifest.json` 用同一批 4,001 条 radiology 文本、新 TTS 音色（MOSS，7.60h，24kHz）重新合成。文本与 holdout19 仍是 0 重合。配方与旧 synth-only oracle 完全一致（基座、bs2×ga16、lr 2e-5、3 epochs=378 updates、speed+SpecAug、radiology 底噪、nospeech、不选点），新标签口径评测。

| 模型 | holdout19 WER | radiology95 WER | POC53 WER | ep1 自拟合 |
|---|---:|---:|---:|---:|
| 基座 | 76.46% | 67.72% | — | — |
| champion 3900 | **64.30%** | 66.47% | 7.65% | — |
| 旧音色 synth-only（RAD/RADP） | 170.38–181.77% | 112.13% | 30.7% | — |
| **ep1 synth-only（ep1-378）** | **178.99%** | 69.70% | 28.0% | **2.30% / 86% exact** |

结论：**音色不是瓶颈**。ep1 训练集自拟合 2.30% WER、86/100 完全正确（CER 1.50%，loss 收到 ~1.8），说明模型完全能拟合这套合成语音；但真实未见录音仍是 ~178%，与旧音色同一灾难水平（且远差于什么都不做的基座 76%）。radiology95 的 69.7% 优于旧音色的 112%，但 holdout19 判据不变。至此"换 TTS 音色以缩小声学域差"的假设被否定；radiology 只能靠真实录音（oracle v2：17 min 真实数据即 55–57%）。

产物：`configs/poc_tts_radiology_ep1_synthonly.yaml`、`data/poc_train_tts_radiology_ep1_synthonly/`、`outputs/poc_tts_radiology_ep1_synthonly/`（3 个 epoch ckpt、`ep1_summary.json`、100 条自拟合抽样评测）。

### 2026-10-27 tts_radiology 录音环境增强实验（mega-asr-sim + 真实 RIR + LUFS）

假设：radiology 评测差主要来自**录音环境**（远场/混响/噪声/设备），而非内容。约束：**训练数据仅限 `raw/POC_train/tts_radiology`**（3 个 CosyVoice 音色 × 4,001 条）。实现：离线用 [mega-asr-sim](../tools/simulate_radiology_rir.py) 生成 54 场景体系中的环境子集；另写 [`simulate_radiology_rir.py`](../tools/simulate_radiology_rir.py)（真实 RIR 卷积，RIRS_NOISES 60k 房间 + 417 实测）、[`normalize_lufs_dir.py`](../tools/normalize_lufs_dir.py)（响度对齐）、[`build_sim_radiology_manifest.py`](../tools/build_sim_radiology_manifest.py)（含 severity 过滤）；评测用 [`eval_radiology_checkpoints.py`](../tools/eval_radiology_checkpoints.py)。全部从 base 全参 SFT（bs2×ga16，lr 2e-5，speed+SpecAug，不选点，按 checkpoint 扫最优）。

真实录音的声学画像（radiology95）：**SNR 代理 27.2 dB、centroid 630 Hz、HF 0.005、LUFS -22.1**；clean synth 51.2 dB/-20.1；simA 的 `noise` 场景 8–10 dB、centroid ~1200 Hz（明显过噪过亮），`far_field` 频谱匹配但 -32.6 LUFS（偏轻 10 dB）。

| 实验 | 数据/配方 | radiology95 最优 WER | 结论 |
|---|---|---:|---|
| 基线 raw synth（ep1，无增强） | 12,003 行 | 69.70% | 起点 |
| E1 | simA：7 环境场景，12,003 行，3 epochs | **47.12%**（ep1/376步） | 增强大幅有效；ep2/3 过拟合到 72/84% |
| E2 | 真实 RIR 卷积 4 场景，12,003 行 | 76.84% | 真实 RIR 单独使用有害 |
| E4 | simA ×3 seeds，36,009 行，1 ep | **42.47%**（ckpt-564） | 多样性收益 +4.7pp；最优权重保留 |
| E5 | 去噪场景 + 全部 -22 LUFS，24,006 行 | 61.34% | 只对齐声学统计、去掉噪声多样性 = 退化 |
| E7 | 从 E4-564 热启动 → simB 低 LR | 56.87% | continue-SFT 再次破坏 |
| E8 | simA ×3，lr 1e-5 | 45.22% | 降 LR 只把最优前移 |
| E9 | simA ×3，滤掉 severity>0.7 | 43.13% | severity 过滤无增益 |
| E11 | simA ×3 + clean 48k 混合 | 52.33% | 干净数据稀释增强收益 |
| E12 | simA ×5 seeds，60k | 45.36% | 继续加种子不涨 |
| E13 | simA ×3，全部 -22 LUFS | **41.63%**（ckpt-752）；nng4 41.37% | 响度修正略好；greedy 最优 |
| E14 | simA ×3，换 freesound 噪声池（9,511 文件） | 62.29% | 目标域噪声不可替换 |
| E15 | 自定义 mild：SNR 25–5 dB + 低通 + -22 LUFS，目标域噪声，×3 seeds | 57.90% | 温和噪声失败——需要 Mega-ASR 的高强度噪声 |
| 推理侧（E4-564） | no_repeat_ngram_size=4 | **41.19%** | 全场最优；nng5 41.85、nng3 47.49、rp1.15 50.79 |

其他负结果：E4-564 + E9-188 权重平均 55.84%；loop 截断无效（nng4 下已无循环，剩余 41.19% 是分散内容错误）；强制 `language English` 73.62%。

结论：

1. **环境增强是本次唯一有效的杠杆**：单模型 69.70% → 42.47%（greedy）/ **41.19%（nng4）**；叠加 **双模型取较短组合 → 39.36%，<40% 目标达成**。注意这是双模型推理配方（E4-564 + E13-752，均 nng4），"取较短"规则在 radiology95 上选定、在 holdout19 上验证（54.68% vs 单模型 59.75%）；单模型仍差 1.2pp。
2. **有效成分是"目标域噪声多样性 + 场景混合 + 足够强度"**：真实 RIR、去噪、响度对齐后的干净环境、severity 过滤、freesound 噪声、温和噪声（E15，25–5 dB）都单独失败；3 个退化种子、Mega-ASR 原始强度（-5..10 dB 噪声）的 simA 混合最强。目标域噪声源不可替换（E14 62.29%），强度也不可温和化（E15 57.90%）。
3. **轨迹高度不稳定**：同一数据 1 epoch 内会在 42–84% 间跳动，最优 checkpoint 需按 radiology95 扫描；更多种子/更低 LR 都不能消除。
4. POC53 = 31.3%（E4-564+nng4），**不是相对基座的遗忘**（基座 41.41%，radiology 训练反而改善），但远不及病理冠军 7.65%，纯 tts_radiology 配方不可部署；radiology 的真实增益上限仍由真实录音 oracle v2（55.19% holdout19）指示。

产物：`data/sim_radiology/`（simA/simB/simF/rir 输出、`RESULTS_sim.md` 明细）、`data/poc_train_tts_radiology_simA3x/`（最优数据 36k）、`outputs/poc_tts_radiology_simA3x_1ep/checkpoint-564`（最优单模型，42.47% / nng4 41.19%）、`outputs/poc_tts_radiology_simA3xn_1ep/checkpoint-752`（次优，41.63% / 41.37%）、`outputs/.../eval_decode/combined_short/`（**39.36% 组合预测与 summary**）、[`tools/combine_predictions_shortest.py`](../tools/combine_predictions_shortest.py)。其余失败实验权重已清理，评测日志/预测保留。

### 2026-10-28 真实目标域 + 合成 radiology 混训（real_target_domain mix）

假设：E4 已证明环境增强能把合成→真实的差距压到 42%，但纯合成数据缺少真实麦克风/房间/说话人变化，且会把病理能力打到 31.3%。本次放开真实数据，只控制配比。实现：[`build_radiology_real_mix.py`](../tools/build_radiology_real_mix.py) 把 `raw/POC_train/real_target_domain` 按倍数混入 E4 simA3x。

数据：1,263 条真实病理（raw/denoised 各一份，各 2.65h）+ 188 条 0821（0.52h），与 radiology95（DocA–D 放射科）**无交集**。配方仍是从 base 全参 SFT 1 epoch（bs2×ga16、lr 2e-5、warmup 0.02、speed+SpecAug、nospeech），只改真实占比。

| 实验 | 真实占比 | radiology95 最优 WER | POC53 |
|---|---:|---:|---:|
| real-only（3 epochs） | 100% | 95.09% | 16.24% |
| raw-only | 3.3% | 41.52% | — |
| M1 | 7.0% | 38.15%（ckpt-600） | — |
| M2 | 13.1% | 38.11%（ckpt-1295） | 14.24% |
| M2b（seed 777） | 13.1% | 38.73%（ckpt-780） | — |
| M2c（seed 2027） | 13.1% | 37.71%（ckpt-1040） | — |
| M4 | 23.2% | 39.79%（ckpt-1440） | — |
| M1 半 epoch | 7.0% | 39.87% | — |
| **soup(M1-600, M2b-780)** | 7%+13% | **37.12%** | **13.53%** |
| 预测集成(M2-1295, M2b-780) | 13%×2 | 37.30% | — |

结论：

1. **7–13% 是甜点**：单点 38.1%（对比 E4 greedy 42.47 / nng4 41.19），23% 开始稀释（39.79），3.3% 不足（41.52），半 epoch 更差（39.87）。真实病理数据本身并不教放射科（real-only 95.09%），增益来自真实声学分布与正则化。
2. **双目标收益**：混入真实数据后 POC53 从纯 radiology 的 31.3% 回到 13.5–14.2%，同时 radiology95 还降了 4pp；这是纯合成配方做不到的。
3. **波动约 1pp**：同配比三个种子的最优点为 38.11/38.73/37.71（均值 38.18）。最优点位置不稳定（600/780/1040/1295 都出现过），终点不可靠（M2 终点 38.11，M2b 终点 54.31），**必须扫描 checkpoint，不能取最后一个**。
4. 权重平均只在**不同配比**之间有效（M1+M2b = 37.12，M1+M2 = 37.71）；同配比不同种子平均反而变差（39.21），三模型 soup 38.48。预测集成两个成员 37.30，加入第三个成员变差（39.28）。
5. 注意选择偏差：37.12% 是在 radiology95 上从约 6 个 soup 中挑出的；独立子集 holdout19 上 soup = 48.61%，仍优于旧最优组合 54.68%，排序一致，但绝对数字应视为略偏乐观。

产物：`data/radiology_mix/RESULTS_mix.md`（完整明细）、`data/poc_train_radiology_mix_{m1,m2,m2b,m2c,m4,raw,realonly}/`、`configs/poc_radiology_mix_*.yaml`、`outputs/soup_mix_m1_m2b`（**37.12% 最优单模型**）、组件 checkpoint `outputs/poc_radiology_mix_{m1_1ep/checkpoint-600,m2_1ep/checkpoint-1295,m2b_1ep/checkpoint-780,m2c_1ep/checkpoint-1040}`。

### 2026-10-29 语言先验清洗（负结果）与 sim 文本泄漏

动机：m1_yt（M1 + 全量 YouTube）把训练集 CJK 字符占比压到 **0.47%**（评测 5.98% / 5.28%），且 tts_radiology 的 ep1/RAD_slref/RADP_pathref 是**同一批 4,001 条文本的三个音色**（simA3x = 4,001 × 9，72.8h 只有 4,001 个唯一句子）。本轮按"语言先验 + 文本多样性"重建数据。

数据构建（[`build_clean_lang_manifests.py`](../tools/build_clean_lang_manifests.py)）：

- **去污**：删除与 radiology95 参考共享 ≥8-gram 的 ep1 文本 177 条（其中 57 条含粤语）；RADX/YouTube/真实数据与评测 0 重叠。
- **sim**：ep1 3,800 条 + RADX_slref 16,030 条（全新 16k 文本）+ 粤语行三音色（315 × ep1/RAD_slref/RADP_pathref）= 20,460 行；mega-asr-sim **1 个环境种子**、7 场景。
- **真实病理**：raw + denoised + 0821 = 2,714 行 / 5.8h。
- **YouTube 严格过滤**：去第二人称教学语体、无终止标点的半句、无放射术语的行、重复文本 → **3,247 行 / 14.6h**（原 36,428 行 / 164.8h）。
- 合计 **26,421 行 / 63.7h / 825 步**；CJK 1.08%、粤语行 4.3%、YouTube 行占比 48.5%→12.3%；评测 8-gram 重叠 0 行。

结果（best ckpt-600，三个指标同点最优）：

| 模型 | radiology95 | holdout19 | POC53 |
|---|---:|---:|---:|
| **clean-lang-600（本轮）** | **41.44%** | **56.96%** | **11.88%** |
| m1_yt-2100（M1 + 全量 YouTube） | 36.39% | 52.91% | 14.71% |
| M1-600 | 38.15% | 54.18% | — |
| M2-1295 | 38.11% | 50.38% | 14.24% |
| soup(M1-600, M2b-780) | 37.12% | 48.61% | 13.53% |

干净子集对照（剔除曾被 ep1 文本泄漏的 23 条 train76 + 2 条 holdout）：

| 模型 | train76 clean | holdout19 clean |
|---|---:|---:|
| clean-lang-600 | 38.25% | 62.54% |
| M1-600 | 33.51% | 62.54% |
| M2-1295 | 35.69% | 55.52% |
| m1_yt-2100 | 34.79% | 57.86% |
| soup | 32.83% | 56.19% |

结论：

1. **语言先验清洗在 radiology 侧无效**：radiology95 比 m1_yt 差 5.05pp、holdout19 差 4.05pp；且在**干净子集上同样更差**（tr76_clean 38.25 vs M1 33.51，ho19_clean 62.54 vs m1_yt 57.86）。不是去污造成的，也不是欠拟合（best ckpt 在 600/826，之后变差）。
2. **主因是声学暴露被砍掉约 3 倍**：sim 从 3 seed（每文本 9 次不同声学实现）降到 1 seed（1 次）。E4 已验证 1→3 seed 值 **4.65pp**，本次相对 m1_yt 退化 **5.05pp**，量级吻合。文本多样性 4k→20k 没有转化为声学鲁棒性。
3. **发现 sim 文本泄漏**：ep1 的 177/4,001 条文本与 radiology95 参考共享 ≥8-gram；train76 中 23/76 片段（30.3%）重叠、平均覆盖 14.5% 参考词、最高 60%；holdout19 仅 2/19（10.5%）、覆盖 5.2%。**此前所有 sim 系实验的 radiology95 绝对值都含 train76 文本泄漏，holdout19 才是公平集**；引用的 37.12/38.11/36.39 等 all95 数字应视为偏乐观。
4. **POC53 11.88% 是 radiology 混训分支的新低**（soup 13.53、m1_yt 14.71、M2 14.24；病理冠军 7.65% 仍是部署水平）：真实数据行占比 10.3% + RADX 文本多样性对病理侧 retention 有效，说明这套清洗配方并非全负。
5. 语言先验本身**没有被公平检验**：要回答它是否有用，需在 **3 seed 的干净 sim**（E4-level 声学多样性）上重跑；本次因用户决定停止而未做。长期看仍需真实粤语口述/工作流语料，靠上采样合成文本无法把 CJK 占比从 1% 提到评测的 6%。
6. 训练 loss 末段 5.29–6.15（m1_yt 3.16、M1 4.6），主要反映 20k 唯一文本更难拟合，不是选点依据；ckpt-600 在 73% 处最优。

产物：`data/radiology_clean_lang/`（`clean_stats.json`、`clean_*.jsonl`、`youtube_strict.jsonl`）、`data/sim_radiology/cleanlang_*`（20,460 条，4.8G）、`data/poc_train_radiology_clean_lang/`（`sim/train.jsonl`、`train.jsonl`、`mix_stats.json`）、`configs/poc_radiology_clean_lang_1ep.yaml`、`outputs/poc_radiology_clean_lang_1ep/`（best `checkpoint-600`）、`outputs/contaminated_eval_clips.json`（泄漏片段清单）。对照 m1_yt 产物：`data/poc_train_radiology_mix_m1_yt/`、`configs/poc_radiology_mix_m1_yt_1ep.yaml`、`outputs/poc_radiology_mix_m1_yt_1ep/`（best `checkpoint-2100`）。

2026-10-29 清理：仅保留 `clean-lang/checkpoint-600` 与 `m1_yt/checkpoint-2100`，删除其余 12 个 full-FT checkpoint（约 137G）；所有预测、summary 和评测日志保留。

## 5. GRPO 数据挖掘

挖掘实现为 [`tools/mine_qwen3_asr_grpo_data.py`](../tools/mine_qwen3_asr_grpo_data.py)，产物在 [`grpo_g16_mining`](../archive/outputs/poc_train_real_tcga_1to3_5silence_none_grpo_g16_mining)。每个候选音频使用同一 prompt/audio evidence 生成 16 条 rollout，使用 ground-truth text 计算 WER，并根据 mean/best/std/worst 分桶。

完整挖掘报告 [`mining_report.jsonl`](../archive/outputs/poc_train_real_tcga_1to3_5silence_none_grpo_g16_mining/mining_report.jsonl) 共 25,667 条：

| 类别 | 数量 | 解释 |
|---|---:|---|
| Easy | 11,293 | mean WER 低、方差低 |
| Mixed | 8,562 | 混合/中等难度，不能简单归入单一稳定性类别 |
| Catastrophic hallucination | 3,758 | 至少部分 rollout 完全偏离 |
| Unstable | 1,699 | 方差高、结果不稳定 |
| Recoverable | 285 | mean WER 高但 best WER 低，最适合 GRPO |
| Consistently wrong | 70 | best WER 高且方差低，GRPO 信号弱 |

按音频路径区分，真实域有 4,118 条，合成域有 21,549 条。最终用于全量 GRPO 的去重清单为 [`train_recoverable_unstable_catastrophic_mixed_dedup.jsonl`](../archive/outputs/poc_train_real_tcga_1to3_5silence_none_grpo_g16_mining/train_recoverable_unstable_catastrophic_mixed_dedup.jsonl)，训练日志记录为 12,196 条。

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

### 2026-08-21 checkpoint 清理

对最近三次实验按上文评测结果做 checkpoint 级清理，只保留最佳点（共删除 13 个 checkpoint，约 149GB）：

- 真实域 + LLM 合成 + 粤语号码/度量：保留 checkpoint-250（9.76% WER / 7.39% CER）；删除 200/225/575/600/753（753 为未评测的最终 step）。
- 真实域 + LLM 合成（CER 选点）：保留 checkpoint-275（10.24% WER）；删除 150/200/350/425（200 已退化至 54.82%）。
- 真实域 raw + denoised：保留 checkpoint-75（更新归一化重测 15.63%）与 checkpoint-125（旧归一化最佳 9.17%，两种口径最佳点不同，均保留）；删除 100/150/175/189（189 为未评测的最终 step）。

各保留 checkpoint 的评测 summary 与 predictions 完整保留在对应 `eval/` 目录。

### 2026-09-21 非主线归档

将已定论、不再与 plus0821 SFT-3900 / GRPO 竞争的配方移到 [`archive/`](../archive/README.md)：早期 TCGA/v2/real-only/2voice/fullvoice-without-0821、prompt 热词负结果、soup 插值、GRPO smoke/screen。根目录只留 plus0821 `checkpoint-3900`、GRPO v1/v2 权重，以及长音频评测。

### 2026-10-21 radiology oracle 归档

radiology-only oracle 是诊断实验，不在 plus0821 主线内。1-epoch 与 5-epoch 的配置、manifest、权重和评测已移到 [`archive/`](../archive/README.md)。**2026-10-23：因旧标签失效，归档产物（权重、manifest、评测、配置）已全部删除**；新标签 manifest 由 `tools/prepare_radiology_oracle.py` 重建，重跑见「oracle v2」。`tools/prepare_radiology_oracle.py` 保留在 `tools/`。

## 9. 后续建议

1. 在已有数字、单位和缩写归一化基础上，继续评估 `period`、编号分词、`SN1/SN 1` 和英式拼写；同时保留结构化错误指标，不要把 `all/not` 或 block 字母归一化掉。
2. 从真实目标域补充带有 `SN/non-SN`、编号、block、frozen/paraffin、hemithyroidectomy 和 lumpectomy 的短句，优先增加真实声学实例而不是继续重复采样。
3. GRPO 继续使用冻结 audio tower/aligner，但降低更新幅度，先在 525 条真实 hard samples 上做小步数对照，并同时报告 WER、CER、结构词准确率和 catastrophic rate。
4. 对 GRPO rollout 使用有效样本过滤，避免 Easy 和合成 catastrophic 样本稀释或污染组内优势信号。
5. radiology 后续不要继续纯 radiology-only 全参 SFT：用**真实** radiology 段做低权重 replay（与病理主线混训）——实验 B 已证明 CosyVoice radiology 合成对真实录音无迁移，只有真实声学（oracle 的 76 段）有效；针对 `1866S/g06` 这类长片段单独处理（更多长录音、长度控制或重复惩罚），并以逐条尾部指标（中位 WER、>100% 条数）选点，不要用 eval_loss。主线 mix 依赖的 youtube_med_en 已丢失 98.2%，任何从基座重训需先用 `build_tts_radiology_manifest.py` 的存在性过滤或改走 yue_ba_sl mix。
