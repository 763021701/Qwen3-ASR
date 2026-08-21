# Hong Kong–Accented English ASR Adaptation Report

## 1. 结论摘要

当前已评测 checkpoint 中的最优模型为 `outputs/tcga2_real_balanced_sft_lr2e5/checkpoint-300`：

- 训练数据：20,000 条真实采样行 + 20,000 条 TCGA2/VoxCPM 合成数据，比例 1:1；真实部分来自 1,263 条训练清单（842 个唯一音频）并均匀上采样。
- 训练配置：全量微调，学习率 `2e-5`，有效 batch size 32，启用速度扰动、加噪和 SpecAugment。原计划 3 epochs，实际运行在 step 742（约 0.59 epoch）停止；当前模型位于 step 300（约 0.24 epoch）。
- 修正标签后的 POC WER：**17.34%**（145/836），句子完全正确率 **28.30%**（15/53）。
- 相比真实数据全量微调基线 `checkpoint-40` 的 **18.90%**，绝对提升 **1.56 个百分点**，相对错误下降约 **8.2%**。

结果说明平衡加入合成数据可以带来增益，但仍需通过“等训练步数的纯真实数据上采样”对照实验，排除增益仅来自更多参数更新的可能。

## 2. 评测口径

- 主要适配指标使用 `data/poc_split/seg_test.jsonl`：53 条、836 个参考词；标签已修正，尤其是 `12712_seg_0001/0002`。
- WER 归一化为小写、去标点、合并空白。
- POC 集同时参与验证和 checkpoint 选择，因此当前结果属于开发集结果，不应视为最终独立测试集结论。
- 早期 474 条实验直接在训练清单上评测，仅用于诊断拟合能力，不能与 POC WER 直接比较。
- `outputs/` 中的维吾尔语、中文医疗、多语种平衡和独立 GigaSpeech LoRA 训练不属于本报告的香港口音病理听写适配主线，未纳入方案排名。

## 3. 实验汇总

### 3.1 早期 474 条训练内诊断

| 方案 | 训练内 WER | 观察 |
|---|---:|---|
| 原始 Qwen3-ASR-1.7B | 62.12% | 基座对香港口音病理听写适配不足 |
| 原始分段数据微调 | 15.54% | 训练集拟合明显 |
| 仅去噪数据 | 21.87% | 去噪未带来收益 |
| 去噪 + 数据增强 | 21.77% | 与仅去噪基本一致 |
| 去噪 + 增强 + 自提取噪声 | 20.05% | 小幅改善 |
| 上述方案延长训练：epoch 3/4/5 | 17.39% / 17.68% / 18.14% | epoch 3 后开始回退 |
| 三源混合：原始 + 去噪 + 增强，共 1,422 行 | 10.22% / 5.71% / 5.75% | epoch 1/2/3；训练内拟合最强，epoch 2 后饱和 |

这些数字主要反映记忆和拟合能力。原始音频优于去噪音频，也说明去噪可能损伤了口音、弱音节的有效声学线索。

### 3.2 合成数据与 POC 适配

| 方案 | 最佳点 | 修正后 WER | 结论 |
|---|---|---:|---|
| 20k TCGA 合成数据 + 在线增强 | 3 epochs | 未留存 | 完成训练，但没有可用的定量 WER |
| 20k TCGA2/VoxCPM 合成数据，仅合成训练 | step 200 | 68.38%* | 真实集 eval loss 1.6889；单独依赖合成数据迁移失败 |
| 1,263 行真实数据，全量微调 | step 40 | 18.90% | 真实数据基线；step 80/120 分别为 24.04%/22.73% |
| 冻结 Audio Tower 和 Align，仅更新完整 LLM | step 60 | 21.53% | eval loss 0.4811；不及全量微调 |
| 仅 LLM LoRA，r=8 | step 300 | 22.13% | 仅训练 8.72M 参数（0.426%）；约 7.5 epochs 后 loss 见底 |
| 真实与合成 1:1，全量微调 | step 100 | 21.41% | eval loss 最低，但 WER 不是最佳 |
| 同上 | step 200 | 18.06% | 超过真实数据基线 |
| 同上 | **step 300** | **17.34%** | 当前最优 |

\* `68.38%` 在 1,263 条真实 combined 清单上评测，与 53 条 POC 的评测集合不同，仅用于说明 synthetic-only 迁移效果很差。

去噪 POC 音频在 `checkpoint-80` 上得到 29.55%（旧标签口径），没有显示去噪收益。LoRA 与冻结声学侧、完整更新 LLM 的结果接近，但二者都明显弱于全量微调，说明当前瓶颈不只在语言模型侧，声学表征及跨模态对齐也需要适配。

## 4. 最优模型分析

### 4.1 Loss 与 WER 不一致

平衡采样实验的验证 loss 在 step 100/200/300 分别为 `0.4212/0.4642/0.5439`，持续变差；对应 WER 却为 `21.41%/18.06%/17.34%`，持续改善。

训练实际继续到 step 742，但 step 400--700 因验证 loss 不在最低三名而被自动删除，未能补算 WER。因此，17.34% 是当前留存并评测的最佳结果，尚不能视为已训练收敛。

因此，token-level 交叉熵 loss 不能可靠代表最终转写质量。后续应以归一化 WER 为主指标选择 checkpoint，loss 仅用于监控训练稳定性。

### 4.2 主要错误来源

最优模型共有 145 个词错误：69 个替换、19 个删除、57 个插入。错误高度集中，前 10 个高错误样本贡献 **84/145（57.9%）**；扩展到前 15 个样本后覆盖 **104/145（71.7%）**；前三个样本已贡献 48 个错误。

下表按词级错误数排序。`S/D/I` 分别表示替换、删除、插入；参考和预测列保留了能体现错误的关键片段。

| 排名 / 样本 | 错误数（S/D/I） | 参考关键内容 | 模型预测关键内容 | 具体错误 |
|---|---:|---|---|---|
| 1. `26SS12099_seg_0009.wav` | 23（1/1/21） | `2 specimen labelled left sentinel node` | `specimen labelled left axillary lymph node is received in formalin ... mass measuring 10 x 8 x 65 cm` | 删除数量 `2`；`sentinel` 被改成 `lymph`；将极短句扩写成完整标本描述，新增 21 个词，属于严重模板幻觉。 |
| 2. `12712_seg_0012.wav` | 14（4/4/6） | `... other smaller uterine fibroids 係总数 q and r endometrium ... end of dictation thank you` | `... others smaller uterine fibroids endometrium ... cervix and endocervical canal thank you slide 4 right fallopian tube resection` | 漏掉混合中文标签、`q/r` 和连接词；`end of dictation` 被改写；末尾生成不存在的 `slide 4 right fallopian tube resection`。 |
| 3. `26SS12099_seg_0008.wav` | 11（7/0/4） | `paraffin blocks f to h were subsequently taken to embed coronal full slabs of the tumour` | `... were submitted for evaluation when embedded no firm nodules were observed ...` | `subsequently taken to embed` 被改写成 `submitted for evaluation`；额外生成 `no firm nodules were observed`，整体从原始操作描述变成另一段病理模板。 |
| 4. `26SS12082_seg_0004.wav` | 6（6/0/0） | `and was bisected and all embedded in block a for frozen and paraffin sections` | `antral speci specimen labelled antral is in block a for frozen and paraffin sections` | `bisected`、`all embedded` 被替换为 `antral specimen labelled antral`，说明模型用常见标本模板覆盖了真实操作步骤。 |
| 5. `26SS12082_seg_0005.wav` | 6（5/1/0） | `SN 2 ... serially sectioned ... block b for frozen and paraffin sections` | `SF2 ... en bloc serially sectioned ... block b for gross and paraaffinity sections` | 删除 `SN`；`SN 2` 识别为 `SF2`；将 `and was` 替换为 `en bloc`，并将 `frozen` 识别为 `gross`，并把 `paraffin` 识别为 `paraaffinity`。 |
| 6. `26SS12099_seg_0013.wav` | 6（4/0/2） | `NonSN ... bisected and all embedded in block k for frozen and paraffin sections` | `the mass then ... was bisected and non embedded in block a for frozen and paraffin section` | 添加 `the mass`；`NonSN` 被拆成语义不完整的 `then/non`；block `k` 变成 `a`；`sections` 变为单数。 |
| 7. `12009_seg_0005.wav` | 5（4/1/0） | `section shows ... serially sectioned all embedded in 2 blocks` | `sectioning shows ... full resection all embedded in two blocks` | `section shows` 改成 `sectioning shows`；删除口语填充词 `uh`；`serially sectioned` 被改成 `full resection`；数字 `2` 被转写为 `two`。 |
| 8. `26SS12082_seg_0002.wav` | 5（3/2/0） | `three lymph nodes labelled SN 1 SN 2 and NonSN` | `three lymph nodes labelled SN1 SN2 and SN` | `SN 1/SN 2` 的空格边界丢失；删除 `SN` 和 `1`；最后的 `NonSN` 被错误归一化为 `SN`。 |
| 9. `12009_seg_0002.wav` | 4（2/1/1） | `一二多一個 uh it consists of a polypoid piece ...` | `25 1 1 it consists of a polypoid piece ...` | 中文短语被错误识别为 `25 1 1`；额外插入 `25`，并删除一个 `uh`。这是中英混合内容和短片段边界问题。 |
| 10. `12712_seg_0005.wav` | 4（1/2/1） | `... fibroids period serial sectioning ... fibroids period the largest fibroid over the` | `... fibroids serial sectioning ... fibroids the largest fibroid over 4 d` | 两个口述的 `period` 被删除；末尾新增 `4 d`；最后的 `the` 被错成 `d`，体现标点口述和截断句处理不稳。 |
| 11. `26SS11678_seg_0006.wav` | 4（3/1/0） | `... nodule for frozen and paraffin sections` | `... nodule for fullthickness parenchymal section` | 删除 `frozen`；`and paraffin sections` 被整体改成 `fullthickness parenchymal section`，属于病理术语模板替换。 |
| 12. `26SS11731_seg_0001.wav` | 4（2/0/2） | `26SS11731 specimen labelled left hemithyroid` | `26 ss 11731 specimen labelled left hemifuryroid` | 病例号被拆成 `26 ss 11731`；`hemithyroid` 被识别为 `hemifuryroid`，同时损失病例号整体匹配。 |
| 13. `26SS11731_seg_0009.wav` | 4（1/0/3） | `... d background thyroid` | `... d background fibroadipose tissue 总共4个 block。` | `background thyroid` 被替换为 `fibroadipose tissue`；末尾新增 `总共4个 block`，属于语义合理但无声学依据的续写。 |
| 14. `26SS12082_seg_0007.wav` | 4（1/3/0） | `and was serially sectioned and all embedded in block c ...` | `anteriorly sectioned and embedded in block c ...` | 删除 `and/was/all`；`serially` 被替换为 `anteriorly`，流程动作和修饰词均发生变化。 |
| 15. `26SS12099_seg_0002.wav` | 4（3/0/1） | `1 specimen labelled left lumpectomy ... a lumpectomy specimen ...` | `the other specimen labelled left lymphadenectomy ... a lymphadenectomy specimen ...` | 数字 `1` 被改为 `the other`；`lumpectomy` 两次都被替换为 `lymphadenectomy`，属于关键手术类型错误。 |

主要错误类型如下：

1. **模板化幻觉和合理续写**：模型根据病理语境生成“听起来合理”的完整句子，造成大量插入，是目前最大单项风险。
2. **病例编号、缩写和块号**：`26SS...`、`SN1/SN2/Non-SN`、block 字母等短实体容易被替换、拆分或遗漏。
3. **病理操作和术语替换**：如 `bisected`、`serially sectioned`、`embedded`、`frozen/paraffin sections` 被改写为其他常见流程。
4. **中英混合及格式归一化**：中文提示词、尺寸表达、乘号、数字和英美拼写差异会形成部分真实错误或计分伪差异。

### 4.3 通用能力退化

在 `data/gigaspeech/test.jsonl` 的 9,102 条通用英语上：

| 模型 | WER |
|---|---:|
| 原始、未微调 Qwen3-ASR-1.7B | 3.44% |
| 当前最优病理模型 | 4.45% |

领域适配造成 **+1.01 个百分点**的绝对退化，相对错误增加约 **29.4%**。病理域收益明确，但需要增加通用语音回放或正则化以控制遗忘。

## 5. 后续优化计划

1. **补齐当前训练**：从留存 checkpoint 恢复并完成计划的 3 epochs；保存候选点时不能只按 loss 淘汰，应先完成 WER 评测。
2. **建立独立测试集**：按病例划分 train/dev/test；POC dev 用于 checkpoint 选择，最终 test 只评测一次。
3. **按 WER 选 checkpoint**：每个保存点直接计算 POC-dev WER，同时记录幻觉率、插入率和实体错误率。
4. **完成关键对照**：将纯真实数据同样上采样至 40k，并保持训练步数、增强和超参数一致，确认合成数据的净贡献。
5. **抑制模板幻觉**：加入更多短音频和“停止输出”样本；按音频时长限制输出长度；过滤重复模板或与真实声学分布偏离的合成数据。
6. **强化难实体**：统一病例号、`SN/Non-SN`、尺寸、block 和中英混合标签；增加实体定向采样，并尝试上下文词表偏置。
7. **优化混合策略**：对比真实:合成 `2:1`、`1:1`、`1:2`，以及先合成预适配、再真实数据收敛的 curriculum 方案。
8. **控制通用域遗忘**：混入少量 GigaSpeech replay 数据，联合监控病理 POC WER 与通用 WER。

## 6. 关键产物

- 最优 checkpoint：`outputs/tcga2_real_balanced_sft_lr2e5/checkpoint-300`
- 最优预测：`outputs/tcga2_real_balanced_sft_lr2e5/eval_step300/predictions.jsonl`
- 修正后摘要：`outputs/tcga2_real_balanced_sft_lr2e5/eval_step300_corrected/wer_summary.txt`
- 通用域基座评测：`outputs/gigaspeech_general_eval/base/wer_summary.txt`
- 通用域最优模型评测：`outputs/gigaspeech_general_eval/balanced_step300/wer_summary.txt`
