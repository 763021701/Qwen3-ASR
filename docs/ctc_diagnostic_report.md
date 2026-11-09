# Qwen3-ASR + Auxiliary CTC 诊断报告

日期：2026-10-30  
对象：`outputs/poc_radiology_mix_m1_ctc_v3_cov_1ep`（M1 数据，λ=1.0，coverage 0.01，
未 detach 的旧实现），ckpt-600 / ckpt-1200。

> 2026-11-09：训练期 CTC 路线终止。代码、工具、配置与诊断产物已归档到
> `archive/ctc_auxiliary/`、`archive/configs/`、`archive/outputs/`、
> `archive/data/`；本文保留作为负结果记录。

## 结论速览

1. **12.5 Hz 不是瓶颈**：训练集 0 条不可对齐，margin 最小 1.29、p01 2.21；
   即使 speed 1.2x，也只有 0.008% 样本 margin <1.2。
2. **CTC 具备独立学习能力，且 CTC 联合训练确实改善了 AuT 可解码性**：冻结
   ckpt-600 的 AuT、换全新 head，32 条真实短语音 150 步内 train CER 降到
   **0%**；解冻 AuT+head 100 步内到 0%。Frozen-AuT probe（2,000 train /
   512 val，600 步）：CTC-600 的 Val CER **35.8%** vs baseline **45.7%**
   （-9.9pp），holdout19 51.8% vs 57.2%。不存在“CTC 泛化不足”的问题。
3. **没有明显的全局方向冲突，但 CTC 在 AuT 上量级占主导**：16 个随机 batch
   的参数空间 cosine **+0.16**（clean）/ **+0.17**（训练增广）；直接看 AuT
   输出的特征梯度，cosine 仅 **+0.015 / +0.022（近正交）**。真正的矛盾是
   量级：λ=1 时 CTC 在 AuT 输出上的特征梯度是 CE 的 **2.7-3.3 倍**，参数空间
   是 **4.6-5.5 倍（1.5-15x）**；逐参数仍有 6-38% 负 cosine，局部且样本相关。
   共享 encoder 的更新由 CTC 主导，这是主要问题。
4. **终止训练期 CTC 路线**。P1b 说明 CTC 泛化/可解码性不是问题；P2 说明
   CTC 在共享 AuT 上梯度主导（2.7-5.5x），造成负迁移，而 LLM 的插入是解码
   策略问题。推理侧 CTC verifier（n-best rerank / 片段裁剪）可作独立小实验，
   但不应再改训练目标；也不建议再调 λ/词表/head 结构。

## P0：CTC 监督有效性审计

### P0.1 帧数 / 目标长度 / 对齐余量

`T` 为 AuT 输出帧数（约 13 帧/秒，即 ~77 ms/帧，不是严格 12.5 Hz）；由
manifest 的 `duration_sec` 经 `_get_feat_extract_output_lengths` 估算，与逐条
feature extractor 可能有 ±1 帧差异。`U` 为 compact vocab 编码后的 target
token 数；`R` 为相邻重复 token 数；CTC 最小需要 `T >= U + R`。

| 集合 | n | T 中位 | U 中位 | R 中位 | margin 最小 | margin p01 | margin p05 | 不可对齐 | margin<1.2 | margin<1.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 38,723 | 82 | 20 | 0 | 1.29 | 2.21 | 2.64 | **0** | 0% | 0.015% |
| holdout19 | 19 | 118 | 33 | 0 | 1.93 | 1.93 | 1.93 | 0 | 0% | 0% |
| radiology95 | 95 | 150 | 44 | 0 | 1.84 | 1.84 | 2.13 | 0 | 0% | 0% |
| poc53 | 53 | 150 | 24 | 0 | 3.00 | 3.00 | 3.55 | 0 | 0% | 0% |

speed 1.2x 后：train 不可对齐 0 条，margin<1.2 仅 **0.008%**（3/38,723），
<1.5 为 0.13%；三个评测集 0 条。**12.5/13 Hz 的时序分辨率不是对齐瓶颈。**

### P0.2 UNK / 不可对齐 / zero_infinity

| 集合 | sp `<unk>` (id 0) | compact vocab 丢弃 token | zero_infinity 忽略 | U > T |
|---|---:|---:|---:|---:|
| train | 0 | 0 | 0 | 0 |
| holdout19 | 0 | 21 / 766 = **2.74%** | 0 | 0 |
| radiology95 | 0 | 113 / 5,130 = **2.20%** | 0 | 0 |
| poc53 | 0 | 62 / 1,432 = **4.33%** | 0 | 0 |

- SenseVoice tokenizer 本身没有 UNK；训练集 vocab 覆盖 100%。
- **评测集有 2-4% token 被“训练集 compact vocab”丢掉**：例如粤语姓氏字
  `盧/姓/羅/曾/顏/黃`、病理编号 `12009 → 1200`。这会给 CTC-only CER 设上限，
  也会让联合训练的 CTC target 少监督这些 token。训练集本身无此问题。
- `zero_infinity` 忽略样本为 0；也没有 `U > T`（当前代码会直接报错的那种）。

### P0.3 分语言 / 医学术语错误（CTC-only，ckpt-1200，radiology95）

| 子集 | n | CER | WER | S | D | I | I/N |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 95 | 40.29% | 64.02% | 1726 | 3246 | 483 | 3.57% |
| English-only | 11 | **48.01%** | 73.54% | 191 | 369 | 32 | 2.60% |
| Cantonese | 74 | 39.71% | 63.60% | 1397 | 2538 | 418 | 3.81% |
| Code-switch | 83 | 39.51% | 63.15% | 1534 | 2874 | 451 | 3.67% |
| Medical | 70 | 40.80% | 64.63% | 1531 | 2678 | 435 | 3.82% |

- 错误以**删除**为主（D 远大于 S），插入率仅 3.6%——CTC 头是保守解码器，不是
  乱插的来源。
- English-only 最差（CER 48%），粤语/中英混合略好（39-40%）。
- 医学术语 occurrence recall 仅 **44.65%**（71/159），医学术语是主要短板。
- 注意 radiology95 含 76 条 train76（与 sim 文本有重叠）；公平集 holdout19 的
  CTC-only 是 WER 73.92% / CER 46.46%（ckpt-1200）。

### P0.4 纯 CTC vs Coverage

固定抽 64 条训练样本，直接分离两项（`pure CTC` 为 per-sample total，
未按 target 长度归一；训练日志里的 `loss_ctc` 是长度/ batch 归一后的值）：

| checkpoint | pure CTC mean | pure CTC p50 | coverage mean | coverage penalty mean | penalty max | blank ratio |
|---|---:|---:|---:|---:|---:|---:|
| ckpt-600 | 42.05 | 35.89 | 26.99 | **0.100** | 4.14 | 76.6% |
| ckpt-1200 | 33.17 | 23.60 | 27.24 | **0.217** | 11.39 | 75.2% |

- pure CTC 从 600→1200 明显下降（42→33），head 一直在学。
- coverage penalty 在 ckpt-600 已经很小（均值 0.10），1200 略升（0.22），
  不是 loss 的主要成分；训练日志里的 `loss_ctc` 是两者之和，但 600 之后主要
  反映 pure CTC。
- **注意**：loss 数值小不代表梯度小——P2 显示 CTC 在 AuT 上的梯度反而远大于 CE。

### P0.5 Tokenizer 还原 / 文本规范化 / Greedy Collapse

- **Greedy Collapse 正确**：合成序列 `[1,1,0,2,2,3,0,3]` 的
  `ctc_greedy_decode` 与参考实现完全一致（`[1,0,2,3,0,3]`，去 blank 后
  `[1,2,3,3]`）。
- **Tokenizer 还原**：训练集 38,723/38,723 精确往返；holdout19 11/19、
  radiology95 50/95、poc53 26/53 精确往返。失败全部来自 compact vocab 丢 token
  （P0.2），不是 `sp.decode` 或空白处理 bug。
- **文本规范化**：`normalize_ctc_text` 的小写/去标点/连字符转空格行为与预期一致；
  与评测用的 `normalize_english` 差异（中文数字转阿拉伯、`SN1→sn 1`）只影响
  CTC-only WER 口径，不是解码 bug。
- **发现并修复一个 detached coverage 的 shape bug**：detached 分支的
  `coverage_log_probs` 是 `(N,T,C)`，却按 `(T,N,C)` 做了 `sum(dim=0)`，在
  batch!=1 时会维度不匹配/算错。已改为先 `transpose(0,1)`，并加单测。
  该 bug 不影响本轮 v3 full run（它用的是未 detach 的旧路径），但会影响后续
  detached 实验。

## P1a：32 条真实短语音过拟合

数据：`data/ctc_diag_32.jsonl`，32 条真实 radiology 片段（非 sim），
duration 2.08-7.71s，margin ≥ 2.23，无 token 丢弃。  
设置：ckpt-600 的 AuT/LLM 权重；**全新 head**；head-only 时冻结 AuT；
full 时只训练 AuT+head；coverage 0.01（detached）；无 LLM loss。

| mode | step | train CER | pure CTC | coverage | blank ratio |
|---|---:|---:|---:|---:|---:|
| head（冻结 AuT） | 25 | 61.30% | 42.81 | 3.10 | 85.6% |
| head | 50 | 9.94% | 9.56 | 0.59 | 74.6% |
| head | 75 | 2.41% | 1.81 | 0.01 | 72.0% |
| head | 100 | 1.05% | 0.57 | 0.01 | 71.8% |
| head | 150 | **0.00%** | 0.13 | 0.00 | 71.4% |
| head | 300 | 0.00% | 0.03 | 0.00 | 71.4% |
| full（AuT+head） | 25 | 86.72% | 52.85 | 5.36 | 93.5% |
| full | 50 | 21.13% | 14.63 | 0.04 | 74.0% |
| full | 75 | 1.36% | 1.92 | 0.31 | 73.3% |
| full | 100 | **0.00%** | 0.18 | 0.00 | 73.1% |
| full | 200 | 0.00% | 0.02 | 0.00 | 73.1% |

结论：**不存在基本学习能力或特征可分性问题**。冻结 AuT 的特征上，一个全新
head 就能把 32 条真实片段拟合到 0 CER；解冻 AuT 后同样可以。失败发生在
38k 全量泛化 + LLM 联合训练，而不是 head/特征/优化器本身。

## P1b：Frozen AuT CTC Probe（原始 SFT vs CTC ckpt）

目的：判断 CTC 联合训练是否真的改善了 AuT 表征的 **CTC 可解码性**。
做法：冻结 AuT，换**完全相同初始化**的全新 head；在固定 2,000 条训练 /
512 条验证（M1 manifest，seed 1234，互不重叠）上训练 600 步（bs=16，
lr=1e-3，coverage 0.01 detached）；三个模型使用同一 split、同一 head 初始化、
同一数据顺序和训练预算。另在 holdout19（未见真实集）上评测。

| AuT 来源 | Train CER | Val CER | holdout19 CER | best Val CER |
|---|---:|---:|---:|---:|
| baseline（原始 SFT M1-600，无 CTC） | 34.64% | 45.71% | 57.16% | 44.62% |
| CTC-v3-600 | 22.10% | **35.77%** | 51.80% | **35.77%** |
| CTC-v3-1200 | 20.73% | **35.05%** | **50.73%** | **35.05%** |

结论：

- **CTC 联合训练确实改善了 AuT 的 CTC 可解码性**：同 600 步的 baseline →
  CTC-600，Val CER 45.71% → 35.77%（-9.9pp），holdout19 57.16% → 51.80%
  （-5.4pp）；Train CER 34.64% → 22.10%。
- CTC-1200 只比 CTC-600 略好（Val 35.05 vs 35.77；holdout 50.73 vs 51.80），
  说明 CTC 对表征的增益在 600 步左右已基本饱和，继续训练没有带来新的可解码性。
- 因此：**“CTC 泛化不足”不是主要解释**——CTC 分支在冻结 AuT probe 上是
  **更好**的，不是更差。失败发生在 CTC 表征没有转化为 LLM 的 WER/插入改善。

## P2：CE vs CTC 在共享 AuT 上的梯度冲突

ckpt-600，16 个随机 batch（bs=2，seed 1234），无数据增强；λ=1.0；
CTC loss 不含 coverage。

批次层面（16 个 batch）：

- cosine 均值 **+0.160**（范围 -0.114 ~ +0.438）；
- ‖g_CE‖ 1.7–27.0，‖g_CTC‖ 15.6–92.5；
- **λ·‖g_CTC‖/‖g_CE‖ 均值 5.51（范围 2.03–15.34）**。

逐层（16 batch 聚合，AuT）：

| 层 | cosine 均值 | 负 cosine 参数比例 | λ·ratio |
|---|---:|---:|---:|
| conv2d1/2/3, conv_out | +0.23 ~ +0.40 | 16-25% | 3.95-4.77 |
| layer 0-13 | +0.20 ~ +0.24 | 18-29% | 4.31-4.67 |
| layer 14-19 | +0.15 ~ +0.22 | 9-19% | 4.73-4.93 |
| layer 20-23 | +0.04 ~ +0.16 | 21-31% | 3.61-4.56 |
| proj1 / proj2 | +0.040 / +0.020 | 25% / 22% | 3.06 / 1.89 |

结论：**没有明显的全局方向性冲突**——整体 cosine +0.16，所有层的 cosine
均值都为正。真正的矛盾是**量级**：λ=1 时 CTC 在 AuT 上的梯度范数是 CE 的
**5.5 倍（2-15x）**，共享 encoder 的更新被 CTC 主导。逐参数仍有 10-38% 的
负 cosine，且随 batch 变化（最早抽的前 4 个 easy sim batch 曾出现层均值
负值，扩到 16 个随机 batch 后消失），说明方向冲突是**局部、样本相关**的，
不是稳定的层级对立。

用训练实际增广（speed 0.5 + SpecAug 0.5 + nospeech 0.3）重跑 16 个随机 batch：
整体 cosine **+0.174**，各层 cosine 均值 +0.01 ~ +0.38，逐参数负 cosine
比例 6-34%；λ·ratio 均值 **4.64（1.45-7.48）**。结论不变：**没有全局方向
冲突，CTC 在 AuT 上仍有约 4-5x 的梯度量级优势。**

**更直接的共享表示梯度**（对 AuT 输出 `h` 求 `dCE/dh` 与 `dCTC/dh`，比参数
空间更贴近“两个目标是否在推同一个表征”）：

| 设置 | feature cosine | λ·‖g_CTC(h)‖/‖g_CE(h)‖ |
|---|---:|---:|
| clean | **+0.015** | **3.30** |
| 训练增广 | **+0.022** | **2.69** |

即在共享 AuT 输出上，两个目标几乎**正交**（不是反向），但 CTC 的特征梯度
约是 CE 的 **2.7-3.3x**。结合参数空间的 4-5x，可以确定：**不是方向性负
迁移，而是 CTC 的梯度量级压过 CE，导致共享 encoder 被 CTC 主导。**

补充：如果只看前 4 个 easy sim batch，会误判成“前/中层系统性反向”；随机
采样后该结论不成立。**主导问题是梯度量级，不是方向。**

## P2 判断：CTC 泛化不足，还是多任务负迁移？

**主要是多任务负迁移 + LLM 解码策略，不是 CTC 泛化不足。**

1. **CTC 泛化不是主要问题**：P1b 显示 CTC-v3-600 的冻结 AuT 在相同 head/
   数据/预算下，Val CER 比 baseline 低 9.9pp、holdout19 低 5.4pp；CTC 分支
   不仅学得动，而且表征更可解码。
2. **负迁移是主要问题**：P2 显示 λ=1 时 CTC 的 AuT 特征梯度是 CE 的 2.7-3.3x、
   参数梯度 4.6-5.5x；shared encoder 被 CTC 主导。CTC 把自己的可解码性
   做上去了，但 LLM 的 CE 只拿到少数梯度；结果是 fair WER 无收益、
   radiology95/POC53 退化、后期出现更多模板循环。
3. **LLM 的插入是解码策略问题**：CTC-only 头本身插入率仅 3-5%，LLM 是 ~16%；
   把 encoder 变得更 CTC-可解码，并不会改变 LLM 依赖语言先验插入的行为。
   换句话说，encoder 不是缺失环节，CTC 监督也不是插入的解药。

**最终判断：终止训练期 CTC 路线。** 不建议再调 λ/词表/head 结构；若还要用
CTC，只能是推理侧的保守 verifier（n-best rerank / 幻觉片段裁剪），且应作为
独立的小实验，不应再改训练目标。

## 四个问题的直接回答

1. **12.5 Hz 是否导致大量样本对齐余量不足？**
   **否。** 训练集 0 条不可对齐，margin 最小 1.29、p01 2.21；speed 1.2x 下
   margin<1.2 仅 0.008%。评测集 margin 最小 1.84。帧率不是瓶颈。

2. **CTC 是否具备独立学习能力？**
   **是，而且 CTC 联合训练还提升了 AuT 的可解码性。** 32 条真实短语音上，
   冻结 AuT + 全新 head 150 步 train CER 0%，解冻 AuT+head 100 步 0%。
   Frozen-AuT probe（2,000/512，600 步）：CTC-600 Val CER **35.8%** vs
   baseline **45.7%**，holdout19 51.8% vs 57.2%。CTC 泛化不是失败原因。

3. **CTC 与 LLM 是否存在明显梯度冲突？**
   **没有明显的全局方向冲突，但量级上 CTC 占主导。** 参数空间整体 cosine
   **+0.16**（clean）/ **+0.17**（训练增广）；直接对共享 AuT 输出求梯度，
   cosine 只有 **+0.015 / +0.022（近正交）**。真正的矛盾是 λ=1 时 CTC 的
   特征梯度是 CE 的 **2.7-3.3 倍**、参数梯度是 **4.6-5.5 倍（1.5-15x）**。
   逐参数仍有 6-38% 负 cosine，局部且随 batch 变化；不能用“前/中层系统性
   反向”来解释。把 λ 降到 0.1-0.3 可把量级比压到 ~0.3-1.6x。

4. **是否值得继续 CTC 路线？**
   **训练期 CTC：终止。** 理由不是 CTC 学不动或泛化不足（P1b 反而更好），
   而是：
   - CTC 在共享 AuT 上梯度主导（特征 2.7-3.3x、参数 4.6-5.5x），造成负迁移；
   - LLM 的插入是解码策略/语言先验问题（CTC-only I/N 3-5% vs LLM ~16%），
     让 encoder 更可解码并不会自动减少 LLM 插入；
   - CTC head 绝对精度仍有限（keep 的 ckpt-1200 trained head holdout CER
     46.5%），不足以当强 teacher。
   **不建议再调 λ/词表/head 结构。** 唯一可选的后续是独立的推理侧 CTC
   verifier（n-best rerank / 幻觉片段裁剪），不再改训练目标；若做，也只需
   小规模验证。

## 产物

- 脚本：`tools/audit_ctc_targets.py`、`tools/audit_ctc_losses.py`、
  `tools/audit_ctc_errors.py`、`tools/diagnose_ctc_overfit.py`、
  `tools/diagnose_ctc_gradients.py`、`tools/probe_frozen_aut.py`
- 数据/结果：`outputs/ctc_diagnostic/`
  - `targets_summary.json`、`targets_{train,holdout19,radiology95,poc53}.json`
  - `losses_600.json`、`losses_1200.json`
  - `ctc_eval_radiology95/ctc_predictions.jsonl`、`errors_radiology95.json`
  - `overfit_32.json`、`frozen_aut_probe.json`（baseline / CTC-600 / CTC-1200）
  - `gradients_600_16b.json`（16 个随机 batch，无增广）、
    `gradients_600_16b_aug.json`（同 16 batch，训练增广：speed+SpecAug+nospeech，
    主参考）；`gradients_600_8b.json` 为 8 batch 复核；
    `gradients_600.json` 是最早 4 个 easy sim batch，仅作偏差示例
- 32 条诊断集：`data/ctc_diag_32.jsonl`
- 修复：`finetuning/ctc_auxiliary.py` 的 detached coverage shape bug；
  `tests/test_ctc_auxiliary.py` 新增对应单测（9 项全过）。