# 噪声与 RIR 的固定划分

训练只读取 v1 的 train 清单，dev/test 用于生成固定的开发集、测试集增强音频。
原始音频、软链接和外部数据集均不移动。清单、指纹缓存及报告位于仓库的
`data/augmentation/splits/v1/`，与其他本地数据一样受 `.gitignore` 排除；
代码和本文档保存在 Git 中。换机器时须复制该目录及素材，或使用相同数据运行生成工具。

## v1 数据与结果

种子为 42。DNS/MUSAN 按来源、类别的时长目标 80/10/10 分层，以完整同源组
分配，允许组大小造成的比例偏差。音乐有艺人信息时跨 MUSAN 子来源绑定同一艺人；
例如某子来源只有很少艺人时，不强求三份都覆盖。语音噪声仅有录音级身份，
不声称实现了说话人隔离。

| 素材 | train | dev | test |
|---|---:|---:|---:|
| 噪声采样条目 | 53,801 | 6,960 | 6,954 |
| 噪声时长（小时） | 231.783520 | 28.540845 | 28.564416 |
| RIR 采样条目 | 48,263 | 6,031 | 6,031 |
| 其中仿真 RIR | 48,000 | 6,000 | 6,000 |
| 其中实测 RIR | 263 | 31 | 31 |
| 仿真房间数 | 480 | 60 | 60 |

噪声输入 67,826 个版本，标准化音频内容归并后保留 67,715 个采样条目。
RIR 输入 120,573 个版本，绑定 16k/48k 同源版本后保留 60,325 个采样条目。
合计 188,399 个输入版本全部由清单中的主条目或 aliases 覆盖。
RIRS_NOISES 内 843 个 point-source noise、92 个 isotropic noise 不是 RIR，未纳入 RIR 索引。

## 同源分组

- DNS AudioSet 使用视频 ID；DNS Freesound 使用原始录音 ID，去掉片段编号及类别前缀。
  13 个无数字 ID 的 B32_Chips_ebrahim 片段作为一个完整来源组。
- ESC-50 使用 `meta/esc50.csv` 的 `src_file`，fold 1–3/4/5 映射 train/dev/test。
  原始录音跨划分时采用 test > dev > train。最终为 1200/398/402 个文件。
- DNS 和 ESC 共用的 52 个 Freesound 原始 ID 统一归组，对应 DNS 片段随 ESC 的划分，
  不允许同一录音换一个数据集名称进入其他 split。
- MUSAN 保留原始录音身份；可识别的艺人归组。指纹相同的整段或重叠片段进一步绑定，
  不把每次随机裁剪作为独立划分单位。
- SLR26 仿真 RIR 使用房间大小和 Room 编号；一个房间的 100 个位置、16k/48k
  版本全部归在一起。每种大小 200 个房间，分别为 160/20/20。
- SLR28 实测 RIR 统一 AIR 的麦克风类型、角度和位置别名；
  RWCP 的 e1a/e1b/e1c、e2a/e2b、jr1/jr2 保守归并；
  REVERB2014 按房间名称归组。无法从现有元数据识别物理房间的 RWCP type4
  整个采集族只用于 train。现有 rir_list 的细粒度 room-id 不直接视为物理房间。

所有分组约束及已发现的指纹关联先合并为连通组，再分配 split。
RIR 同源版本只选一份进入采样池，优先已有 16k 版本；48k 原文件保留为 aliases。
DNS 中没有 rir_list 的 248 个实测文件也收入库存，它们均对应原库的实测 RIR，
因此作为 aliases 覆盖，而不是遗漏或重复采样。

## 生成与独立验证

从仓库根目录，用 qwen3_asr_ctc_smoke 环境：

```bash
python tools/split_augmentation_assets.py \
  --noise_dir data/augmentation/noise \
  --rir_dir data/augmentation/rir \
  --output_dir data/augmentation/splits/v1 \
  --seed 42 --workers 8

python tools/split_augmentation_assets.py \
  --output_dir data/augmentation/splits/v1 --verify
```

输出六份 `noise.{train,dev,test}.jsonl`、`rir.{train,dev,test}.jsonl`，
以及 `inventory.jsonl`、`summary.json` 和 `fingerprints.sqlite`。
`summary.json` 保存输入数量、分布、种子、指纹版本和所有 JSONL 的 SHA-256。
已生成的划分不覆盖；素材或分组规则改变时创建 v2，不改写 v1。
中断且尚未生成 summary 的构建可重跑，使用文件签名匹配的指纹缓存。

独立验证检查：

1. train/dev/test 的原始录音 ID、房间/艺人绑定键、连通组 ID 无交集。
2. 主条目与全部 aliases 的实际路径不跨划分，每个输入版本恰好出现一次。
3. 标准化单声道 16k PCM 哈希，以及增益归一化 4 秒窗口指纹无跨划分交集。
4. 原文件大小、mtime 与指纹库存一致；冻结清单的 SHA-256 未变化；六份清单均非空。

v1 上述交集全部为 0。噪声 67,826 个文件和 RIR 60,325 个主版本实际计算了指纹；
RIR 的 60,248 个其他采样版本通过来源/房间身份绑定，不伪称其音频哈希相同。
窗口指纹使用 1 秒步长，排除静音和近常量窗口。它能补充发现部分裁剪重复，
不保证识别任意变速、任意偏移、重新录制或来源不明的编辑音频。
`verification.json` 为本次独立检查输出。

## 训练与评测接入

四份 robust 数据配方均已替换全目录扫描：

```yaml
simulation:
  asset_split: train
  noise_manifest: ../../data/augmentation/splits/v1/noise.train.jsonl
  rir_manifest: ../../data/augmentation/splits/v1/rir.train.jsonl
```

路径相对 YAML 文件解析。noise_dir 与 noise_manifest、rir_dir 与 rir_manifest
分别互斥。指定清单后，缺文件、空清单、混合 split 或 split 不符直接报错，
不回退全目录或白噪声。SFT/LoRA 训练入口拒绝使用 dev/test 的增强清单。
原有目录模式保留供旧脚本显式使用，不用于当前训练配方。

生成固定的 dev 增强集示例（test 改为 test 清单和 asset_split）：

```bash
python tools/materialize_robust_eval.py \
  --jsonl /path/to/clean_dev.jsonl \
  --output_dir outputs/robust_dev_v1 \
  --noise_manifest data/augmentation/splits/v1/noise.dev.jsonl \
  --rir_manifest data/augmentation/splits/v1/rir.dev.jsonl \
  --asset_split dev --reverb_mode mixed \
  --rir_real_prob 0.7 --rir_mix_prob 0.8 --seed 42
```

生成的评测 JSONL 记录 split、清单 SHA-256、实际素材 ID/组 ID、噪声裁剪位置、
RIR 通道和样本随机种子，便于复现。语音数据的 train/dev/test 仍须独立隔离；
正式验收继续采用 HM-ASR-Benchmark。

此改动不改变来源采样权重：NoisePool 仍按清单条目均匀抽样。
历史训练可访问全量旧素材池，新的划分不能证明历史检查点未见过留出素材。


## 受控鲁棒性评测 v1

`tools/build_controlled_robust_eval.py` 将语音、噪声裁剪、RIR 通道和音量固定，
同一份音频供基座与 LoRA 模型比较。它与随机模拟训练独立，也补充自然场景的
HM-ASR-Benchmark 结果；自然测试集上的好坏不能直接代表抗噪声能力。

当前输出为 `outputs/robust_mix_1000h/robustness_v1/`，包括：

- GigaSpeech 英语、WenetSpeech 普通话、Common Voice 粤语、KeSpeech 方言，
  分别采用官方 dev/test 语音；KeSpeech 按九个 subdialect 均衡选取。
- 每个语料每个 split 固定 200 条、2–15 秒，共 1,600 条语音。
- 每条语音生成 10 种条件，共 16,000 条音频、80 份清单：原始录音；
  20/10/0 dB 加噪；实测/仿真 RIR；两种 RIR 各叠加 10/0 dB 噪声。
- dev 语音只搭配 dev 噪声/RIR，test 同理；不读取 train 增强素材。
  另检查选中语音的 dev/test PCM、录音 ID 和有元数据的说话人身份无交集。
  这些检查不声称覆盖没有身份元数据的说话人，也不声称检查过基座预训练数据。
- 六种噪声来源按语句数均衡分配：DNS AudioSet、DNS Freesound、ESC-50、
  MUSAN music/noise/speech；同一句在不同 SNR 下使用同一个噪声片段。
  仿真 RIR 均衡覆盖 small/medium/large，实测 RIR 单独报告，不使用合成回退。

这里的 clean 指未叠加新增强的原始录音，仍可能存在原生背景噪声。
SNR 定义为整句参考波形 RMS 与新增噪声 RMS 之比；RIR+噪声使用混响后的
波形作为参考。因此它衡量新增干扰的强度，不是原始录音的总声学 SNR。

混响沿用训练加载器的 RIR 预处理（对齐峰值、抑制直达峰、裁剪及能量归一化），
卷积后截取原句长度，按原音/混响分量 RMS = 6 dB 混合。
这是固定 dry/wet 混响条件，不代表完整物理房间的 DRR 或 RT60 分级。
每句的所有条件共享一个防削波增益，保持相对电平；保存为单声道 16 kHz PCM16 FLAC。
不启用变速或 SpecAugment。

生成器复核下载的官方 dev 归档 SHA-256；test 使用 HM-ASR 已有的官方测试数据。
`dev_source_archives.json` 明确记录 dev 的仓库、版本、归档及校验和；
WenetSpeech dev 从已有 `wenetspeech_eval_wav/dev/metadata.csv` 读取。
不得将旧 `outputs/eval_clean` 小样本直接标为 dev/test，其中存在 train 来源。

在训练环境中生成及复核（已冻结的目录拒绝覆盖）：

```bash
python tools/build_controlled_robust_eval.py \
  --archives /absolute/path/dev_source_archives.json \
  --output_dir /absolute/path/robustness_v1 --samples 200 --seed 42

python tools/build_controlled_robust_eval.py \
  --output_dir /absolute/path/robustness_v1 --verify_only
```

`protocol.json` 固定协议和六份素材清单哈希；`speech_selection.jsonl` 保存语音来源；
`manifests/` 保存每个条件的引用、增强轨迹、裁剪位置、通道和音频 SHA-256；
`audio_checksums.json`、`suite_verification.json` 支持推理前后独立检查。
验证包括素材归属、同源组隔离、音频字节、成对文本/时长/增益、无削波、
编码后的 SNR 与 dry/wet 分量比例。完整素材库存的指纹与来源隔离检查仍由上文
`split_augmentation_assets.py --verify` 执行，不能只用抽样评测的覆盖量代替。

v1 中同一原始录音的不同噪声片段可能共享 `asset_id`。
审计单个文件必须同时匹配 asset_id、规范化实际路径与 group_id，
不能把 asset_id 当作唯一文件键；冻结的 v1 清单不因此改写。

### 推理与指标

使用 HM-ASR canonical 环境和生成的 `datasets.yaml` 对两模型各执行一次 `run`。
保持 BF16、greedy、相同 batch size / max_new_tokens / 语言设置与 scorer。
两次推理必须使用完全相同的清单、音频和非模型指纹；并发推理只用于质量评估，
不能用于比较推理速度。完成后调用 HM-ASR `compare`，再用同一 canonical 环境：

```bash
python tools/score_controlled_robust_eval.py \
  --output_dir /absolute/path/robustness_v1
```

输出文件：

| 文件 | 内容 |
|---|---|
| `robustness_metrics.json/csv` | 80 组绝对 WER/CER、相对 clean 的额外退化、模型收益 |
| `noise_family_metrics.json/csv` | 336 组噪声来源结果，clean 使用同一语句子集 |
| `noise_curve_auc.json/csv` | 8 条 0/10/20 dB 误差曲线的归一化面积 |
| `robustness_summary.json` | 指标定义及生成/验证结果 |
| `robustness_report.md` | dev/test 分开的可读结果表 |

设错误率为 E，分别报告：

- 绝对错误率：E(增强条件)，越低越好；英文用 WER，中文/粤语用 CER。 标准化后为空的参考文本由官方 scorer 跳过；输出同时记录抽取数和有效计分数。
- 额外退化：E(增强条件) − E(clean)，以百分点计。
- 退化优势：基座的额外退化 − LoRA 的额外退化，正值表示 LoRA 增幅更小。
- 相对错误率降低 RER：(基座 E − LoRA E) / 基座 E × 100%，正值表示 LoRA 更好。
- 噪声曲线面积：梯形积分除以 20 dB，即 (E20 + 2×E10 + E0) / 4，越低越好。

绝对错误率与额外退化须一起阅读：clean 已变差的模型，即使额外退化小，
增强后的绝对识别质量也可能更差。WER/CER 不混成一个总分，dev/test 分别报告。
每组只有 200 句、每噪声来源约 33–34 句；这些是固定子集上的点估计。
实测 RIR 的 dev/test 各只有两个保守来源组，不能由此泛化到所有真实房间。
当前 test 仅用于最终评估；后续选择参数或检查点应使用 dev，并保留 test 的独立性。
各噪声族采用各自固定语音子集，不能只按 family 的绝对错误率对噪声难度排序。
本轮 KeSpeech 没有做完整的方言 × 噪声族交叉实验；详细计数见 `coverage_breakdown.json`。
