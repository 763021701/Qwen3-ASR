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
