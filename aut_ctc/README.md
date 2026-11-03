# AuT-CTC

冻结 Qwen3-ASR-1.7B 的音频编码器（AuT），在其 `ln_post` 输出上训练 Conformer-lite CTC Decoder（中、英、粤、中文方言）。设计见 `references/aut_ctc_design/DESIGN.md`（v1.1）。本目录与 `ctc` 分支上照搬 Fun-ASR-Nano 的 CTC 实现互不依赖。

## 环境

- conda 环境 `whisperlivekit`（transformers 4.57.6）。该环境装的是 PyPI 版 qwen-asr 0.0.6，**不含**下面的 AuT 窗口修复，所以一律在仓库根目录用 `python -m aut_ctc.…` 运行（cwd 优先于 site-packages）；`frontend.py` 发现导入的不是仓库里的 `qwen_asr` 会直接报错。也可以 `pip install -e .`，但会改变同一环境里 WhisperLiveKit 用到的 qwen_asr。环境里已装 opencc 1.4.2（繁转简）和 num2words 0.5.14（英文 TN）。
- `qwen_asr/core/transformers_backend/modeling_qwen3_asr.py`：AuT 的块对角窗口原先只在 flash-attn（`cu_seqlens`）下生效，sdpa/eager 实际是整段全注意力、打包 batch 会串扰；现在显式传入 `_prepare_attention_mask` 的窗口 mask。这也会让本仓库 transformers 后端处理 >8 s 音频时的 LLM 输出向官方（vLLM/FA2）行为靠拢。

## 用法

```bash
# 测试（CPU；--real 额外用真实 AuT 权重）
python -m aut_ctc.tests.test_frontend --real
python -m aut_ctc.tests.test_tokenizer --sp_model /root/autodl-tmp/models/SenseVoiceSmall/chn_jpn_yue_eng_ko_spectok.bpe.model --texts texts.txt
python -m aut_ctc.tests.test_model
python -m aut_ctc.tests.test_streaming
# M2 数据报告（与训练同一套过滤逻辑），可选写出过滤后的清单；top_oov_chars 是 extra_chars.txt 的补字候选
python -m aut_ctc.tools.data_stats --sp_model /root/autodl-tmp/models/SenseVoiceSmall/chn_jpn_yue_eng_ko_spectok.bpe.model \
    --manifests data/train_zh.jsonl data/train_en.jsonl --report exp/data_report.json
# M3 层探针
python -m aut_ctc.tools.probe_layers --config aut_ctc/configs/base.yaml aut_ctc/configs/probe.yaml --exp_dir exp/probe
# M4 离线训练（多卡：torchrun --nproc_per_node N -m aut_ctc.train ...）；exp_dir 里有 last.pt 时自动续训
# 共享 GPU（与 6006/6008 服务同卡）：每个 micro-batch 400 s、累计 3 步 = 设计默认的每步 1200 s。
# 实测（V=19911，现在的 19945 只多 0.2%）：分配峰值 5.8 GB，进程占用约 8.2 GB，给同卡服务留约 3.9 GB；超出上限只会让本进程 OOM
python -m aut_ctc.train --config aut_ctc/configs/base.yaml --exp_dir exp/base \
    --set data.max_batch_seconds=400 optim.grad_accum=3 optim.gpu_memory_limit_gb=8
# M5 动态窗口流式训练（stream.yaml 叠加在 base.yaml 上，init_from 指向 M4 最好的检查点）
python -m aut_ctc.train --config aut_ctc/configs/base.yaml aut_ctc/configs/stream.yaml --exp_dir exp/stream
# 评测：离线；或按流式方式（AuT 窗口 W + 块因果掩码）
python -m aut_ctc.evaluate --exp_dir exp/base --manifest data/dev_zh.jsonl --out exp/base/hyps.jsonl
python -m aut_ctc.evaluate --exp_dir exp/stream --manifest data/dev_zh.jsonl --window 200 --left_chunks 4
```

流式识别：`StreamingRecognizer(AuTFrontend(...), decoder, tokenizer, window=200, left_chunks=4)`，`accept_waveform(pcm)` 返回 `{"final", "partial"}`，结束时调用 `finalize()`。

## 数据（ModelScope 上的 WebDataset 分片）

数据集 `huangpppu/{Gigaspeech-train-clean, WenetSpeech, KeSpeech, WenetSpeech-Chuan, WenetSpeech-Yue}` 都是 WebDataset（每条 `<key>.flac` + `<key>.json`/`.txt`，FLAC 16 kHz，约 60 MiB/小时）。

- 训练直接读本地分片里的成员，不解包：清单条目 `audio` 指向 tar，`tar_offset`/`tar_size` 是 FLAC 成员的字节范围。
- 下载：`python -m aut_ctc.tools.ms_download list|get`（用本机已登录的 ModelScope 账号；并行、按 sha256 校验、失败自动续传）。实测每条连接约 1 MiB/s、总带宽约 8–9 MiB/s，而训练消耗约 35 MiB/s（约 2,100 倍实时），所以先下载到本地再训练，不做在线 streaming。
- 清单：`python -m aut_ctc.tools.wds_manifest`（按小时随机抽样、`--where` 过滤 json 字段、`--min_confidence` + `--confidence_tsv` 按 ROVER 置信度过滤；时长从 FLAC 头读取）。
- KeSpeech 按 `style` 区分普通话（`Mandarin`/`Accent`）和 8 种次方言（`Dialect`）；`phase` 只是采集批次。
- WenetSpeech-Chuan / Yue 的标注是伪标签，分片里没有置信度，从本地元数据导出 `key\tconfidence` 表后过滤：训练取 ≥ 0.7（Chuan 留约 64–69% 时长，Yue 约 97%），开发集取 ≥ 0.9。
- 开发集来自不参与训练的分片（WenetSpeech 官方 DEV、KeSpeech 官方 dev，GigaSpeech/Chuan/Yue 单独的分片），且与开发集分片同一来源包/相邻的分片都不进训练。
- 配比与构建脚本在 `/root/autodl-tmp/huangpu/aut_ctc_data/`（`build_mix200h.py`、`build_mix3000h.py`），训练配置是 `configs/mix200h.yaml`、`configs/mix3000h.yaml`。

## 词表

SenseVoiceSmall 词表裁成中英紧凑子集，再补回方言相关的部分，V = 19,945：

- 原设计的 19,911（blank、unk、汉字 10,124、拉丁子词 9,674、标点 69、数字 10、标签 32），加上 4 个方言/粤语标签 `<|yue|> <|wuyu|> <|minnan|> <|dialect|>`（二期 query 用）、孤立 `▁`、`extra_chars.txt` 里的 29 个方言字。
- 方言字：SenseVoice 和 Fun-ASR-Nano 的 CTC 词表汉字完全相同（10,124 个），Fun-ASR-Nano 对缺的字用 UTF-8 字节回退；这里改为每个字一个 id，候选来自本地语料的缺字统计和吴/粤/闽/客家常用字。补字后含 `<unk>` 的句子：上海话 9.0%→0，粤语 0.66%→0.0003%，四川话 0.007%→0.003%。拿到训练数据后用 `data_stats` 的 `top_oov_chars` 再补，**开训后不要再改**（会改变输出层）。
- 编码：文本前加 `▁`，句首英文词与句中切分一致（否则 speech → s/pe/e/ch）；孤立 `▁` 只保留在两个英文/数字 piece 之间（qwen、russian 等词没有 ▁ 开头的 piece，丢掉会解码成 "andqwen"），挨着汉字或在两端时丢弃。
- 规范化：带重音的拉丁字母折叠（é→e、ß→ss，原词表没有）；CJK 范围包括扩展 B 及以后（𠮶、𠊎）。

## 与设计文档的差异

| 项 | 设计 | 本实现 |
|---|---|---|
| AuT 入口 | transformers ≥5.22 原生 + `Qwen3-ASR-1.7B-hf` | 仓库内 qwen-asr 的 AuT（本地 `Qwen3-ASR-1.7B` 权重，只读 `thinker.audio_tower.*`）：打包 batch = 拼接 mel + `feature_lens`；`ln_post` 和中间层用 forward hook 取 |
| 特征提取 | `Qwen3ASRFeatureExtractor` | checkpoint 自带的 `WhisperFeatureExtractor`，`padding="longest"`、不截断；<0.5 s 补零到 0.5 s；mel 不必补到 100 的倍数 |
| 短 batch | — | batch 里没有满 1 s 的块时补一条 1 s 静音哑元（独立窗口），使尾块卷积补零与训练一致 |
| 离线长音频 | >60 s 才用带状注意力 | 离线解码始终传 `local_frames=390`：≤30 s 时与全注意力完全相同，30–60 s 也不会遇到训练外的 RoPE 距离 |
| TN | 中文 WeTextProcessing、英文 NeMo | 含汉字或 lang 非英文（zh/yue/wuyu…）用 `cn_tn.py`（从 `ctc` 分支复制）；英文用 num2words（`en_tn.py`：金额、百分比、序数、年代、小数、千分位、年份、前导零、词内数字，英式 "and"；"911" 这类依语境的读法和单位缩写不处理）；数字间有斜杠（分数、日期）的样本丢弃并计数 |
| 繁转简 | OpenCC t2s，统一做 | 按条启用：清单里写 `"t2s": "hk2s"`（或 `t2s`/`tw2s`），只用于繁体来源。对简体文本 OpenCC 不是空操作（个别短语里 坏→坯、沈→沉）；嗰喎餸噚 保持原样（OpenCC 会转成 𠮶㖞𩠌㖊，后三个不在词表） |
| `sampling_weights` | 按小时数加权 | 每步按权重选清单，等权 = 各清单步数相同 |
| 词表 | V=19,911，丢弃所有孤立 `▁` | 见上面“词表”一节：+4 个方言标签、+孤立 `▁`（仅在英文词之间）、+29 个方言字，V=19,945 |

## 未实现

特征缓存（`CachedFeatureDataset`、`cache_features.py`、特征时间遮挡、T21）、伪标签工具（需要 vLLM）、流式 v2（KV 缓存）、延迟基准、二/三期（query 数据通路、withitn、束搜索/LM/热词、ONNX）、EMA。

## 测试状态

T1–T20 通过：T7–T10 用真实 SenseVoiceSmall `.bpe.model`（ModelScope `iic/SenseVoiceSmall`，sha256 已核对）；T20 用合成词表在 MagicData 上海话 50 条上过拟合（MER 0）。T21 依赖未实现的特征缓存。
