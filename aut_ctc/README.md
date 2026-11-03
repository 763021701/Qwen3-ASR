# AuT-CTC

冻结 Qwen3-ASR-1.7B 的音频编码器（AuT），在其 `ln_post` 输出上训练 Conformer-lite CTC Decoder（中英）。设计见 `references/aut_ctc_design/DESIGN.md`（v1.1）。本目录与 `ctc` 分支上照搬 Fun-ASR-Nano 的 CTC 实现互不依赖。

## 环境

- conda 环境 `whisperlivekit`（transformers 4.57.6）。该环境装的是 PyPI 版 qwen-asr 0.0.6，**不含**下面的 AuT 窗口修复，所以一律在仓库根目录用 `python -m aut_ctc.…` 运行（cwd 优先于 site-packages）；`frontend.py` 发现导入的不是仓库里的 `qwen_asr` 会直接报错。也可以 `pip install -e .`，但会改变同一环境里 WhisperLiveKit 用到的 qwen_asr。
- `qwen_asr/core/transformers_backend/modeling_qwen3_asr.py`：AuT 的块对角窗口原先只在 flash-attn（`cu_seqlens`）下生效，sdpa/eager 实际是整段全注意力、打包 batch 会串扰；现在显式传入 `_prepare_attention_mask` 的窗口 mask。这也会让本仓库 transformers 后端处理 >8 s 音频时的 LLM 输出向官方（vLLM/FA2）行为靠拢。

## 用法

```bash
# 测试（CPU；--real 额外用真实 AuT 权重）
python -m aut_ctc.tests.test_frontend --real
python -m aut_ctc.tests.test_tokenizer --sp_model /path/to/chn_jpn_yue_eng_ko_spectok.bpe.model --texts texts.txt
python -m aut_ctc.tests.test_model
python -m aut_ctc.tests.test_streaming
# M2 数据报告（与训练同一套过滤逻辑），可选写出过滤后的清单
python -m aut_ctc.tools.data_stats --sp_model ... --manifests data/train_zh.jsonl data/train_en.jsonl --report exp/data_report.json
# M3 层探针
python -m aut_ctc.tools.probe_layers --config aut_ctc/configs/base.yaml aut_ctc/configs/probe.yaml --exp_dir exp/probe
# M4 离线训练（多卡：torchrun --nproc_per_node N -m aut_ctc.train ...）；exp_dir 里有 last.pt 时自动续训
python -m aut_ctc.train --config aut_ctc/configs/base.yaml --exp_dir exp/base --set data.max_batch_seconds=400
# M5 动态窗口流式训练（stream.yaml 叠加在 base.yaml 上，init_from 指向 M4 最好的检查点）
python -m aut_ctc.train --config aut_ctc/configs/base.yaml aut_ctc/configs/stream.yaml --exp_dir exp/stream
# 评测：离线；或按流式方式（AuT 窗口 W + 块因果掩码）
python -m aut_ctc.evaluate --exp_dir exp/base --manifest data/dev_zh.jsonl --out exp/base/hyps.jsonl
python -m aut_ctc.evaluate --exp_dir exp/stream --manifest data/dev_zh.jsonl --window 200 --left_chunks 4
```

流式识别：`StreamingRecognizer(AuTFrontend(...), decoder, tokenizer, window=200, left_chunks=4)`，`accept_waveform(pcm)` 返回 `{"final", "partial"}`，结束时调用 `finalize()`。

## 与设计文档的差异

| 项 | 设计 | 本实现 |
|---|---|---|
| AuT 入口 | transformers ≥5.22 原生 + `Qwen3-ASR-1.7B-hf` | 仓库内 qwen-asr 的 AuT（本地 `Qwen3-ASR-1.7B` 权重，只读 `thinker.audio_tower.*`）：打包 batch = 拼接 mel + `feature_lens`；`ln_post` 和中间层用 forward hook 取 |
| 特征提取 | `Qwen3ASRFeatureExtractor` | checkpoint 自带的 `WhisperFeatureExtractor`，`padding="longest"`、不截断；<0.5 s 补零到 0.5 s；mel 不必补到 100 的倍数 |
| 短 batch | — | batch 里没有满 1 s 的块时补一条 1 s 静音哑元（独立窗口），使尾块卷积补零与训练一致 |
| 离线长音频 | >60 s 才用带状注意力 | 离线解码始终传 `local_frames=390`：≤30 s 时与全注意力完全相同，30–60 s 也不会遇到训练外的 RoPE 距离 |
| TN | 中文 WeTextProcessing、英文 NeMo | 中文用 `cn_tn.py`（从 `ctc` 分支复制）；英文没有 TN，含数字的英文样本丢弃并计数 |
| 繁转简 | OpenCC t2s | 装了 opencc 才做，否则告警跳过 |
| `sampling_weights` | 按小时数加权 | 每步按权重选清单，等权 = 各清单步数相同 |

## 未实现

特征缓存（`CachedFeatureDataset`、`cache_features.py`、特征时间遮挡、T21）、伪标签工具（需要 vLLM）、流式 v2（KV 缓存）、延迟基准、二/三期（query 数据通路、withitn、束搜索/LM/热词、ONNX）、EMA。

## 测试状态

T1–T8、T11–T20 通过：T7/T8 用合成 SentencePiece 词表，T20 用合成词表在 MagicData 上海话 50 条上过拟合（MER 0）。T9/T10 需要真实 `.bpe.model`，本机没有。
