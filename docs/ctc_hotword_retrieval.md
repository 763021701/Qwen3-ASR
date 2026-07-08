# CTC 热词检索 → LLM 上下文偏置

用 CTC 粗识检索热词,注入 LLM system prompt 做上下文偏置。**一次 audio_tower + 一次 LLM 推理**,长音频 pre-proj 切片送 CTC decoder 避开 O(N²) 显存。

## 流程

```
audio ──processor(text,audio,truncation=False)──► ONE audio_tower(return_ctc_hidden=True)
                                                    │
                          ┌─────────────────────────┴────────────────────────┐
                          ▼ pre-proj (hidden_states[0])                       ▼ post-proj (last_hidden_state)
                   CTC decoder(整段 or ≤25s 切片)                        LLM generate(一次)
                          │ log-probs                                    ▲ audio_features(预计算)
                          ▼                                              │
                   ctc_rag_hw 检索 ──► 热词 ──► context ──────────────────┘
```

## 特性

- **一次 audio_tower**:整段音频一次前向,pre-proj 喂 CTC、post-proj 喂 LLM,不重复编码。
- **一次 LLM 推理**:整段音频一次 generate,避免分块导致的多次 `language` 标记和"热词 context 压过短 chunk 音频"的 echo 问题。
- **CTC 自适应切片**:音频 ≤ `--ctc_slice_threshold_sec`(默认 60s)整段送 CTC decoder;超长则 pre-proj 切 ≤25s 段、分段解码、拼接 log-probs(避开 CTC decoder 全注意力 O(N²) 显存,600s 实测整段 OOM、切片 0.13s 通过)。
- **encoder 无关检索**:复用 `ctc_rag_hw` 的 `CTCRagRetrieverFromLogProbs`(只需 `blank_id` + `ctc_tokenizer.decode`),不加载 Fun-ASR-Nano encoder。qwen3-asr CTC 与 Fun-ASR-Nano 共用 SenseVoiceTokenizer/vocab/blank_id,检索算法直接生效。
- **`--compare`**:同时跑无热词 baseline,对比偏置效果。

## 前置准备

```bash
pip install pypinyin
pip install -e /path/to/ctc_rag_hw --no-deps   # 避免 pyproject 里 funasr 依赖解析冲突
```
- CTC checkpoint(含 CTC 权重,如 `outputs/curriculum_ctc/stage4/checkpoint-8128`;基模型无 CTC 权重会报错)。
- 热词文件:每行一个热词,`#` 开头注释。

## 用法

```bash
PYTHONPATH=. python scripts/infer_qwen3_asr_ctc_hotword.py \
  --checkpoint outputs/curriculum_ctc/stage4/checkpoint-8128 \
  --audio audio.wav \
  --hotwords hotwords.txt \
  --language Chinese --compare
```

输出:`[ctc mode]`(整段/切片)、`[time]`(各阶段耗时)、`[ctc greedy]`、`[hotwords]`、`[context]`、`[biased]`、`[baseline]`(若 --compare)。

## 关键参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--ctc_slice_threshold_sec` | 60 | > 此值 pre-proj 切片送 CTC(避 O(N²) 显存)。GPU 显存充足可调到 ~120-180s。 |
| `--ctc_slice_sec` | 25 | 切片时每段长度。 |
| `--context_format` | space | space(默认,Qwen3-ASR-main 同款)/comma/structured/nano_style。整段音频下 `space` 无 echo。 |
| `--max_hotwords` | 32 | 注入 LLM 的热词数上限。 |
| `--ctc_topk` | 30 | CTC lattice 每帧 top-k。 |
| `--language` | ""(自动) | 强制语言可避免输出带 `language X<asr_text>` 前缀。 |
| `--max_new_tokens` | 1024 | 长音频(几分钟)建议 2048+ 避免截断。 |

## 注意事项

1. **必须 `PYTHONPATH=.`**(否则 `from qwen_asr...` 导入失败)。
2. **长音频必须用 `processor(text, audio, truncation=False)`**。直接调 `processor.feature_extractor(...)` 会被 WhisperFeatureExtractor 默认截断到 30s(`n_samples=480000`,`truncation=True`)——这是之前误诊的"~25s CTC 天花板"的根因。processor 的 `__call__` 内部已设 `truncation=False`,所以 `processor(text, audio)` 不截断。
3. **CTC decoder 是全注意力 O(N²)**:常规长度(≤几分钟)整段 OK;超长(≥10min)整段会 OOM,走 pre-proj 切片。切片不损失检索质量(整段 vs 切片热词几乎一致)。
4. **echo 现象**(已消除):早期 per-chunk 方案里,短 chunk 音频 + bare `space` 热词 context 会让 LLM 把热词表当前缀复述。当前整段方案下(audio embeddings 量大、context 不再压过音频)`space` 格式无 echo,已设为默认;`nano_style` 是备选(把热词包进句子)。
5. **偏置收益**:仅在 LLM 本身会出错的词(罕见专名、医学术语、领域行话)上显现;常见词 LLM 本就识别对,偏置无变化。
6. **热词 context 进 system prompt**(Qwen3-ASR-main `format_hotword_context` 同款约定)。

## 相关脚本

- `scripts/infer_qwen3_asr_ctc.py` — 最简 CTC 推理。⚠️ 直接调 `processor.feature_extractor`,**仅适 ≤30s**。
- `scripts/infer_qwen3_asr_ctc_chunked.py` — 纯 CTC(fixed/silence 切分),不含 LLM。
- `scripts/infer_qwen3_asr_shared_forward.py` — 共享前向验证(证明 audio_tower 只跑一次)。
- `scripts/infer_qwen3_asr.py` — 标准 AR 推理(无 CTC),`--mode {full,chunk}`。
