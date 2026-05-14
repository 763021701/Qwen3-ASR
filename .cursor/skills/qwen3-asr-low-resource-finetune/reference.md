# Qwen3-ASR 小语种微调 — 参考与排错

## 语言列表来源

权威列表：`qwen_asr/inference/utils.py` 中的 `SUPPORTED_LANGUAGES`。

推理侧 `validate_language` 会拒绝不在列表中的名称。训练数据中的 `language {Name}` 应与该列表一致，避免与预训练/推理约定冲突。

## 微调脚本与语料解耦

`finetuning/qwen3_asr_sft.py` 将每条样本的整条 `text` 作为 `target`（在 chat prefix 之后预测）。因此 **不要** 在 jsonl 里只存裸转写而不带 `language …<asr_text>` 前缀（除非用户明确要改 collator 逻辑，本仓库默认不支持）。

## Tokenizer 验证建议

- `tools/verify_tokenizer_cv_ug.py`：从 TSV 的 `sentence` 列做 round-trip；新语料可仿照其逻辑，改为从生成的 jsonl 读 `text`，或对 strip 掉前缀后的转写做检查。
- 若大量出现 UNK：检查是否应用了与训练一致的 Unicode 规范化；或考虑换更大 checkpoint / 与用户确认是否可接受子词切分。

## 常见问题

| 现象 | 排查 |
|------|------|
| `FileNotFoundError` 音频 | `audio` 是否绝对路径；文件是否在训练节点可见 |
| loss 异常或不学 | 抽样打印 `text` 是否含错误前缀；是否多空格/错误 `language` 名 |
| OOM | 减小 `batch_size`，增大 `grad_acc`；检查超长音频是否需过滤 |
| 评测与训练不一致 | 评测 jsonl 的 `text` 格式须与训练相同；eval 脚本若 strip 前缀，需与 `evaluation/cantonese/eval_cantonese_asr_jsonl.py` 等保持一致 |

## 已有转换脚本（可抄结构）

- `evaluation/chinese/wsc/prepare_wsc_eval_qwen3.py`（WSC / 四川话，非粤语）
- `evaluation/cantonese/wsyue_asr/prepare_wsyue_asr_eval_qwen3.py`

## 评测脚本线索

仓库内按语种/任务有不同 `evaluation/<lang>/eval_*_jsonl.py` 与 `evaluation/<lang>/baselines/`；`evaluation/README.md` 含常用评测调用示例。新语种优先复用与「同推理接口」最接近的 eval 脚本并改 `--language` 或 jsonl 路径。
