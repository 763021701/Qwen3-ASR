---
name: qwen3-asr-low-resource-finetune
description: End-to-end Qwen3-ASR fine-tuning for new corpora or low-resource languages in this repo—jsonl manifest contract, conversion scripts, tokenizer checks, training and eval. Use when the user adds a new speech dataset, adapts a new language, asks for ASR SFT data prep, or mentions Qwen3-ASR jsonl / prepare_* / verify_tokenizer.
---

# Qwen3-ASR 小语种与新语料微调

## 目标

在新数据集或新小语种任务出现时，按本 skill **独立完成**：原始数据探查 → 写出/选择 pipeline 配置 → 生成训练/验证/测试 jsonl → 校验 manifest → 启动 `finetuning/qwen3_asr_sft.py` → 用现有 eval 脚本跑评测。

优先使用统一入口：

```bash
python tools/qwen3_asr_pipeline.py --config CONFIG.yaml --stage all
```

先用 dry-run 检查命令：

```bash
python tools/qwen3_asr_pipeline.py --config CONFIG.yaml --stage all --dry_run 1
```

## 数据契约（必须遵守）

训练/验证文件为 **JSONL**，每行一个对象，字段：

| 字段 | 必填 | 说明 |
|------|------|------|
| `audio` | 是 | 本地可读的音频路径（建议绝对路径）；训练时由 collator 用 librosa 按 `--sr` 重采样（默认 **16000**） |
| `text` | 是 | **完整**监督串，格式：`language {LanguageName}<asr_text>{转写正文}`，无额外换行破坏该模式 |
| `prompt` | 否 | 若存在，会进入 chat prefix；多数 ASR SFT 可省略 |

`text` 中的 `{LanguageName}` 必须与仓库内 **`qwen_asr/inference/utils.py` 中 `SUPPORTED_LANGUAGES`** 的某项 **完全一致**（含大小写规则：首字母大写、其余小写可用 `normalize_language_name`）。若语料语言 **不在列表中**：先与用户确认是否用最近邻已有语言名，或是否要在该文件中 **扩展列表** 并同步检查推理/评测脚本。

参考实现（评估集转换时的前缀拼接）：

- `evaluation/chinese/wsc/prepare_wsc_eval_qwen3.py`（WSC-Eval / 四川话；`_TEXT_PREFIX`）
- `evaluation/cantonese/wsyue_asr/prepare_wsyue_asr_eval_qwen3.py`

仓库内已生成的 jsonl 示例：`data/uyghur/common_voice/ug_train_qwen3.jsonl`。

## 工作流清单（按顺序执行）

```
- [ ] 1. 阅读原始数据：目录结构、元数据格式（tsv/csv/json/JSONL/Kaldi scp+text）、音频扩展名与路径列名
- [ ] 2. 选定 `SUPPORTED_LANGUAGES` 中的语言标签；与用户确认不在列表时的策略
- [ ] 3. 编写或更新 `configs/...yaml`，用 `tools/qwen3_asr_pipeline.py --stage prepare` 输出 train/dev/test jsonl
- [ ] 4. 校验：运行 `tools/qwen3_asr_pipeline.py --stage validate`，确认 `audio` 文件存在；`text` 均含 `language ` 与 `<asr_text>`
- [ ] 5. Tokenizer：对 **最终 `text` 串**（或至少 `<asr_text>` 后正文）抽样 encode，检查 UNK 与 decode 回退（可参考 `evaluation/tools/verify_tokenizer_cv_ug.py` 的逻辑，按新语料改输入源）
- [ ] 6. 训练：`python tools/qwen3_asr_pipeline.py --config CONFIG.yaml --stage train`
- [ ] 7. 评测：`python tools/qwen3_asr_pipeline.py --config CONFIG.yaml --stage eval`，默认选择训练输出目录下 step 最大的 `checkpoint-*`
```

## 新数据集：转换脚本写法

当没有现成 `prepare_*` 覆盖该语料时，**由 agent 新建脚本**（建议路径 `evaluation/<language>/<dataset>/prepare_<dataset>_qwen3.py` 或 `data/scripts/prepare_<dataset>_qwen3.py`），要求：

1. 使用 `argparse`，参数至少包含：`--output_jsonl`、`--dataset_dir`（或等价根路径）、`--language`、`--max_samples`（0 表示全量）。
2. 逐行写出 JSON：`json.dumps({"audio": abs_path, "text": supervised}, ensure_ascii=False)`，文件 **UTF-8**。
3. `supervised` 用本 skill 的 `text` 格式；可用仓库内小工具拼字符串（见 `scripts/format_label.py`）。
4. 路径统一为 `os.path.abspath` 或 pathlib，避免训练机 cwd 不一致导致读音频失败。
5. 对缺失音频、空转写行做过滤或计数日志，避免静默产生坏样本。

不必修改 `finetuning/qwen3_asr_sft.py` 的语种分支——**语种信息只应出现在数据的 `text` 里**；仅当需要 LoRA/多卡等通用能力时再改训练脚本。

## 训练入口速查

- 脚本：`finetuning/qwen3_asr_sft.py`
- 数据：`load_dataset("json", data_files=...)`，必需 `audio` + `text`
- 常用参数：`--model_path`、`--train_file`、`--eval_file`、`--output_dir`、`--sr`、`--batch_size`、`--grad_acc`、`--lr`、`--epochs`、`--save_steps`、`--resume` / `--resume_from`

## 附加材料

- 更细的排错与语言列表说明：[reference.md](reference.md)
- 生成单条 `text` 字段的 CLI 辅助：[scripts/format_label.py](scripts/format_label.py)  
  从仓库根目录运行示例：  
  `python .cursor/skills/qwen3-asr-low-resource-finetune/scripts/format_label.py --language Uyghur --transcript "…"`
