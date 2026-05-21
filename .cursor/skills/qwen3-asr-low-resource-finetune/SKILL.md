---
name: qwen3-asr-low-resource-finetune
description: End-to-end Qwen3-ASR fine-tuning for new corpora or low-resource languages in this repo—jsonl manifest contract, conversion scripts (must read docs/normalize_label.md before writing prepare/jsonl scripts), tokenizer checks, optional online audio augmentation, training and eval. Use when the user adds a new speech dataset, adapts a new language, asks for ASR SFT data prep, label normalization, data augmentation, or mentions Qwen3-ASR jsonl / prepare_* / verify_tokenizer / normalize_label.
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

## 启动前主动确认（必须执行）

开始改配置、写转换脚本或启动训练前，若任何参数或可选操作不确定，必须先主动询问用户，而不是替用户静默选择。一次只问关键问题，给出推荐默认值。至少覆盖：

- 数据切分：是否已有 train/dev/test；若需切分，确认比例与 `split_seed`。
- 语言标签：`SUPPORTED_LANGUAGES` 中没有精确标签时，确认近邻标签或是否扩展语言列表。
- 转写规范化：数字、标点、繁简、大小写、混说文本的策略。
- 训练预算：模型、epoch、batch/grad_acc、学习率、checkpoint 保存频率、是否 resume。
- 可选在线增强：是否开启 `training.augment`；若开启，确认 SpeedPerturbation、AddNoise、SpecAugment 的概率和强度。默认建议先不开启；低资源或噪声域偏移明显时再开启。
- 其它可选操作：tokenizer 抽样检查、max_samples smoke test、dry-run、评测脚本/指标、是否只跑 prepare/validate/train/eval 某个阶段。

若用户没有偏好，明确说明将使用保守默认值，并在执行前展示会写入 config 或命令的关键参数。

## 数据契约（必须遵守）

训练/验证文件为 **JSONL**，每行一个对象，字段：

| 字段 | 必填 | 说明 |
|------|------|------|
| `audio` | 是 | 本地可读的音频路径（建议绝对路径）；训练时由 collator 用 librosa 按 `--sr` 重采样（默认 **16000**） |
| `text` | 是 | **完整**监督串，格式：`language {LanguageSpec}<asr_text>{转写正文}`，无额外换行破坏该模式；`{LanguageSpec}` 为 **单个** `SUPPORTED_LANGUAGES` 中的语言名，或 **逗号分隔** 的多语混说标签（如 `Chinese,English`），逗号两侧空格可有可无 |
| `prompt` | 否 | 若存在，会进入 chat prefix；多数 ASR SFT 可省略 |

`text` 中的 `{LanguageSpec}`：每一段（逗号分隔后的原子）必须落在 **`qwen_asr/inference/utils.py` 中 `SUPPORTED_LANGUAGES`**；组合名（如 `Chinese,English`）**不会**作为一整条写进该列表。`normalize_language_spec` 会把**多个**原子语言按 `SUPPORTED_LANGUAGES` **文件内顺序**重排为统一写法（例如 `English,Chinese` 与 `Chinese,English` 等价，均规范为 `Chinese,English`）。规范化与校验请用 `normalize_language_spec` / `validate_language_spec`（单语仍可用 `normalize_language_name` / `validate_language`）。若语料语言 **不在列表中**：先与用户确认是否用最近邻已有语言名，或是否要在该文件中 **扩展列表** 并同步检查推理/评测脚本。

参考实现（评估集转换时的前缀拼接）：

- `evaluation/chinese/wsc/prepare_wsc_eval_qwen3.py`（WSC-Eval / 四川话；`_TEXT_PREFIX`）
- `evaluation/cantonese/wsyue_asr/prepare_wsyue_asr_eval_qwen3.py`

仓库内已生成的 jsonl 示例：`data/uyghur/common_voice/ug_train_qwen3.jsonl`。

## 标签正文规范化（写 jsonl 脚本前必读）

**在编写或扩展任何生成训练/验证/测试 jsonl 的脚本之前**，必须先完整阅读仓库根目录手册：

- **[docs/normalize_label.md](../../../docs/normalize_label.md)**（相对本 skill：`../../../docs/normalize_label.md`）

该文件是转写正文（`<asr_text>` **之后**的字符串）规范化的权威说明。`normalize_target_text` / `format_label.py` **只**拼 `language …<asr_text>` 前缀，**不会**自动去标点或按语种清洗正文；须在拼前缀 **之前** 对手册适用语种执行 normalize。

| `SUPPORTED_LANGUAGES` 标签（示例） | 手册章节 |
|-----------------------------------|----------|
| `English` | ENGLISH |
| `Chinese` | MANDARIN |
| `Cantonese` | CANTONESE |
| `Chinese,English` 等混说 | 按句内/词级分别套用对应章节，或与用户确认统一策略 |
| 手册未列语种（如 `Uyghur`） | 与用户确认是否沿用 GLOBAL RULES + 邻近语种规则，或单独约定 |

**GLOBAL RULES**（手册）：train/dev/test 同一套规则；Unicode 规范化；去掉不可见/控制字符与非语音标注；默认去标点（除非用户明确要求保留标点预测）；空白规范化；normalize 后为空则丢弃样本；数字策略全库一致。

实现方式（任选，须在脚本或共享模块中写清）：

1. 在 `tools/convert_to_qwen3_asr_jsonl.py` 增加可开关的 `--normalize_transcript` / `--label_locale`，或
2. 新建 `tools/normalize_transcript.py`（或 skill 内 `scripts/normalize_transcript.py`）供各 `prepare_*` 调用，或
3. 在专用 `tools/prepare_*.py` 内内联实现，但逻辑须与手册一致并在 `--report_json` 中统计 `skipped_empty_after_normalize`。

拼完规范化正文后再调用 `qwen3_asr_supervised_text`（见 [scripts/format_label.py](scripts/format_label.py)）生成最终 `text` 字段。

## 工作流清单（按顺序执行）

```
- [ ] 0. 主动询问并确认所有不确定参数与可选操作，尤其是切分、normalize、训练预算、是否开启在线增强
- [ ] 1. 阅读原始数据：目录结构、元数据格式、音频路径与扩展名/容器格式；训练阶段由 collator 读文件，**非常见格式或读盘报错**时对照 [reference.md](reference.md) 排查环境与依赖
- [ ] 2. 选定 `SUPPORTED_LANGUAGES` 中的语言标签；与用户确认不在列表时的策略
- [ ] 2b. **阅读 [docs/normalize_label.md](../../../docs/normalize_label.md)**，确定本语料的转写 normalize 策略（含数字、繁简、是否保留标点）
- [ ] 3. 编写或更新 `configs/...yaml` 与转换脚本；`prepare` 阶段对转写按手册 normalize 后再写 jsonl；如用户确认开启增强，在 `training:` 中写入增强参数
- [ ] 4. 校验：运行 `tools/qwen3_asr_pipeline.py --stage validate`，确认 `audio` 文件存在；`text` 均含 `language ` 与 `<asr_text>`
- [ ] 5. Tokenizer：对 **最终 `text` 串**（或至少 `<asr_text>` 后正文）抽样 encode，检查 UNK 与 decode 回退（可参考 `tools/verify_tokenizer_cv_ug.py` 的逻辑，按新语料改输入源）
- [ ] 6. 训练：`python tools/qwen3_asr_pipeline.py --config CONFIG.yaml --stage train`
- [ ] 7. 评测：`python tools/qwen3_asr_pipeline.py --config CONFIG.yaml --stage eval`（默认选用输出目录下 step 最大的 `checkpoint-*`）；失败时查 [reference.md](reference.md)
```

## 新数据集：转换脚本写法

训练/微调数据转换优先复用或扩展顶层 `tools/convert_to_qwen3_asr_jsonl.py`（由 `tools/qwen3_asr_pipeline.py --stage prepare` 调用）；**具体 `dataset.source_type` 与参数以仓库内 `tools/qwen3_asr_pipeline.py` / `convert_to_qwen3_asr_jsonl.py` 为准**，新语料格式在二者之一中实现并在 yaml 里接通 `prepare` 即可。

当新语料无法用通用转换器表达时，**由 agent 扩展 `tools/convert_to_qwen3_asr_jsonl.py` 或在 `tools/` 下新建通用转换脚本**；只有评测集专用、带下载/子集/去重等 benchmark 规则的 prepare 脚本，才放在 `evaluation/<language>/<dataset>/prepare_<dataset>_qwen3.py`。要求：

0. **先读** [docs/normalize_label.md](../../../docs/normalize_label.md)，再写代码；不得默认「原样 strip + 拼前缀」即足够。
1. 使用 `argparse`，参数至少包含：`--output_jsonl`、`--dataset_dir`（或等价根路径）、`--language`、`--max_samples`（0 表示全量）；若语种在手册中有专章，增加与手册一致的选项（如 `--number_policy`、`--hanzi_script`）。
2. 逐行写出 JSON：`json.dumps({"audio": abs_path, "text": supervised}, ensure_ascii=False)`，文件 **UTF-8**。
3. 流程：`raw_transcript` → **按手册 normalize** → `qwen3_asr_supervised_text(language, normalized)`（见 [scripts/format_label.py](scripts/format_label.py)）；`supervised` 即最终 `text`。
4. 路径统一为 `os.path.abspath` 或 pathlib，避免训练机 cwd 不一致导致读音频失败。
5. 对缺失音频、normalize 前空转写、normalize 后空串做过滤或计数日志（`skipped_empty_after_normalize` 等），避免静默产生坏样本。
6. train / dev / test 使用**同一套** normalize 实现（与手册 GLOBAL RULES 一致）。

不必修改 `finetuning/qwen3_asr_sft.py` 的语种分支——**语种信息只应出现在数据的 `text` 里**；仅当需要 LoRA/多卡等通用能力时再改训练脚本。

## 训练入口速查

- 脚本：`finetuning/qwen3_asr_sft.py`
- 数据：`load_dataset("json", data_files=...)`，必需 `audio` + `text`
- 常用参数：`--model_path`、`--train_file`、`--eval_file`、`--output_dir`、`--sr`、`--batch_size`、`--grad_acc`、`--lr`、`--epochs`、`--save_steps`、`--resume` / `--resume_from`
- 在线增强（默认关闭，只用于 train collator，不用于 eval）：`--augment 1` 开启；常用参数 `--augment_prob`、`--speed_prob`、`--speed_factors`、`--noise_prob`、`--noise_snr_min`、`--noise_snr_max`、`--specaug_prob`、`--specaug_time_mask_param`、`--specaug_freq_mask_param`、`--specaug_num_time_masks`、`--specaug_num_freq_masks`。pipeline YAML 中写在 `training:` 下会透传给训练脚本。

增强建议：

- 默认先用 `augment: 0` 建 baseline。
- 低资源、说话人/录音条件覆盖不足时，可先尝试 `augment: 1`、`speed_factors: "0.9,1.0,1.1"`、中等 `noise_prob`、较低到中等 `specaug_prob`。
- AddNoise 当前为合成白噪声；若用户需要真实背景噪声，先确认噪声数据路径、采样率、授权和混合策略后再扩展。

## 附加材料

- 转写正文规范化手册（**写 jsonl 前必读**）：[docs/normalize_label.md](../../../docs/normalize_label.md)；细则摘要见 [reference.md](reference.md)「标签规范化」。
- 排错、环境与 checkpoint 评测加载等：**只维护在** [reference.md](reference.md)，避免与主 skill 重复。
- 生成单条 `text` 字段的 CLI 辅助：[scripts/format_label.py](scripts/format_label.py)（**不**含正文 normalize，normalize 后再传入 `--transcript`）
  从仓库根目录运行示例：  
  `python .cursor/skills/qwen3-asr-low-resource-finetune/scripts/format_label.py --language English --transcript "…"`
