# Qwen3-ASR 小语种微调 — 参考与排错

## 语言列表来源

权威列表：`qwen_asr/inference/utils.py` 中的 `SUPPORTED_LANGUAGES`。

推理侧 `validate_language` / `validate_language_spec` 会拒绝不在 `SUPPORTED_LANGUAGES` 中的原子语言名。训练数据中的 `language {Name}` 或 `language {Name,Name,...}` 中，**每个**原子名应与该列表一致。多语混说时，`normalize_language_spec` 会按 **`SUPPORTED_LANGUAGES` 在 `utils.py` 中的定义顺序**对多个原子重排，使 `English,Chinese` 与 `Chinese,English` 等写法在推理与校验中一致。`SUPPORTED_LANGUAGES` 仅为原子语言列表，不包含 `Chinese,English` 这类组合字符串。

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
| 训练一开始 DataLoader 报 `audioread.exceptions.NoBackendError` / m4a 打不开 | 多为 **未安装 ffmpeg**。在 conda 环境中执行：`conda install -c conda-forge ffmpeg`（或保证系统 PATH 中有 ffmpeg）。`PySoundFile failed. Trying audioread` 警告在 m4a 上常见，有 ffmpeg 后 audioread 可工作 |
| `Can't load feature extractor` / `preprocessor_config.json` | **`Qwen3ASRModel.from_pretrained(本地 checkpoint)`** 需要与基座一致的 Processor 文件。若 checkpoint 目录里只有 `tokenizer*`、`config.json`、`model.safetensors`，从 **`--model_path` 对应的 HuggingFace 快照或本地基座目录** 复制 **`preprocessor_config.json`**（及若缺的 **`chat_template.json`**）到该 `checkpoint-*` 目录后再跑 eval |
| `Cannot use apply_chat_template because this processor does not have a chat template` | 同上：把基座里的 **`chat_template.json`** 放进 checkpoint 目录 |
| 粤语 eval 中途退出：`Install OpenCC first…` | `pip install opencc-python-reimplemented`；或若脚本支持且可接受，改用 `--hanzi_script_norm off`（以脚本参数为准） |
| 粤语 eval 退出：`Install cn2an first…` | `pip install cn2an`（打分阶段数字归一化依赖） |
| `pip: bad interpreter: No such file or directory` | 该环境 `pip` 脚本的 shebang 损坏；改用 **`python -m pip install 包名`** |

## Checkpoint 与评测加载（Qwen3-ASR）

本仓库 `finetuning/qwen3_asr_sft.py` 通过 Transformers Trainer 保存的 checkpoint **通常只含权重与 tokenizer**，不一定含 **Whisper 系 feature extractor / chat template** 等 Processor 侧文件。`qwen_asr.inference.qwen3_asr.Qwen3ASRModel.from_pretrained(ckpt)` 内部会 `AutoProcessor.from_pretrained(ckpt)`，因此缺文件会在 **eval 或独立推理** 阶段暴露。

**权宜做法（与一次真实跑通一致）**：对每个要评测的 `checkpoint-*`，从训练时使用的 **`--model_path` 基座**（例如 `Qwen/Qwen3-ASR-1.7B` 的 HF 缓存快照目录）复制至少：

- `preprocessor_config.json`
- `chat_template.json`

到该 checkpoint 目录（与 `model.safetensors` 同级）。

**长期改进方向**（可选开发）：在保存 checkpoint 时调用 `processor.save_pretrained(output_dir)` 或与基座做一次文件合并，避免手工拷贝。

## 已有转换脚本（可抄结构）

- `tools/convert_to_qwen3_asr_jsonl.py`（训练/微调数据通用转换器；pipeline prepare 阶段默认调用）
- `evaluation/chinese/wsc/prepare_wsc_eval_qwen3.py`（WSC / 四川话，非粤语）
- `evaluation/cantonese/wsyue_asr/prepare_wsyue_asr_eval_qwen3.py`

## 评测脚本线索

仓库内按语种/任务有不同 `evaluation/<lang>/eval_*_jsonl.py` 与 `evaluation/<lang>/baselines/`；`evaluation/README.md` 含常用评测调用示例。新语种优先复用与「同推理接口」最接近的 eval 脚本并改 `--language` 或 jsonl 路径。
