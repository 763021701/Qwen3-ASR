# Qwen3-ASR GRPO

入口：`finetuning/qwen3_asr_grpo.py`。输入 JSONL 需要 `audio`、`text`，可选
`prompt`；默认冻结 encoder/aligner，只训练 LLM 和输出头。需要 CUDA。

## 默认行为与兼容性

默认 `--advantage_mode group --temperature_schedule constant --temperature 0.8`。
**概率模式默认已改为 `temperature`**：采样及 old/new policy log-prob 都使用
`softmax(logits / T)`，默认 `top_p=1、top_k=0`。显式指定其他截断值会报错。
生成不会继承 checkpoint 的 logits 惩罚设置；old/new 前向均为 eval 模式，
但更新前向保留梯度。

`--policy_probability_mode legacy` 恢复原始 log-prob 打分和默认
`top_p=0.95、top_k=50`，用于概率模式对照；其他 bug 修复仍生效。
OPD、RAFT、数据挖掘调用共享函数时，仍默认获得原始 log-prob 和原有生成行为。

Temperature 模式的 KL 是采样前缀上完整词表的
`KL(policy_raw || reference_raw)`，不是温度分布 KL，也不是完整轨迹 KL。
Reference 使用自己的音频编码；模型须共享词表、特殊 token 和音频预处理配置。
按 32 个 completion 位置分块计算 FP32 概率，并在反向重算中间值；仍需保存
policy/reference logits，实际显存开销需要 CUDA 环境验证。
Legacy 使用原有 sampled KL regularizer。日志 `kl` 与 loss 中未乘 beta 的项一致，
`logp_gap` 单独记录参考与 policy 原始 token log-prob 的平均差。

每个 rollout 只进行一次带梯度前向，不做多轮 PPO，因此 ratio 通常接近 1，
clipping 的实际作用有限。每条序列按有效 token（包含首个 EOS）平均，再按 batch 平均。

## Greedy baseline

`--advantage_mode greedy` 为每条音频额外生成一个确定性 greedy 转写，和 G 条
采样共享 policy 音频编码、prompt、EOS、长度上限及奖励参数。

```text
A_i = reward_i - reward_greedy
```

不除组内标准差、不裁剪优势、不将 greedy 转写加入训练候选。Greedy 模式允许
G=1；group 模式要求 G≥2。零方差组仍可有非零 greedy 优势；优势全零时仍计算 KL。
原有 group 模式继续采用组均值/标准差标准化。

## 温度与更新计数

固定温度使用 `--temperature`。线性退火使用：

```text
--temperature_schedule linear --temperature 1.0 \
--temperature_end 0.1 --temperature_anneal_steps 1000
```

令 u 为已完成 optimizer 更新次数，D 为退火步数：
`T(u) = T_start + (T_end-T_start) * min(u/D, 1)`。
首批 T=1.0，完成 1000 次更新后的下一批 T=0.1，此后保持；同一个梯度累积窗口
温度不变。要求 D>0 且 `0<T_end≤T_start`。训练较短时可能不会达到终温。

`global_step / max_steps / save_steps` 继续按 rollout batch 计数；
`optimizer_step` 按实际参数更新计数。累积跨 epoch 连续进行，最后不满窗口时按实际
batch 数平均梯度。保存请求延迟到更新边界，最终保存也在尾部更新之后。
Checkpoint 保存计数和下一批将使用的温度/调度参数，但尚不支持恢复训练。

## 命令示例

以下在仓库根目录运行，将 MODEL、TRAIN、OUT 替换为实际路径，每个实验使用独立 OUT。

```bash
# Group 优势，新的温度一致概率模式
python finetuning/qwen3_asr_grpo.py --model_path MODEL --train_file TRAIN \
  --output_dir OUT --advantage_mode group --reward_mode cer

# Greedy baseline，固定温度
python finetuning/qwen3_asr_grpo.py --model_path MODEL --train_file TRAIN \
  --output_dir OUT --advantage_mode greedy --reward_mode cer --temperature 0.8

# Greedy baseline，线性退火（终温是实验选项，并非推荐最优值）
python finetuning/qwen3_asr_grpo.py --model_path MODEL --train_file TRAIN \
  --output_dir OUT --advantage_mode greedy --reward_mode cer \
  --temperature_schedule linear --temperature 1.0 --temperature_end 0.1 \
  --temperature_anneal_steps 1000

# 旧概率模式对照
python finetuning/qwen3_asr_grpo.py --model_path MODEL --train_file TRAIN \
  --output_dir OUT --policy_probability_mode legacy --advantage_mode group \
  --reward_mode cer --temperature 0.8
```

奖励模式仍保留原有规范化口径。`weighted_cer_loop` 只惩罚至少出现两次且次数超过
参考的 n-gram；统计的是重复片段覆盖率，不是精确额外重复字符数。

## 观测与验证边界

`run_config.json` 保存 requested/resolved 参数及计数单位。日志增加温度、optimizer
step、组内奖励 std、优势绝对值均值/最大值、clip fraction。Greedy 模式增加
平均 greedy reward、reward delta、better/equal/worse（容差 1e-6）、same_text。
Same_text 比较解析转写并只裁剪首尾空白，不做大小写或标点归一化。

以独立固定验证集的 **greedy WER/CER** 选择 checkpoint；采样奖励提升不保证 greedy
改善。低温可能缺少探索，且会改变梯度尺度。本次未增加自动评测/模型选择。

本次仅补充回归测试和静态检查，未安装依赖、运行测试、dry-run 或训练。
数值稳定性、真实显存开销及 greedy 收益均待 CUDA 环境验证。
