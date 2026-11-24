# HM-ASR 全量 test 噪声鲁棒性评测

本轮使用当前 HM-ASR 启用测试集的全部有效语音，不使用此前每语料 200 句的抽样，
不设 2–15 秒等时长过滤或截断。原始启用的 20 套数据中，按用户要求排除
AliMeeting、AMI、Renji（含 easy/hard），剩余 16 套、143,549 条源语音、205.073 小时。
WenetSpeech meeting 与 RAMC 仍在范围内；不将用户指定以外的数据集额外排除。

输入源语音中若存在全零、空或非有限值波形，SNR 无法定义。这些样本记入
quarantine，每个 ID、参考文本、原因及波形指纹均保留，并再次从源分片读取波形复核。
所有源 ID 必须恰好属于有效评测集或无效源清单；没有未交代的丢弃。
生成结果分别记录 source / evaluated / invalid 数量。官方 normalizer 将参考文本
清空时，仍使用 canonical scorer 的空参考跳过策略，在报告中另外记录有效计分数。

## 条件与控制

每句有效语音生成四个版本：

- clean：原始录音加统一的音量控制，不做去噪。
- noise_snr20、noise_snr10、noise_snr0：分别叠加 20/10/0 dB 的新增噪声。

只读取 data/augmentation/splits/v1/noise.test.jsonl，不读取 train/dev 噪声。
六类噪声 DNS AudioSet、DNS Freesound、ESC-50、MUSAN music/noise/speech 按语句数
均衡分配。先对源 ID 做固定种子的哈希排序，再分配噪声来源，避免与方言或清单顺序绑定。
每句在三个 SNR 下使用同一个噪声文件、裁剪位置及重复补齐方式。

SNR 是整句参考波形 RMS 与新增噪声 RMS 的比值；原始录音可能已有噪声，
该值不代表新增噪声与原生背景噪声合计后的总声学 SNR。
语音统一至 RMS 0.05，四个条件共享防削波增益，保存为 16 kHz 单声道 PCM16 FLAC。
生成后重新解码测量 SNR，容差 0.1 dB。clean 也重新推理，保证与加噪版本采用相同增益。
不使用 RIR、变速或 SpecAugment。

## 文件与流程

主输出：outputs/robust_mix_1000h/full_test_noise_v1/

- protocol.json：范围、原始测试集清单哈希、噪声划分哈希与固定协议。
- webdataset/：每套语料每种条件的 tar 分片、manifest.jsonl、STATS.json。
- ready/：单套语料完成全部四条件并通过校验后，原子写入的完成记录。
- quarantine/：物理上无法定义 SNR 的无效源音频及波形复核记录。
- runtime/：本轮冻结的 HM-ASR 代码、粤语 normalization、Qwen 推理包与生成/计分工具。
- per_dataset/：两模型对每套完整语料的 canonical 推理结果。
- status.json、partial_results.json：实际进度与已完成语料的成对指标。
- baseline/、final/：全部子任务完成后合并的 canonical 结果及原始子任务路径。
- robustness_metrics.csv、noise_family_metrics.csv、noise_curve_auc.csv：条件指标、噪声来源细分、曲线面积。
- report.md：最终汇总，附引用的 AliMeeting/AMI/Renji 原始真实场景结果。

生成与推理可以流水执行：单套语料的四条件全部校验通过后，基座和 LoRA 才读取它；
其余语料同时继续生成。两个模型输入相同分片，保持 BF16、greedy、batch size 16、
max_new_tokens 1024、按数据集指定语言。并发运行只用于识别质量评测，不用于速度比较。
粤语使用本轮冻结的 masr_zh_yue，即 s2hk 和 ref/hyp 对称同音/异体字归一。
生成器可通过 --benchmark_root 指定 HM-ASR 目录；计分器支持 HMASR_BENCHMARK_ROOT。
在冻结运行目录中，计分器优先使用相邻的 runtime/hmasr_benchmark，避免其他代码更新改变口径。

默认生成完整测试集。smoke_limit 仅用于独立的端到端小规模检查，不用于正式评测。
中断的未冻结构建可使用 --resume；协议不符则拒绝续跑。已写入 generation_summary.json
的构建拒绝覆盖。分片按样本数滚动并保留至少 25 GiB 的空间预算。

最终合并只接受真实完成、模型/解码/代码指纹一致的子任务。
推理后再次复核分片和素材归属，并从成对预测复算 WER/CER、相对 clean 的退化量、
相对基座的错误率降低比例、六种噪声来源结果与 0/10/20 dB 曲线面积。
不把不同语料的 WER/CER 混为一个总分。
