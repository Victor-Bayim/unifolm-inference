# UnifoLM-WMA-0 · Inference implementation and benchmarks

ASC26 团队项目的推理源码与 20 例计时记录。包含 DDIM 条件缓存、CUDA GEGLU、Triton GroupNorm + SiLU，以及双 GPU 执行路径。此包将已有实现与计时记录整理为可以直接检查、复算的目录。

## 快速核查 / Quick verification

Python 3.10+ 即可，无需 CUDA、模型权重或额外 Python 依赖：

```sh
python verify.py
python -m unittest test_records -v
```

复算结果：20 例所选计时的均值为 **909.6807 s → 244.9054 s**；均值之比为 **3.71441667×**。每例比值的平均数为 3.72100202，两种聚合不混用。

The commands recompute the archived measurements, check JSON/CSV agreement and file hashes, and exercise the original duration parser and aggregation utility. They do not launch GPU inference.

## 从结果到实现 / Navigation

| 内容 | 路径 |
| --- | --- |
| 20 例完整时间、比例与运行批次标签 | `benchmarks/cases.csv` |
| 基线与优化计时 JSON | `benchmarks/baseline.json`, `benchmarks/ours.json` |
| 原项目逐例比值及单轮记录 | `benchmarks/speedup_real_vs_merged.json`, `benchmarks/real_times.json` |
| 汇总指标 | `benchmarks/summary.json` |
| 本次核查结果 | `validation.json` |
| 文件校验和、修改说明、历史版本对应关系 | `manifest.json` |
| 团队代码演进的提交标识 | `benchmarks/source-history.json` |
| 双 GPU 调度、权重加载与输出 | `project/scripts/evaluation/world_model_interaction.py` |
| DDIM 采样与固定条件缓存 | `project/src/unifolm_wma/models/samplers/ddim.py` |
| Attention 缓存及布局 | `project/src/unifolm_wma/modules/attention.py` |
| 融合算子 | `project/src/unifolm_wma/modules/networks/fused_ops.py`, `project/ops/gelu_and_mul.cu` |
| 汇总与 PSNR 计算工具 | `project/build_results_json_from_runs.py`, `project/psnr_score_for_challenge.py` |

## 测量方法 / Measurement method

- 时间来自归档 JSON 中的 `real` 记录，按相同场景与 case 配对。每个场景四例，共五个场景。
- JSON 保留了 `baseline_1/2` 和 `saved_1..4` 的来源标签；同包汇总工具按每例基线最大时间、优化最小时间选样。这里的均值是 **20 个已选案例记录的均值**，不是各轮重复实验的平均值。历史命令行与全部逐轮日志不在此包中，不能重新验证整个选样过程。
- `benchmarks/baseline.json` 与 `ours.json` 仅替换私人绝对路径字段，时间数值、`raw` 时长文本和相对运行标签保留。`cases.csv` 和 `summary.json` 由这些 JSON 派生。
- 本组计时 JSON 不包含对应 PSNR。质量评测工具随源码保留；其他版本的质量结果不与此组时间拼接。

Elapsed time is the archived end-to-end shell `real` duration. The documented aggregation selects maximum baseline and minimum optimized time across run roots, then averages the selected records over 20 cases. This is a selected-run comparison, not a repeat-run mean or median. Matched PSNR is not present in these timing JSONs.

## GPU 环境运行 / GPU execution

源码的推理入口明确创建两个 CUDA 进程，GPU 0 和 GPU 1 分工；历史技术配置记录为 2 × A800 80GB PCIe，项目所在平台为八卡集群。

完整推理需要 Linux、两张兼容 CUDA 的 GPU、PyTorch 2.3.1 环境、NVCC、模型 checkpoint 以及赛题输入。依赖版本保留在 `project/pyproject.toml` 和 `project/pixi.lock`；上游安装与输入格式说明保留在 `project/README.md`。模型权重、输入数据集及生成视频未打入本包。

环境与数据准备完成后，在 `project/` 下构建 CUDA 扩展，配置 `configs/inference/world_model_interaction.yaml` 中的数据与模型路径，并运行：

```sh
python setup.py build_ext --inplace
export UNIFOLM_CHECKPOINT=/path/to/unifolm_wma_dual.ckpt
bash scripts/run_world_model_interaction.sh
```

该启动脚本保留项目的 50 步 DDIM、16 帧、320 × 512 与双 GPU 入口。请根据下载的赛题输入配置场景、case 和 prompt 路径；本包不假装在无 GPU 的环境中完成原始性能复现。

## 来源与本次整理 / Provenance

归档 main 提交为 `b7c5f751ba11e98553c743af480335d872dc9100`。工作区也含未提交文件；`manifest.json` 对每个文件列出其与 main 的关系。本包体现团队实现，不根据目录归属推断个人提交作者。

原始源码、上游版权与 LICENSE 保留。发布整理只对 checkpoint 路径作参数化、对计时来源中的私人绝对路径作替换。`projector.py` 的两个条件赋值、缺失冒号和错误消息变量在本次整理中修复，原文与差异保存在 `archived/`；历史计时在这些整理改动之前取得。此模块的运行行为未做 GPU 验证。

`verify.py`、`test_records.py`、CSV、汇总与清单属于此次新增的复算/核查工具。单元测试使用临时合成数据测试解析规则，不把测试数据冒充历史实验日志。

The archived implementation and upstream attribution are retained. Added verification tools and packaging repairs are explicitly separated from historical code and measurements. This package does not assert that all team code was authored by a single contributor.
