# UnifoLM inference optimization · ASC

ASC 团队项目的推理优化代码、启动配置与归档计时。方向包括条件 KV 缓存、CUDA GEGLU、Triton GroupNorm + SiLU，以及双 GPU 特征流水。本人角色为参赛队员；本仓库展示团队实现，不把所有代码归为个人原创。

This repository restores the supplied team implementation and archived measurements. It is an inspectable inference snapshot with explicitly separated maintenance changes, not a verified reproduction of the original A800 experiment.

## 实现导航 / Implementation map

| 内容 | 关键文件 |
| --- | --- |
| 条件 KV 缓存、注意力布局、GEGLU 调用 | [attention.py](project/src/unifolm_wma/modules/attention.py) |
| DDIM 采样 | [ddim.py](project/src/unifolm_wma/models/samplers/ddim.py) |
| 上下文缓存、视频主干与特征发送 | [wma_model.py](project/src/unifolm_wma/modules/networks/wma_model.py) |
| GPU 1 的预测头及特征接收 | [conditional_unet1d.py](project/src/unifolm_wma/models/diffusion_head/conditional_unet1d.py) |
| Triton GroupNorm + SiLU | [fused_ops.py](project/src/unifolm_wma/modules/networks/fused_ops.py) |
| CUDA GEGLU、Python 绑定与编译入口 | [ops/](project/ops/)、[setup.py](project/setup.py) |
| 双进程启动、缓存生命周期、权重与输出 | [world_model_interaction.py](project/scripts/evaluation/world_model_interaction.py) |
| 推理参数与依赖快照 | [configs/inference/](project/configs/inference/)、[pyproject.toml](project/pyproject.toml)、[pixi.lock](project/pixi.lock) |
| 20 例时间与复算 | [benchmarks/](benchmarks/)、[verify.py](verify.py)、[test_records.py](test_records.py) |

`project/` 保留所提供的完整 Python 包和配置，避免只上传一个无法导入的代码节选。训练及其他辅助文件是依赖快照的一部分，本次没有验证训练流程。原先空的 `src/` 占位目录已移除。

## CPU 检查 / CPU checks

记录复算只需 Python 3.10+ 标准库：

```sh
python verify.py
python -m unittest test_records -v
```

缓存与单例启动器回归另外需要 CPU PyTorch 和 PyYAML：

```sh
python -m unittest test_attention_cache test_single_case_launcher -v
```

2026-11-15：原有 8 项记录测试及新增 8 项回归均通过；20 例 JSON/CSV 复算、79 份历史文件哈希和 59 份项目 Python 语法检查通过。新增测试从真实源码 AST 提取 `forward`、`efficient_forward` 等方法，使用实际 CPU 小张量验证缓存开关等价、缓存清空与自注意力隔离。仅 xFormers 算子调用与布局变换使用 CPU 数学替代；没有加载模型或运行 CUDA。详细范围见 [VALIDATION.md](docs/VALIDATION.md)。

## GPU 环境准备 / GPU prerequisites

历史环境为 Linux、Python 3.10.18、CUDA-enabled PyTorch 2.3.1、torchvision 0.18.1、xFormers 0.0.27、NVCC，以及两张 A800 80GB PCIe。双进程路径要求至少两张 GPU，不能用单卡运行。历史依赖在 `project/pyproject.toml` 和 `project/pixi.lock` 中保留；额外直接使用的 h5py、safetensors、Triton 版本列于 [requirements-inference-extra.txt](requirements-inference-extra.txt)。

先在 Linux 配好匹配 CUDA toolkit 的 PyTorch 2.3.1 与 NVCC，再安装和构建：

```sh
cd project
python -m pip install "setuptools==65.6.3" wheel
python -m pip install --no-build-isolation -e .
python -m pip install -r ../requirements-inference-extra.txt
python setup.py build_ext --inplace
cd ..
```

编译目标为 `ops.my_gelu_op`，对应 `project/ops/binding.cpp` 与 `gelu_and_mul.cu`；启动器会把 `project/` 和 `project/src/` 加入 `PYTHONPATH`。这些 GPU 安装与编译命令本次没有执行，不能把 CPU 检查当作编译成功记录。

模型和输入需单独准备：

- 模型权重、OpenCLIP 编码器资源、赛题视频与转换后的输入数据均未打入仓库。
- 当前入口通过 `safetensors.torch.load_file` 读取 checkpoint；文件内容必须为兼容的 safetensors 权重，不能只凭 `.ckpt` 后缀判断格式。
- 输入目录需含对应场景 CSV、`videos/`、`transitions/` 和归一化统计。上游的输入格式、模型及数据链接见保留的 [project/README.md](project/README.md)。请核对 checkpoint 与该快照配置的键和形状。

## 单例运行 / One-case launch

从仓库根目录准备一例。默认只验证元数据、生成配置并打印命令，**不启动 GPU**：

```sh
CUDA_VISIBLE_DEVICES=0,1 python tools/run_single_case.py \
  --checkpoint /path/to/weights.safetensors \
  --prompt-dir /path/to/converted_prompts \
  --dataset unitree_z1_stackbox --case-index 0 \
  --output-dir ./results/stackbox-case0
```

确认资源后追加 `--run` 才启动两个 CUDA 进程。`--case-index` 是 CSV 的零基行号，不是原始 case 名称。启动器只保留当前场景的配置权重、选择单行，并沿用 50 步 DDIM、16 帧、320×512、batch 1 及该场景的历史 rollout 数。可用 `--n-iter 1` 做较短烟测，但这会改变工作量，结果不能与历史计时直接比较。

旧批量脚本保留在 `project/scripts/run_world_model_interaction.sh` 供审查。该快照的 GPU 1 只执行一例的 `n_iter` 轮，而 GPU 0 原来遍历 CSV；多行 CSV 的批量生命周期未验证，不应直接把旧脚本当作可靠的全场景复现入口。新增单例选择避免这一不匹配；通信、CUDA stream、融合算子数值与硬件端到端执行仍需在目标环境验证。

## 结果边界 / Measurement boundaries

归档 20 例的选样耗时均值为 **909.6807 → 244.9054 s**，均值之比 **3.71441667×**；每例比值的平均数为 **3.72100202**，两者不混用。原汇总规则逐例取可用基线轮次最大时间、优化轮次最小时间。这是归档选样，不能解释为同轮配对或重复运行的典型加速。

`benchmarks/baseline.json` 与 `ours.json` 保留时长原文及相对批次标签。全部逐轮日志、原始命令行、模型权重、视频产物和对应这组选样的 PSNR 不在现有材料中，不能重新验证完整选样过程或同一配置的速度/质量关系。PSNR 工具 [psnr_score_for_challenge.py](project/psnr_score_for_challenge.py) 仅供审查；未提供的质量成绩不补造。

## 来源与维护 / Provenance and maintenance

上游框架：[Unitree Robotics / unifolm-world-model-action](https://github.com/unitreerobotics/unifolm-world-model-action)。**原材料没有确认官方上游 revision**；不能用其当前 main 反推历史基线。归档团队版本标识 `b7c5f751ba11e98553c743af480335d872dc9100` 不等于已核实的官方上游 commit，也不是本仓库的 HEAD。

[manifest.json](manifest.json) 保留原历史文件哈希；[maintenance.json](maintenance.json) 逐项列出维护前原文、维护后哈希与 [patches/](patches/)。新增维护只包含：显式的 `context is not None` 布尔条件，以及单例 CSV 行选择。`efficient_forward` 原实现未改动。维护前文件在 `archived/before-maintenance/`；早先 projector 整理修复的原文和补丁也保留在 `archived/`。这些修复均不追溯解释历史性能。

保留的源码版权注释和 LICENSE 不代表个人重新许可。源包的 LICENSE 与 pyproject license 元数据存在不同描述，详见 [PROVENANCE.md](docs/PROVENANCE.md)；本次未替上游作许可决定，也未新增个人授权声明。
