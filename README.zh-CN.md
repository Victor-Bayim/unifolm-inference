[English](README.md) | **简体中文**

# UnifoLM 推理优化 · ASC

ASC 团队项目的推理优化代码、启动配置与归档计时。方向包括条件 KV 缓存、CUDA GEGLU、Triton GroupNorm + SiLU，以及双 GPU 特征流水。

## 实现导航

| 内容 | 关键文件 |
| --- | --- |
| 条件 KV 缓存、注意力布局、GEGLU 调用 | [attention.py](project/src/unifolm_wma/modules/attention.py) |
| DDIM 采样 | [ddim.py](project/src/unifolm_wma/models/samplers/ddim.py) |
| 上下文缓存、视频主干与特征发送 | [wma_model.py](project/src/unifolm_wma/modules/networks/wma_model.py) |
| GPU 1 的预测头及特征接收 | [conditional_unet1d.py](project/src/unifolm_wma/models/diffusion_head/conditional_unet1d.py) |
| Triton GroupNorm + SiLU | [fused_ops.py](project/src/unifolm_wma/modules/networks/fused_ops.py) |
| CUDA GEGLU、Python 绑定与编译入口 | [ops/](project/ops)、[setup.py](project/setup.py) |
| 双进程启动、缓存生命周期、权重与输出 | [world_model_interaction.py](project/scripts/evaluation/world_model_interaction.py) |
| 推理参数与依赖快照 | [configs/inference/](project/configs/inference)、[pyproject.toml](project/pyproject.toml)、[pixi.lock](project/pixi.lock) |
| 20 例时间与复算 | [benchmarks/](benchmarks)、[verify.py](verify.py)、[test_records.py](test_records.py) |

## CPU 检查

记录复算只需 Python 3.10+ 标准库：

```sh
python verify.py
python -m unittest test_records -v
```

缓存与单例启动器回归另外需要 CPU PyTorch 和 PyYAML：

```sh
python -m unittest test_attention_cache test_single_case_launcher -v
```

## GPU 环境准备

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

编译目标为 `ops.my_gelu_op`，对应 `project/ops/binding.cpp` 与 `gelu_and_mul.cu`；启动器会把 `project/` 和 `project/src/` 加入 `PYTHONPATH`。

模型和输入需单独准备：

- 模型权重、OpenCLIP 编码器资源、赛题视频与转换后的输入数据均未打入仓库。
- 当前入口通过 `safetensors.torch.load_file` 读取 checkpoint；文件内容必须为兼容的 safetensors 权重，不能只凭 `.ckpt` 后缀判断格式。
- 输入目录需含对应场景 CSV、`videos/`、`transitions/` 和归一化统计。上游的输入格式、模型及数据链接见保留的 [project/README.md](project/README_cn.md)。请核对 checkpoint 与该快照配置的键和形状。

## 单例运行

从仓库根目录准备一例。默认只验证元数据、生成配置并打印命令，**不启动 GPU**：

```sh
CUDA_VISIBLE_DEVICES=0,1 python tools/run_single_case.py \
  --checkpoint /path/to/weights.safetensors \
  --prompt-dir /path/to/converted_prompts \
  --dataset unitree_z1_stackbox --case-index 0 \
  --output-dir ./results/stackbox-case0
```

确认资源后追加 `--run` 才启动两个 CUDA 进程。`--case-index` 是 CSV 的零基行号，不是原始 case 名称。启动器只保留当前场景的配置权重、选择单行，并沿用 50 步 DDIM、16 帧、320×512、batch 1 及该场景的历史 rollout 数。可用 `--n-iter 1` 做较短烟测，但这会改变工作量，结果不能与历史计时直接比较。

## 结果边界

归档 20 例的选样耗时均值为 **909.6807 → 244.9054 s**，均值之比 **3.71441667×**；每例比值的平均数为 **3.72100202**，两者不混用。原汇总规则逐例取可用基线轮次最大时间、优化轮次最小时间。

`benchmarks/baseline.json` 与 `ours.json` 保留时长原文及相对批次标签。

## 来源与维护

上游框架：[Unitree Robotics / unifolm-world-model-action](https://github.com/unitreerobotics/unifolm-world-model-action)。
