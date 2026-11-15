[English](README.md) | **简体中文**

# UnifoLM 推理引擎：3.7× 加速

针对 UnifoLM-WMA-0（ASC）的推理优化。在 20 个归档案例上，平均选定时间从 **909.68 s** 降至 **244.91 s**（**3.71×**）。模型权重和输入格式与上游框架保持兼容。

主要改动在注意力、采样和特征流水线路径：

- 条件 KV 缓存与注意力布局控制，使得已计算的 key/value 在 DDIM 步骤间被复用，而不是每步重新计算。
- 自定义 CUDA GEGLU（`ops/gelu_and_mul.cu` + Python 绑定），替代通用 PyTorch GEGLU 的前馈激活。
- Triton 融合的 GroupNorm + SiLU，将两个内存受限内核合并为一个。
- 双 GPU 特征流水线：预测头在一块 GPU 上运行，特征接收与下一阶段在另一块上运行，重叠计算与传输。
- 视频主干中的上下文缓存，当条件未变化时避免重复计算中间特征。

双进程启动器（`tools/run_single_case.py`）管理缓存生命周期，从场景 CSV 中选择单个案例，只有加上 `--run` 时才会启动两个 CUDA 进程。历史环境为 Linux + PyTorch 2.3.1 + 两块 A800 80GB GPU。双 GPU 路径无法在单卡上运行。

CPU 端检查可重新计算归档计时记录，并在无 GPU 情况下验证缓存和启动器逻辑。完整 GPU 运行需要另外提供模型权重、OpenCLIP 资源和转换后的 prompt 目录。

```bash
# CPU 验证
python verify.py
python -m unittest test_records -v

# GPU（准备好权重和数据后）
CUDA_VISIBLE_DEVICES=0,1 python tools/run_single_case.py \
  --checkpoint /path/to/weights.safetensors \
  --prompt-dir /path/to/prompts \
  --run
```

上游：[Unitree Robotics / unifolm-world-model-action](https://github.com/unitreerobotics/unifolm-world-model-action)。
