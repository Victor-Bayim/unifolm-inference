[English](README.md) | **简体中文**

# UnifoLM 推理引擎：3.7× 加速

针对 UnifoLM-WMA-0 世界模型（ASC）的硬件感知推理优化。在 20 个归档案例上，平均选定推理时间从 **909.68 s** 降至 **244.91 s**，实现 **3.71×** 加速，同时与上游权重和数据格式完全兼容。

## 优化内容

工作针对视频世界模型与动作预测推理中的主要开销：

- 条件 KV 缓存，并对注意力布局进行精确控制，减少扩散步骤间的冗余计算。
- 自定义 CUDA GEGLU 内核及 Python 绑定，替代通用 PyTorch 路径上的前馈激活。
- Triton 融合的 GroupNorm + SiLU，将两个内存受限操作合并为单个内核。
- 双 GPU 特征流水线，将预测头与特征接收拆分到两块 A800 80GB GPU 上，重叠计算与通信。
- 上下文缓存与优化的 DDIM 采样路径，在可能处复用中间特征。

所有改动仅限于推理路径；模型权重、checkpoint 格式与输入数据布局保持不变。

## 性能

- 平均选定时间（20 例）：909.6807 s → 244.9054 s
- 加速比（均值之比）：**3.714×**
- 逐案例平均加速比：**3.721×**

结果完整归档于 `benchmarks/`，并可通过提供的验证脚本重新计算。双 GPU 路径需要两块 A800 80GB GPU。

## 快速开始

```bash
# CPU 验证记录与缓存逻辑
python verify.py
python -m unittest test_records -v

# GPU 执行（准备好权重与数据后）
CUDA_VISIBLE_DEVICES=0,1 python tools/run_single_case.py \
  --checkpoint /path/to/weights.safetensors \
  --prompt-dir /path/to/prompts \
  --run
```

上游框架：[Unitree Robotics / unifolm-world-model-action](https://github.com/unitreerobotics/unifolm-world-model-action)。
