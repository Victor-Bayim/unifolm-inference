[English](README.md) | **简体中文**

# UnifoLM 推理引擎：3.7× 加速

针对 UnifoLM-WMA-0 世界模型的硬件感知推理优化。在 20 个归档案例上，平均选定推理时间从 **909.68 s** 降至 **244.91 s**，实现 **3.71×** 加速。

## 优化内容

- 条件 KV 缓存，并对注意力布局进行精确控制
- 自定义 CUDA GEGLU 内核及 Python 绑定
- Triton 融合的 GroupNorm + SiLU
- 双 GPU 特征流水线（预测头与特征接收拆分）
- 上下文缓存与优化的 DDIM 采样路径

所有改动与上游模型权重及数据格式完全兼容。

## 性能

- 平均选定时间：909.6807 s → 244.9054 s
- 加速比（均值之比）：**3.714×**
- 逐案例平均加速比：**3.721×**

结果完整归档并可重新计算。双 GPU 路径需要两块 A800 80GB GPU。

## 快速开始

```bash
# CPU 验证
python verify.py
python -m unittest test_records -v

# GPU 执行（准备好权重与数据后）
CUDA_VISIBLE_DEVICES=0,1 python tools/run_single_case.py \
  --checkpoint /path/to/weights.safetensors \
  --prompt-dir /path/to/prompts \
  --run
```

上游框架：Unitree Robotics / unifolm-world-model-action。
