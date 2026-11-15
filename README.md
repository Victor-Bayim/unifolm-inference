**English** | [简体中文](README.zh-CN.md)

# UnifoLM Inference Engine: 3.7× Acceleration

Hardware-aware inference optimizations for the UnifoLM-WMA-0 world model. Across 20 archived cases, mean selected inference time drops from **909.68 s** to **244.91 s**, delivering a **3.71×** speedup.

## Optimizations

- Conditional KV caching with precise control over attention layout
- Custom CUDA GEGLU kernel with Python bindings
- Triton-fused GroupNorm + SiLU
- Dual-GPU feature pipeline that splits prediction head and feature reception
- Context caching and refined DDIM sampling path

All changes remain fully compatible with upstream model weights and data formats.

## Performance

- Mean selected time: 909.6807 s → 244.9054 s
- Speedup (ratio of means): **3.714×**
- Mean per-case speedup: **3.721×**

Results are archived and recomputable. The dual-GPU path requires two A800 80GB GPUs.

## Quick Start

```bash
# CPU verification
python verify.py
python -m unittest test_records -v

# GPU execution (after weights and data are prepared)
CUDA_VISIBLE_DEVICES=0,1 python tools/run_single_case.py \
  --checkpoint /path/to/weights.safetensors \
  --prompt-dir /path/to/prompts \
  --run
```

Upstream framework: Unitree Robotics / unifolm-world-model-action.
