**English** | [简体中文](README.zh-CN.md)

# UnifoLM Inference Engine: 3.7× Acceleration

Hardware-aware inference optimizations for the UnifoLM-WMA-0 world model (ASC). Across 20 archived cases, mean selected inference time drops from **909.68 s** to **244.91 s**, delivering a **3.71×** speedup while remaining fully compatible with upstream weights and data formats.

## What Was Optimized

The work targets the dominant costs in video-based world-model and action-prediction inference:

- Conditional KV caching with precise attention layout control, reducing redundant computation across diffusion steps.
- A custom CUDA GEGLU kernel with Python bindings, replacing the generic PyTorch path for the feed-forward activation.
- Triton-fused GroupNorm + SiLU, collapsing two memory-bound operations into a single kernel.
- A dual-GPU feature pipeline that splits the prediction head and feature reception across two A800 80GB GPUs, overlapping computation and communication.
- Context caching and a refined DDIM sampling path that reuses intermediate features where possible.

All changes are isolated to the inference path; model weights, checkpoint format, and input data layout remain unchanged.

## Performance

- Mean selected time (20 cases): 909.6807 s → 244.9054 s
- Speedup (ratio of means): **3.714×**
- Mean per-case speedup: **3.721×**

Results are fully archived in `benchmarks/` and can be recomputed with the provided verification scripts. The dual-GPU path requires two A800 80GB GPUs.

## Quick Start

```bash
# CPU verification of records and cache logic
python verify.py
python -m unittest test_records -v

# GPU execution (after weights and data are prepared)
CUDA_VISIBLE_DEVICES=0,1 python tools/run_single_case.py \
  --checkpoint /path/to/weights.safetensors \
  --prompt-dir /path/to/prompts \
  --run
```

Upstream framework: [Unitree Robotics / unifolm-world-model-action](https://github.com/unitreerobotics/unifolm-world-model-action).
