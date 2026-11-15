**English** | [简体中文](README.zh-CN.md)

# UnifoLM Inference Engine: 3.7× Acceleration

Inference optimizations for UnifoLM-WMA-0 (ASC). On the 20 archived cases the mean selected time drops from **909.68 s** to **244.91 s** (**3.71×**). Model weights and input format stay compatible with the upstream framework.

The main changes sit in the attention, sampling, and feature-pipeline paths:

- Conditional KV cache and attention layout control so that previously computed keys/values are reused across DDIM steps instead of being recomputed.
- A custom CUDA GEGLU (`ops/gelu_and_mul.cu` + Python binding) that replaces the generic PyTorch GEGLU for the feed-forward activation.
- Triton-fused GroupNorm + SiLU that collapses two memory-bound kernels into one.
- Dual-GPU feature pipeline: the prediction head runs on one GPU while feature reception and the next stage run on the other, overlapping compute and transfer.
- Context caching in the video backbone so intermediate features are not recomputed when the conditioning has not changed.

The dual-process launcher (`tools/run_single_case.py`) manages cache lifecycle, selects a single case from the scene CSV, and starts the two CUDA processes only when `--run` is given. Historical environment was Linux + PyTorch 2.3.1 + two A800 80GB GPUs. The dual-GPU path cannot run on a single card.

CPU-side checks recompute the archived timing records and exercise the cache and launcher logic without a GPU. Full GPU runs require the model weights, OpenCLIP resources, and converted prompt directories to be supplied separately.

```bash
# CPU verification
python verify.py
python -m unittest test_records -v

# GPU (after weights and data are ready)
CUDA_VISIBLE_DEVICES=0,1 python tools/run_single_case.py \
  --checkpoint /path/to/weights.safetensors \
  --prompt-dir /path/to/prompts \
  --run
```

Upstream: [Unitree Robotics / unifolm-world-model-action](https://github.com/unitreerobotics/unifolm-world-model-action).
