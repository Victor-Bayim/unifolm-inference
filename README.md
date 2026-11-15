# UnifoLM Inference Engine: 3.7×

UnifoLM-WMA-0 inference on two A800 80GB. 20 cases, mean time **909.68 s → 244.91 s**, **3.71×**.

The time was sitting in attention, the feed-forward activation, and the feature handoff between GPUs.

- Conditional KV cache. Keys and values computed on one DDIM step stay resident and are reused on the next. Attention layout is fixed so the cache hits.
- CUDA GEGLU (`ops/gelu_and_mul.cu`, Python binding) replaces the PyTorch GEGLU on the feed-forward.
- Triton fuses GroupNorm and SiLU into one kernel.
- Dual-GPU pipeline. Prediction head on GPU 0, feature reception and the next stage on GPU 1. Compute and the transfer overlap.
- Video-backbone context cache. Unchanged conditioning does not recompute intermediate features.

`tools/run_single_case.py` owns the cache lifecycle, picks one row from the scene CSV, and only launches the two CUDA processes with `--run`. Measured stack: Linux, PyTorch 2.3.1, two A800 80GB.

```bash
python verify.py
python -m unittest test_records -v

CUDA_VISIBLE_DEVICES=0,1 python tools/run_single_case.py \
  --checkpoint /path/to/weights.safetensors \
  --prompt-dir /path/to/prompts \
  --run
```

[unitreerobotics/unifolm-world-model-action](https://github.com/unitreerobotics/unifolm-world-model-action)
