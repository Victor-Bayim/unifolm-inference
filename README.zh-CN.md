[English](README.md) | **简体中文**

# UnifoLM 推理引擎：3.7×

UnifoLM-WMA-0 在两块 A800 80GB 上的推理。20 个案例，平均时间 **909.68 s → 244.91 s**，**3.71×**。

时间主要在注意力、前馈激活，以及两块 GPU 之间的特征交接。

- 条件 KV 缓存。上一步 DDIM 算出的 key/value 留在显存里，下一步直接用。注意力布局固定，缓存能命中。
- CUDA GEGLU（`ops/gelu_and_mul.cu` + Python 绑定）替掉 PyTorch GEGLU 的前馈激活。
- Triton 把 GroupNorm 和 SiLU 融成一个内核。
- 双 GPU 流水线。预测头在 GPU 0，特征接收和下一阶段在 GPU 1。计算和传输重叠。
- 视频主干的上下文缓存。条件没变就不重算中间特征。

`tools/run_single_case.py` 管缓存生命周期，从场景 CSV 取一行，加 `--run` 才拉起两个 CUDA 进程。测量环境：Linux，PyTorch 2.3.1，两块 A800 80GB。

```bash
python verify.py
python -m unittest test_records -v

CUDA_VISIBLE_DEVICES=0,1 python tools/run_single_case.py \
  --checkpoint /path/to/weights.safetensors \
  --prompt-dir /path/to/prompts \
  --run
```

[unitreerobotics/unifolm-world-model-action](https://github.com/unitreerobotics/unifolm-world-model-action)
