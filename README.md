**English** | [简体中文](README.zh-CN.md)

# UnifoLM inference optimization · ASC

Inference optimization code, launch configuration and archived timings for an ASC team project. Areas include conditional KV caching, CUDA GEGLU, Triton GroupNorm + SiLU, and a dual-GPU feature pipeline.

## Implementation map

| Area | Key files |
| --- | --- |
| Conditional KV cache, attention layout and GEGLU calls | [attention.py](project/src/unifolm_wma/modules/attention.py) |
| DDIM sampling | [ddim.py](project/src/unifolm_wma/models/samplers/ddim.py) |
| Context caching, video backbone and feature transmission | [wma_model.py](project/src/unifolm_wma/modules/networks/wma_model.py) |
| GPU 1 prediction head and feature reception | [conditional_unet1d.py](project/src/unifolm_wma/models/diffusion_head/conditional_unet1d.py) |
| Triton GroupNorm + SiLU | [fused_ops.py](project/src/unifolm_wma/modules/networks/fused_ops.py) |
| CUDA GEGLU, Python bindings and build entry | [ops/](project/ops), [setup.py](project/setup.py) |
| Dual-process launch, cache lifecycle, weights and output | [world_model_interaction.py](project/scripts/evaluation/world_model_interaction.py) |
| Inference parameters and dependency snapshot | [configs/inference/](project/configs/inference), [pyproject.toml](project/pyproject.toml), [pixi.lock](project/pixi.lock) |
| Timings and recomputation for 20 cases | [benchmarks/](benchmarks), [verify.py](verify.py), [test_records.py](test_records.py) |

## CPU checks

Record recomputation requires only the Python 3.10+ standard library:

```sh
python verify.py
python -m unittest test_records -v
```

Cache and single-case launcher regressions additionally require CPU PyTorch and PyYAML:

```sh
python -m unittest test_attention_cache test_single_case_launcher -v
```

## GPU prerequisites

The historical environment used Linux, Python 3.10.18, CUDA-enabled PyTorch 2.3.1, torchvision 0.18.1, xFormers 0.0.27, NVCC and two A800 80GB PCIe GPUs. The dual-process path requires at least two GPUs and cannot run on one. Historical dependencies are retained in `project/pyproject.toml` and `project/pixi.lock`; additional directly imported h5py, safetensors and Triton versions are listed in [requirements-inference-extra.txt](requirements-inference-extra.txt).

On Linux, configure PyTorch 2.3.1 and NVCC with a matching CUDA toolkit, then install and build:

```sh
cd project
python -m pip install "setuptools==65.6.3" wheel
python -m pip install --no-build-isolation -e .
python -m pip install -r ../requirements-inference-extra.txt
python setup.py build_ext --inplace
cd ..
```

The build target is `ops.my_gelu_op`, using `project/ops/binding.cpp` and `gelu_and_mul.cu`. The launcher adds `project/` and `project/src/` to `PYTHONPATH`.

Prepare the model and inputs separately:

- Model weights, OpenCLIP encoder resources, challenge videos and converted input data are not bundled in the repository.
- The entry point reads checkpoints with `safetensors.torch.load_file`; contents must be compatible safetensors weights. A `.ckpt` suffix alone does not identify the format.
- Input directories must contain scene CSV files, `videos/`, `transitions/` and normalization statistics. See [project/README.md](project/README.md) for upstream input formats, models and data links. Check checkpoint keys and shapes against the snapshot configuration.

## One-case launch

Prepare one case from the repository root. By default, this validates metadata, generates configuration and prints commands, **without starting GPUs**:

```sh
CUDA_VISIBLE_DEVICES=0,1 python tools/run_single_case.py \
  --checkpoint /path/to/weights.safetensors \
  --prompt-dir /path/to/converted_prompts \
  --dataset unitree_z1_stackbox --case-index 0 \
  --output-dir ./results/stackbox-case0
```

After preparing resources, append `--run` to start two CUDA processes. `--case-index` is the zero-based CSV row index, not the original case name. The launcher retains only the current scene's configuration weights, selects one row, and uses 50-step DDIM, 16 frames, 320×512, batch 1 and the scene's historical rollout count. `--n-iter 1` provides a shorter smoke test but changes the workload, so its timing is not directly comparable to historical timings.

## Measurement boundaries

Mean selected times for the 20 archived cases are **909.6807 → 244.9054 s**, with a ratio of means of **3.71441667×**. The mean per-case ratio is **3.72100202**; these aggregations are distinct. The original aggregation selects the maximum available baseline time and minimum optimized time for each case.

`benchmarks/baseline.json` and `ours.json` retain original duration text and relative batch labels.

## Provenance and maintenance

Upstream framework: [Unitree Robotics / unifolm-world-model-action](https://github.com/unitreerobotics/unifolm-world-model-action).
