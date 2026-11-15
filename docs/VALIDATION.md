# Validation · 2026-11-15

Executed on Windows with Python 3.12.14, PyTorch 2.15.0+cpu and PyYAML 6.0.3, separate from the archived Linux/Python 3.10.18/PyTorch 2.3.1/CUDA environment:

```text
python verify.py
python -m unittest test_records test_attention_cache test_single_case_launcher -v
python tools/run_single_case.py --help
```

Results:

- 16 tests passed: 8 original record tests, 7 AST attention/maintenance tests, 1 single-case configuration/argv test.
- 20 selected cases recomputed: baseline mean 909.6807 s, optimized mean 244.9054 s, ratio of means 3.714416668640218. Mean of per-case ratios: 3.7210020151505843.
- 79 original snapshot hashes verified; 59 project Python files parsed; 2 maintained source hashes checked independently. A separate static check parsed all 66 repository Python files, resolved every explicit local unifolm_wma import module, and confirmed both declared CUDA extension source files are present.
- Before/after maintenance diffs match the active source and preserved original copies.
- Cache on/off outputs, cache reuse, cross-session clearing and self-attention isolation were checked using real small CPU tensors and AST-extracted source methods. The historical Tensor truth-value failure is reproduced by a regression test.
- The efficient forward AST is unchanged. CPU tensor tests cover ordinary forward and the single-context/self-attention branch of efficient forward; the image/state/action branch with a CUDA action mask is not executed. CPU tests substitute mathematically equivalent attention and layout operations, and do not import the full GPU/model package.
- Single-case preparation was tested with temporary synthetic CSV metadata. No model was loaded and no GPU subprocess was launched.

No CUDA compilation, Triton execution, A800 inference, end-to-end PSNR or new performance measurement was performed. The original selected-record timing JSON has no matched PSNR; the source of all historical quality and timing claims remains separate.
