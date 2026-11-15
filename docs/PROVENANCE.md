# Snapshot provenance

- Official framework: https://github.com/unitreerobotics/unifolm-world-model-action . Its historical base revision is not established by the supplied material and remains unverified.
- Archived team main identifier: `b7c5f751ba11e98553c743af480335d872dc9100`. Some supplied workspace files differ from that archived team commit. The relation for each file is recorded in `manifest.json`; this identifier is not asserted to be an official Unitree revision.
- `benchmarks/source-history.json` records team history identifiers. It is not proof that one person authored all optimizations. No personal module ownership is inferred from filesystem placement or snapshot import commits.
- The supplied snapshot included repaired projector syntax and a parameterized checkpoint path. Original projector text and its earlier patch remain under `archived/`; `docs/archive-readme.md` preserves the preceding packaging explanation.
- This restoration adds the explicit non-None cache condition and a CSV case selector. Exact pre-maintenance source copies and generated unified diffs are retained. `maintenance.json` records before/after SHA-256 values. `verify.py` checks original snapshot hashes through the preserved copies for these two maintained files, and checks active maintained hashes separately.
- `archived/verify-before-maintenance.py.txt` retains the prior verification program. The active verifier only extends integrity handling; the eight original record tests and all timing inputs are retained.
- The source package's LICENSE text says Creative Commons Attribution-NonCommercial 4.0, while its pyproject metadata declares BSD-3-Clause. Both are preserved as supplied. This repository does not resolve that upstream metadata difference or apply a new personal license.

## Maintenance tests are not historical GPU runs

The pre-maintenance ordinary attention forward applies Python truth testing to a multi-element context tensor. The new CPU regression demonstrates that failure and exercises the maintained method on real PyTorch tensors. The efficient path is tested and its AST is unchanged. These tests substitute CPU attention mathematics for the CUDA/xFormers call; they do not validate Triton, NVCC compilation, CUDA stream behavior, NCCL coordination, checkpoint compatibility or original competition quality.

The historical GPU 1 routine processes one CSV case worth of iterations. The new one-case launcher generates a narrowed dataset config and selects a CSV row; multi-case scheduling is deliberately not advertised as validated. No baseline/optimized hardware measurements were repeated in this restoration.
