# UnifoLM Inference Optimization (ASC)

Inference acceleration work on UnifoLM-WMA-0 for the ASC competition.

## Overview

UnifoLM-WMA-0 is a world-model-action framework. This repository contains the inference-path optimizations used in the competition setting (condition caching, operator fusion, and dual-GPU pipeline).

## Role

Team member on the ASC entry. Optimizations were developed as part of the team effort.

## Key Results

On the archived selection (20 cases, dual A800):

- End-to-end average time: 909.7 s → 244.9 s (3.71×)

## Structure

- `src/` — modified inference and utility code
- `scripts/` — related launch and evaluation scripts
- `configs/` — configuration used for the optimized path

## Notes

This is built on the upstream UnifoLM-WMA-0 project. See the original repository for the base framework. Results are from the competition evaluation setup and are not claimed as general production numbers.

## License

See upstream license. This snapshot is provided for portfolio reference.
