# UnifoLM Inference Optimization (ASC)

Inference acceleration work on UnifoLM-WMA-0 for the ASC competition.

## Upstream

This work is based on the official UnifoLM-WMA-0 framework by Unitree Robotics:

**https://github.com/unitreerobotics/unifolm-world-model-action**

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

Results are from the competition evaluation setup and are not claimed as general production numbers. See the upstream repository for the base framework, training code, and model weights.

## License

See upstream license. This snapshot is provided for portfolio reference.
