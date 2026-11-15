#!/usr/bin/env python3
"""Prepare one archived-shape case; print the launch command unless --run is set."""
import argparse
import csv
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / 'project'
SCENARIOS = {
    'unitree_z1_stackbox': (12, 4),
    'unitree_z1_dual_arm_stackbox': (7, 4),
    'unitree_z1_dual_arm_stackbox_v2': (11, 4),
    'unitree_z1_dual_arm_cleanup_pencils': (8, 4),
    'unitree_g1_pack_camera': (11, 6),
}


def prepare(args):
    import yaml
    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    prompt_dir = Path(args.prompt_dir).expanduser().resolve(strict=True)
    if not checkpoint.is_file() or not prompt_dir.is_dir():
        raise ValueError('checkpoint must be a file and prompt-dir must be a directory')
    with (prompt_dir / (args.dataset + '.csv')).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if args.case_index < 0 or args.case_index >= len(rows):
        raise ValueError(f'case-index must be in 0..{len(rows) - 1}')
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    config['data']['params']['test']['params']['data_dir'] = str(prompt_dir)
    config['data']['params']['dataset_and_weights'] = {args.dataset: 1.0}
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    generated_config = output / 'single-case-config.yaml'
    generated_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    n_iter, frame_stride = SCENARIOS[args.dataset]
    if args.n_iter is not None:
        if args.n_iter <= 0:
            raise ValueError('n-iter must be positive')
        n_iter = args.n_iter
    command = [sys.executable, 'scripts/evaluation/world_model_interaction.py',
               '--ckpt_path', str(checkpoint), '--config', str(generated_config),
               '--prompt_dir', str(prompt_dir), '--dataset', args.dataset,
               '--case_index', str(args.case_index), '--savedir', str(output),
               '--seed', '123', '--bs', '1', '--height', '320', '--width', '512',
               '--unconditional_guidance_scale', '1.0', '--ddim_steps', '50',
               '--ddim_eta', '1.0', '--video_length', '16', '--frame_stride', str(frame_stride),
               '--n_action_steps', '16', '--exe_steps', '16', '--n_iter', str(n_iter),
               '--timestep_spacing', 'uniform_trailing', '--guidance_rescale', '0.7', '--perframe_ae']
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(PROJECT / 'src'), str(PROJECT), env.get('PYTHONPATH', '')])
    return command, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Checkpoint read by safetensors.torch.load_file')
    parser.add_argument('--prompt-dir', required=True, help='Converted prompt/data tree with CSV, videos and transitions')
    parser.add_argument('--dataset', choices=SCENARIOS, required=True)
    parser.add_argument('--case-index', type=int, default=0, help='Zero-based row index, not the case label')
    parser.add_argument('--config', default=str(PROJECT / 'configs/inference/world_model_interaction.yaml'))
    parser.add_argument('--output-dir', default=str(ROOT / 'results/single-case'))
    parser.add_argument('--n-iter', type=int, default=None, help='Override rollout count; changes the historical workload')
    parser.add_argument('--run', action='store_true', help='Actually launch the two-GPU process; default only prepares/prints')
    args = parser.parse_args()
    command, env = prepare(args)
    print('Working directory:', PROJECT)
    print('Command:', shlex.join(command))
    if args.run:
        subprocess.run(command, cwd=PROJECT, env=env, check=True)
    else:
        print('Prepared only. No GPU/model code was imported or executed. Add --run after checking inputs.')


if __name__ == '__main__':
    main()
