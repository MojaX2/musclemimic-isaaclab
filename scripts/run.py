"""Supported train/evaluate entry point; run with Isaac Lab's Python interpreter."""
import argparse
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BEST = ROOT / 'artifacts/best-policy'
CODE = ROOT / 'examples/isaaclab_newton_fullbody'
EXPECTED = {'newton': '1.6.0', 'mujoco': '3.12.0', 'mujoco-warp': '3.12.0',
            'warp-lang': '1.17.0', 'rsl-rl-lib': '5.5.1', 'tensordict': '0.13.0',
            'musclemimic_models': '1.0.6'}

def model_assets(path=None):
    if path:
        folder = Path(path).expanduser().resolve()
    else:
        spec = importlib.util.find_spec('musclemimic_models')
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError('Install requirements-runtime.txt or pass --model-assets DIR')
        folder = Path(next(iter(spec.submodule_search_locations))) / 'model'
    if not (folder / 'meshes').is_dir() or not (folder / 'scene').is_dir():
        raise RuntimeError(f'Model assets need meshes/ and scene/: {folder}')
    return folder

def prepare_assets(folder):
    target = ROOT / 'outputs/isaac_velocity/assets'
    target.mkdir(parents=True, exist_ok=True)
    xml = (BEST / 'model.xml.template').read_text().replace('__MODEL_ASSET_ROOT__', folder.as_posix())
    for name, data in [('model.xml', xml.encode()), ('initial.npz', (BEST / 'initial.npz').read_bytes())]:
        dest = target / name
        if dest.exists() and dest.read_bytes() != data:
            raise RuntimeError(f'Refusing to overwrite different model asset: {dest}. Use a clean checkout.')
        dest.write_bytes(data)
    return target

def doctor(folder, require_render):
    import torch
    versions = {}
    for package, expected in EXPECTED.items():
        try:
            value = metadata.version(package)
        except metadata.PackageNotFoundError:
            if package == 'musclemimic_models' and folder:
                value = 'external assets'
            else:
                raise RuntimeError(f'Missing {package}; install requirements-runtime.txt')
        versions[package] = value
        if value != expected and value != 'external assets':
            raise RuntimeError(f'{package} {value}; supported version is {expected}')
    for module in ['isaaclab', 'isaaclab_newton', 'tensorboard']:
        if importlib.util.find_spec(module) is None:
            raise RuntimeError(f'Missing {module}; use the pinned Isaac Lab Python environment')
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA-capable NVIDIA GPU and working PyTorch CUDA installation are required')
    if require_render:
        import imageio_ffmpeg
        imageio_ffmpeg.get_ffmpeg_exe()
        if importlib.util.find_spec('PIL') is None:
            raise RuntimeError('Install Pillow for video rendering')
    return {'python': sys.version, 'packages': versions, 'gpu': torch.cuda.get_device_name(0),
            'model_assets': str(folder)}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['doctor', 'train', 'evaluate'])
    p.add_argument('--model-assets', type=Path)
    p.add_argument('--num-envs', type=int, default=4096)
    p.add_argument('--iterations', type=int, default=2000, help='Additional PPO updates for training')
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--checkpoint', type=Path, default=BEST / 'checkpoint.pt', help='Evaluation checkpoint')
    p.add_argument('--resume', type=Path, help='Explicitly resume training; omitted means fresh weights')
    p.add_argument('--preset', choices=['bootstrap', 'velocity'], default='bootstrap')
    p.add_argument('--eval-seconds', type=float, default=20.)
    p.add_argument('--first-eval-seconds', type=float, default=300.)
    p.add_argument('--eval-interval', type=float, default=3600.)
    p.add_argument('--video-interval', type=float, default=3600.)
    p.add_argument('--no-render', action='store_true')
    p.add_argument('--skin-observations', action='store_true', help='Use coarse whole-body touch inputs; checkpoint layout must match')
    a = p.parse_args()
    if a.num_envs < 4 or a.num_envs % 4:
        p.error('--num-envs must be a positive multiple of four (one group per evaluation command)')
    if a.iterations < 1 or a.eval_seconds <= 0:
        p.error('--iterations and --eval-seconds must be positive')
    if a.mode != 'train' and a.resume:
        p.error('--resume is for train; use --checkpoint for evaluate')
    folder = model_assets(a.model_assets)
    info = doctor(folder, not a.no_render)
    print(json.dumps(info, indent=2), flush=True)
    if a.mode == 'doctor':
        return
    run = (a.run_dir or ROOT / 'outputs' / a.mode).resolve()
    if run.exists() and any(run.iterdir()):
        p.error(f'Run directory is not empty: {run}; choose a new --run-dir')
    checkpoint = a.checkpoint if a.mode == 'evaluate' else a.resume
    if checkpoint and not checkpoint.is_file():
        p.error(f'Checkpoint does not exist: {checkpoint}')
    prepare_assets(folder)
    command = [sys.executable, str(CODE / 'train_velocity_curriculum.py'),
               '--run-dir', str(run), '--num-envs', str(a.num_envs), '--solver-iterations', '20',
               '--phase-harmonics', '4', '--phase-frequency', '0.8928571428571429',
               '--support', '0', '--root-assistance', '0', '--stochastic-eval', '--eval-phase', 'random',
               '--eval-velocity-perturbation', '.02', '--eval-seconds', str(a.eval_seconds)]
    if checkpoint:
        command += ['--resume', str(checkpoint.resolve())]
    if a.mode == 'evaluate':
        command += ['--eval-only']
    else:
        bootstrap = a.preset == 'bootstrap'
        command += ['--iterations', str(a.iterations), '--desired-kl', '.2', '--trunk-height-weight', '5',
                    '--cartesian-positive-score', '--cartesian-targets',
                    str(ROOT / 'artifacts/reference/periodic_cartesian_fullcycle_targets.npz'),
                    '--cartesian-weight', '17.5' if bootstrap else '8.75',
                    '--reference-states', str(ROOT / 'artifacts/reference/projected_cycle_walking_states.npz'),
                    '--reference-reset-probability', '.8' if bootstrap else '.5',
                    '--command-min', '1' if bootstrap else '.4', '--command-max', '1' if bootstrap else '1.2',
                    '--standing-probability', '0' if bootstrap else '.3',
                    '--episode-seconds', '10' if bootstrap else '20',
                    '--first-eval-seconds', str(a.first_eval_seconds), '--eval-interval', str(a.eval_interval),
                    '--video-interval', str(a.video_interval), '--eval-at-end']
    if a.no_render:
        command += ['--no-render']
    if a.skin_observations:
        command += ['--skin-observations']
    run.mkdir(parents=True, exist_ok=True)
    (run / 'launch.json').write_text(json.dumps({'environment': info, 'command': command}, indent=2) + '\n')
    subprocess.run(command, cwd=ROOT, env={**os.environ, 'PYTHONUNBUFFERED': '1', 'MUJOCO_GL': 'egl'}, check=True)

if __name__ == '__main__':
    try:
        main()
    except RuntimeError as exc:
        sys.exit(str(exc))
