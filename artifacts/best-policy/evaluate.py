"""Prepare portable assets and evaluate the archived checkpoint (no learning)."""
import argparse
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--model-assets', type=Path, required=True,
               help='musclemimic_models/model directory containing meshes/ and scene/')
p.add_argument('--num-envs', type=int, default=4096)
p.add_argument('--run-dir', type=Path, default=Path('outputs/best_policy_eval'))
p.add_argument('--prepare-only', action='store_true')
p.add_argument('--eval-seconds', type=float, default=20.)
a = p.parse_args()
here = Path(__file__).resolve().parent
root = here.parents[1]
assets = a.model_assets.resolve()
if not (assets / 'meshes').is_dir() or not (assets / 'scene').is_dir():
    p.error('--model-assets must contain meshes/ and scene/')
out = root / 'outputs/isaac_velocity/assets'
out.mkdir(parents=True, exist_ok=True)
content = (here / 'model.xml.template').read_text().replace('__MODEL_ASSET_ROOT__', assets.as_posix())
for name, data in [('model.xml', content.encode()), ('initial.npz', (here / 'initial.npz').read_bytes())]:
    target = out / name
    if target.exists() and target.read_bytes() != data:
        p.error(f'Refusing to replace different existing asset: {target}; use a clean checkout')
    target.write_bytes(data)
if a.prepare_only:
    print(out)
    sys.exit(0)
cmd = [sys.executable, str(root / 'examples/isaaclab_newton_fullbody/train_velocity_curriculum.py'),
       '--eval-only', '--no-render', '--resume', str(here / 'checkpoint.pt'), '--run-dir', str(a.run_dir.resolve()),
       '--num-envs', str(a.num_envs), '--solver-iterations', '20', '--phase-harmonics', '4',
       '--phase-frequency', '0.8928571428571429', '--support', '0', '--root-assistance', '0',
       '--eval-seconds', str(a.eval_seconds), '--stochastic-eval', '--eval-phase', 'random',
       '--eval-velocity-perturbation', '.02', '--command-min', '.4', '--command-max', '1.2']
# No reference-reset bank or imitation target is needed for upright-start inference.
subprocess.run(cmd, cwd=root, check=True)
