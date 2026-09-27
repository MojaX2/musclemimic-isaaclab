"""Render a training evaluation in an isolated CPU process."""
from pathlib import Path
import subprocess,sys,json
folder=Path(sys.argv[1])
metrics=json.loads((folder/'metrics.json').read_text())
label='Isaac Lab | learned velocity policy'
if metrics.get('initial_state')=='reference':label='Isaac Lab | reference-start diagnostic'
subprocess.run([sys.executable,str(Path(__file__).with_name('render_policy_comparison.py')),str(folder),'--output',str(folder/'policy.mp4'),'--left-label',label],check=True)

subprocess.run([sys.executable,str(Path(__file__).with_name("analyze_velocity_rollout.py")),str(folder)],check=True)
