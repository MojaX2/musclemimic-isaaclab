"""Read-only, compact training/process inspection; run in the host PID namespace."""
import argparse
import json
import math
import time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('run_dir',type=Path)
a=p.parse_args()
out={'run_dir':str(a.run_dir.resolve())}
meta=json.loads((a.run_dir/'process.json').read_text())
proc=Path('/proc')/str(meta['pid'])
try:
    stat=(proc/'stat').read_text().split()
    command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    out['process']={'pid':meta['pid'],'state':stat[2],
        'same_start_time':stat[21]==meta['proc_start_ticks'],
        'training_command':any(name in command for name in ('train_velocity_g1.py','train_velocity_curriculum.py'))}
    out['process']['live']=out['process']['same_start_time'] and out['process']['training_command'] and stat[2] not in ('Z','X')
except FileNotFoundError:
    out['process']={'pid':meta['pid'],'live':False,'reason':'PID absent in this namespace'}
metrics=a.run_dir/'metrics.jsonl'
rows=[]
if metrics.exists():
    for line in metrics.read_text().splitlines():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:pass  # Concurrent writer may have a partial last line.
if rows:
    keys=['iteration','wall_seconds','training_episodes_this_run','exploration_std_cap',
          'exploration_std_mean','episode_seconds','episode_return','velocity_error',
          'head_height_gap','torso_collapse_fraction','single_support_fraction',
          'reward/swing_clearance','reward/phase_clearance','reward/touchdown','reward/prolonged_double_support','nonfinite']
    out['latest']={k:rows[-1].get(k) for k in keys}
    out['log_age_seconds']=round(time.time()-metrics.stat().st_mtime,1)
    out['recent_episode_seconds']=[round(r['episode_seconds'],3) for r in rows[-6:]]
    out['recent_nonfinite_state_count']=sum(r.get('nonfinite',0) for r in rows[-6:])
    out['recent_nonfinite_losses']=any(not math.isfinite(v) for r in rows[-6:] for v in r.get('loss',{}).values())
evals=sorted(a.run_dir.glob('evaluations/*/metrics.json'),key=lambda f:f.stat().st_mtime)
if evals:
    out['evaluation']=json.loads(evals[-1].read_text())
    out['evaluation_age_seconds']=round(time.time()-evals[-1].stat().st_mtime,1)
    foot=evals[-1].with_name('foot_motion.json')
    if foot.exists():out['foot_motion']=json.loads(foot.read_text())
out['stop_requested']=(a.run_dir/'STOP').exists()
out['checkpoint_exists']=(a.run_dir/'latest.pt').exists()
print(json.dumps(out,indent=2))
