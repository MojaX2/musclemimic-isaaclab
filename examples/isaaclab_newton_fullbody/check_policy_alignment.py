"""Check the saved first transition against the original MJX rollout."""
import argparse
import json
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('reference',type=Path)
parser.add_argument('candidate',type=Path)
parser.add_argument('--output',type=Path)
args=parser.parse_args()
report={}
for phase in ['initial','first_step']:
    reference=np.load(args.reference/f'{phase}.npz')
    candidate=np.load(args.candidate/f'{phase}.npz')
    metrics={}
    for field in ['qpos','qvel','act','action','obs','actuator_force','sensordata']:
        if field not in reference or field not in candidate:continue
        difference=np.asarray(reference[field])-np.asarray(candidate[field])
        assert np.isfinite(difference).all(),(phase,field)
        metrics[field]={'max_abs':float(np.max(np.abs(difference))), 'rms':float(np.sqrt(np.mean(difference**2)))}
    report[phase]=metrics
assert report['initial']['qpos']['max_abs']==0
assert report['initial']['qvel']['max_abs']==0
assert report['first_step']['qpos']['max_abs']<1e-4
assert report['first_step']['qvel']['max_abs']<1e-3
assert report['first_step']['action']['max_abs']<1e-3
assert report['first_step']['obs']['rms']<1e-2
if args.output:args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
