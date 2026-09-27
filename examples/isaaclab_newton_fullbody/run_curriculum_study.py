"""Bounded, sequential GPU comparison; no silent restart or automatic winner."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser()
p.add_argument('--study-dir',type=Path,required=True)
p.add_argument('--wait-run',type=Path)
p.add_argument('--dry-run',action='store_true')
a=p.parse_args()
trainer=Path(__file__).with_name('train_velocity_curriculum.py')
checkpoint=ROOT/'outputs/isaac_velocity_g1/run_005/stopped.pt'
cases=[('warm_no_support',0.,True),('warm_support30',.3,True),('fresh_support30',.3,False)]
commands=[]
for name,support,warm in cases:
    cmd=['/home/k_miyazawa/IsaacLab/.venv/bin/python',str(trainer),
         '--num-envs','1024','--iterations','250','--run-dir',str(a.study_dir/name),
         '--support',str(support),'--command-min','.4','--command-max','.4',
         '--standing-probability','0','--reset-velocity-noise','.01',
         '--stochastic-eval','--eval-phase','random','--eval-velocity-perturbation','.02',
         '--eval-support','0','--eval-at-end','--first-eval-seconds','1000000000']
    if warm:cmd+=['--resume',str(checkpoint)]
    commands.append(cmd)
if a.dry_run:
    print(json.dumps(commands,indent=2));raise SystemExit
a.study_dir.mkdir(parents=True,exist_ok=False)
snapshot=a.study_dir/'source_snapshot';snapshot.mkdir()
hashes={}
for source in trainer.parent.glob('*.py'):
    shutil.copy2(source,snapshot/source.name)
    hashes[source.name]=hashlib.sha256(source.read_bytes()).hexdigest()
(snapshot/'manifest.json').write_text(json.dumps(hashes,indent=2))
status={'state':'waiting','cases':[],'created':datetime.datetime.now().astimezone().isoformat()}
def save():
    tmp=a.study_dir/'status.json.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(a.study_dir/'status.json')
save()
if a.wait_run:
    meta=json.loads((a.wait_run/'process.json').read_text())
    while True:
        try:
            stat=Path(f"/proc/{meta['pid']}/stat").read_text().split()
            live=stat[21]==meta['proc_start_ticks'] and stat[2] not in ('Z','X')
        except FileNotFoundError:live=False
        if not live:break
        if (a.study_dir/'STOP').exists():status['state']='stopped';save();raise SystemExit
        time.sleep(10)
    # A missing handle alone is not evidence that the requested evaluation saved.
    if not (a.wait_run/'stopped.pt').exists():
        status['state']='error';status['error']='Preceding run exited without stopped.pt';save();raise SystemExit(1)
for (name,support,warm),cmd in zip(cases,commands):
    if (a.study_dir/'STOP').exists():break
    run=a.study_dir/name
    item={'name':name,'support':support,'warm_start':warm,'args':cmd,'state':'running'}
    status['cases'].append(item);status['state']='running';status['active_case']=name
    with (a.study_dir/(name+'.log')).open('x') as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,env={**os.environ,'PYTHONUNBUFFERED':'1','MPLCONFIGDIR':'/tmp/muscle-g1-mpl'},stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        item['pid']=proc.pid;item['proc_start_ticks']=Path(f'/proc/{proc.pid}/stat').read_text().split()[21];save()
        while proc.poll() is None:
            if (a.study_dir/'STOP').exists() and run.exists():
                (run/'STOP').touch()
            time.sleep(10)
    item['exit_code']=proc.returncode
    item['state']='finished' if proc.returncode==0 else 'error'
    item['evaluations']=[json.loads(f.read_text()) for f in sorted(run.glob('evaluations/*/metrics.json'))]
    save()
    if proc.returncode!=0:
        status['state']='error';save();raise SystemExit(proc.returncode)
status['state']='stopped' if (a.study_dir/'STOP').exists() else 'finished'
status.pop('active_case',None);save()
