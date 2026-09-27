"""Request one evaluation at a logged iteration of an existing live run.

Never launches or restarts training. Exits if the original process ends or a stop
is requested. Metrics are sampled every ten updates, so the trigger is approximate.
"""
import argparse
import json
import time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('run_dir',type=Path)
p.add_argument('--iteration',type=int,required=True);a=p.parse_args()
meta=json.loads((a.run_dir/'process.json').read_text())
while True:
    try:
        stat=Path(f"/proc/{meta['pid']}/stat").read_text().split()
    except FileNotFoundError:
        print('Original training process ended',flush=True);break
    if stat[21]!=meta['proc_start_ticks'] or stat[2] in ('Z','X'):
        print('Original training process ended',flush=True);break
    if (a.run_dir/'STOP').exists():
        print('Stop requested; no evaluation request added',flush=True);break
    status=a.run_dir/'status.json'
    try:
        current=json.loads(status.read_text())['iteration']
    except (FileNotFoundError,json.JSONDecodeError):
        time.sleep(10);continue
    if current>=a.iteration:
        (a.run_dir/'EVALUATE').touch()
        print(f'Requested evaluation at logged iteration {current}',flush=True)
        break
    time.sleep(10)
