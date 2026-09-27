"""Measure recorded foot motion; contact geometry alone does not prove walking."""
import json
import sys
from pathlib import Path
import mujoco
import numpy as np

folder=Path(sys.argv[1])
r=np.load(folder/'rollout.npz')
m=mujoco.MjModel.from_xml_path(str(folder/'model.xml'));d=mujoco.MjData(m)
feet=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'calcn_'+side) for side in ['r','l']]
foot_bodies=[{mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name+'_'+side) for name in ['calcn','toes']} for side in ['r','l']]
head=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'head')
pelvis=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'pelvis')
positions=[];contacts=[];head_gaps=[]
for q in r['qpos']:
    d.qpos[:]=q;mujoco.mj_forward(m,d)
    positions.append(d.xpos[feet].copy())
    if head>=0 and pelvis>=0:head_gaps.append(float(d.xpos[head,2]-d.xpos[pelvis,2]))
    touching=[False,False]
    for c in d.contact:
        bodies=set(m.geom_bodyid[c.geom])
        if c.dist<=.002 and 0 in bodies:
            for j,ids in enumerate(foot_bodies):touching[j]|=bool(ids&bodies)
    contacts.append(touching)
positions=np.asarray(positions);contacts=np.asarray(contacts)
reset=r['reset'].astype(bool) if 'reset' in r else None
transitions=(~contacts[:-1])&contacts[1:]
if reset is not None:transitions[reset[1:]]=False
summary={
    'seconds':len(positions)*float(r['dt']),
    'reset_count':int(reset.sum()) if reset is not None else None,
    'contact_fraction_right_left':contacts.mean(0).tolist(),
    'touchdown_count_right_left':transitions.sum(0).tolist(),
    'foot_forward_separation_range_m':float(np.ptp(positions[:,0,0]-positions[:,1,0])),
    'foot_vertical_range_right_left_m':np.ptp(positions[:,:,2],axis=0).tolist(),
    'geometric_single_support_fraction':float((contacts.sum(1)==1).mean()),
    'geometric_flight_fraction':float((contacts.sum(1)==0).mean()),
    'caveat':'Recomputed geometric contacts, not contact forces. Touchdowns may include stumbling; inspect video and survival metrics. Older recordings without reset markers cannot exclude reset-induced transitions.',
}
if head_gaps:
    summary['head_above_pelvis_m']={'minimum':float(np.min(head_gaps)),
        'median':float(np.median(head_gaps)),
        'fraction_below_0_36m':float((np.asarray(head_gaps)<.36).mean())}
if 'command' in r and 'velocity' in r:
    valid=~reset if reset is not None else np.ones(len(positions),dtype=bool)
    summary['recorded_command_mean_mps']=float(r['command'].mean())
    summary['recorded_velocity_mae_mps']=float(np.abs(r['velocity'][valid]-r['command'][valid]).mean())
    summary['recorded_velocity_mean_mps']=float(r['velocity'][valid].mean())
if 'foot_clearance' in r:
    # Count complete swings only; reset jumps and initial contact are not steps.
    dt=float(r['dt']);heights=r['foot_clearance']
    air=np.zeros(2);peak=np.zeros(2);seen_stance=np.zeros(2,dtype=bool)
    events=[];chain=0;longest=0;last_side=None;last_time=None
    for i,touching in enumerate(contacts):
        if reset is not None and reset[i]:
            air[:]=0;peak[:]=0;seen_stance[:]=False
            chain=0;last_side=None;last_time=None
            continue
        landed=touching & (air>0)
        valid_step=landed & seen_stance & (air>=.12-1e-6) & (air<=.6+1e-6) & (peak>=.04)
        if landed.sum()==1 and valid_step.any():
            side=int(np.flatnonzero(valid_step)[0]);now=i*dt
            chain=chain+1 if last_side is not None and side!=last_side and now-last_time<=1.2 else 1
            longest=max(longest,chain);last_side=side;last_time=now
            events.append({'time_seconds':now,'side':'right' if side==0 else 'left',
                           'air_seconds':float(air[side]),'peak_clearance_m':float(peak[side])})
        seen_stance |= touching
        air=np.where(touching,0.,air+dt)
        peak=np.where(touching,0.,np.maximum(peak,heights[i]))
    summary['qualified_swing_landings']=events
    summary['longest_alternating_landing_chain']=longest
    summary['qualified_swing_definition']='Observed stance then 0.12–0.60 s airborne, clearance >=0.04 m, unilateral landing; chain gap <=1.2 s. 25 Hz geometric-contact estimate, not force-contact ground truth.'
(folder/'foot_motion.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary))
