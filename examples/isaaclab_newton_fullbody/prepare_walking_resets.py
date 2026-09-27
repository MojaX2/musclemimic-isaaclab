"""Prepare reference-state initialization, not a policy or a motion controller.

Phase is interpolated between alternating liftoffs. Frames outside two measured
liftoffs are excluded rather than assigning an arbitrary extrapolated phase.
The output retains original MuJoCo joint ordering; consumers must map by name.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def prepare(source, model_path, initial_path, output):
    r=np.load(source,allow_pickle=False)
    m=mujoco.MjModel.from_xml_path(str(model_path));d=mujoco.MjData(m)
    names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(m.njnt)]
    if names!=r['joint_names'].tolist():raise ValueError('Reference joint order differs from model')
    q=r['qpos'];v=r['qvel'];frequency=float(r['frequency'])
    feet=[[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,s+n)
           for n in ['_foot_col1','_foot_col3','_foot_col4','_bofoot_col1','_bofoot_col2']]
          for s in ['r','l']]
    if min(sum(feet,[]))<0:raise ValueError('Missing foot collision geometry')
    heights=[]
    for pose in q:
        d.qpos[:]=pose;mujoco.mj_forward(m,d);height=[]
        for ids in feet:
            row=d.geom_xmat[ids].reshape(-1,3,3)[:,2,:];size=m.geom_size[ids]
            capsule=m.geom_type[ids]==int(mujoco.mjtGeom.mjGEOM_CAPSULE)
            extent=np.where(capsule,size[:,0]+abs(row[:,2])*size[:,1],
                            np.sqrt(((row*size)**2).sum(1)))
            height.append(np.min(d.geom_xpos[ids,2]-extent))
        heights.append(height)
    heights=np.asarray(heights);air=heights>.005;events=[]
    for side in range(2):
        for start in np.flatnonzero(air[1:,side]&~air[:-1,side])+1:
            end=start
            while end<len(q) and air[end,side]:end+=1
            if (end-start)/frequency>=.12 and heights[start:end,side].max()>=.04:
                events.append((int(start),side))
    events.sort();phase=np.full(len(q),np.nan)
    for (start,side),(end,next_side) in zip(events,events[1:]):
        if side==next_side or not .2<=(end-start)/frequency<=.8:continue
        phase[start:end]=side*np.pi+np.arange(end-start)/(end-start)*np.pi
    initial=np.load(initial_path)['qpos'];base=initial[3:7]
    def yaw(wxyz):
        w,x,y,z=np.moveaxis(wxyz,-1,0)
        return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
    heading=yaw(q[:,3:7])-yaw(base)
    command=-np.cos(heading)*v[:,0]-np.sin(heading)*v[:,1]
    keep=np.isfinite(phase)&np.isfinite(q).all(1)&np.isfinite(v).all(1)&(command>=.3)&(command<=1.2)
    ids=np.flatnonzero(keep)
    if not len(ids):raise ValueError('No phase-aligned moving frames')
    poses=q[ids].copy();poses[:,:2]=initial[:2]
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,qpos=poses,qvel=v[ids],phase=phase[ids]%(2*np.pi),
                        command=command[ids],joint_names=r['joint_names'],source_frame=ids)
    report={'source':str(source),'frames':len(ids),'liftoffs_frame_side':events,
            'command_range_mps':[float(command[ids].min()),float(command[ids].max())],
            'phase_definition':'Right liftoff=0, left liftoff=pi; interpolate between alternating measured liftoffs',
            'limitations':'Kinematic reset bank, no muscle activation reference. Newton validation and standing-start evaluation still required.'}
    output.with_suffix('.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(prepare(a.source,a.model,a.initial,a.output)))
