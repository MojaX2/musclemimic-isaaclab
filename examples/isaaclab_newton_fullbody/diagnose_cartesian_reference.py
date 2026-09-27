"""Offline reward discrimination on reference, standing, and recorded Isaac states.

Does not change training or establish physical realizability. Cartesian site
positions follow the original MimicReward's pelvis-relative world coordinates.
"""
import argparse,json
from pathlib import Path
import mujoco,numpy as np
p=argparse.ArgumentParser();p.add_argument('rollout',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[2];base=root/'outputs/isaac_velocity_g1'
m=mujoco.MjModel.from_xml_path(str(root/'outputs/isaac_velocity/assets/model.xml'));d=mujoco.MjData(m)
names=['upper_body_mimic','head_mimic']+[f'{side}_{part}_mimic' for side in ('left','right') for part in ('shoulder','elbow','hand','hip','knee','ankle','toes')]
ids=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,n) for n in names]);pelvis=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'pelvis_mimic');assert np.all(ids>=0) and pelvis>=0
feet=np.array([i for i,n in enumerate(names) if 'ankle' in n or 'toes' in n])
def positions(qs):
 out=[]
 for q in qs:
  d.qpos[:]=q;mujoco.mj_kinematics(m,d);out.append(d.site_xpos[ids]-d.site_xpos[pelvis])
 return np.array(out)
def features(phase):
 return np.stack([np.ones_like(phase)]+[f(k*phase) for k in range(1,5) for f in (np.cos,np.sin)],axis=-1)
bank=np.load(base/'reference_candidates/projected_cycle_walking_states.npz');ref=positions(bank['qpos']);design=features(bank['phase']);coeff=np.linalg.lstsq(design,ref.reshape(len(ref),-1),rcond=None)[0]
standing=positions(np.load(root/'outputs/isaac_velocity/assets/initial.npz')['qpos'][None])[0]
roll=np.load(a.rollout);recorded=positions(roll['qpos']);results={}
for label,xyz,phase in [('reference',ref,bank['phase']),('standing',np.broadcast_to(standing,ref.shape),bank['phase']),('recorded',recorded,roll['phase'])]:
 target=(features(phase)@coeff).reshape(xyz.shape);error=(xyz-target)**2
 results[label]=dict(cartesian_rmse_m=float(np.sqrt(error.mean())),mean_source_position_score=float(np.exp(-100*error.mean((1,2))).mean()),mean_foot_position_score=float(np.exp(-100*error[:,feet].mean((1,2))).mean()),foot_rmse_m=float(np.sqrt(error[:,feet].mean())))
result=dict(rollout=str(a.rollout.resolve()),results=results,sites=names,source='musclemimic/core/reward/trajectory_based.py: rpos reward exp(-100*mean squared relative site error)',caveat='Offline position term only; recorded episodes include resets/falls, different velocity commands and phase distributions. Not a full-return comparison or proof of achievable gait. Four-harmonic targets fit the projected reference bank.')
a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
