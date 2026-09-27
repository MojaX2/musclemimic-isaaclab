"""Fit one reference gait cycle for leg and trunk imitation reward guidance."""
import json
from pathlib import Path
import mujoco
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
folder=ROOT/'outputs/isaac_velocity_g1/reference_candidates'
source=Path('/home/k_miyazawa/.musclemimic/caches/AMASS/MyoFullBody/gmr/KIT/314/walking_medium09_poses.npz')
r=np.load(source);m=mujoco.MjModel.from_xml_path(str(ROOT/'outputs/isaac_velocity/assets/model.xml'))
names=['flex_extension','lat_bending','axial_rotation']+[f'{joint}_{side}' for side in ['r','l'] for joint in ['hip_flexion','hip_adduction','hip_rotation','knee_angle','ankle_angle','subtalar_angle']]
ids=[m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in names]
frames=np.arange(443,555);phase=np.interp(frames,[443,500,555],[0,np.pi,2*np.pi])
features=np.stack([np.ones_like(phase)]+[f(k*phase) for k in range(1,5) for f in [np.cos,np.sin]],axis=-1)
q=r['qpos'][frames][:,ids];coef=np.linalg.lstsq(features,q,rcond=None)[0]
frequency=100/112
np.savez_compressed(folder/'periodic_trunk_leg_targets.npz',coefficients=coef,joint_names=names,frequency=frequency,harmonics=4)
bank=np.load(folder/'aligned_walking_states.npz');selected=(bank['source_frame']>=443)&(bank['source_frame']<555)
reset={k:bank[k] if k=='joint_names' else bank[k][selected] for k in bank.files};reset['command']=np.ones(selected.sum())
np.savez_compressed(folder/'cycle_walking_states.npz',**reset)
meta=dict(source=str(source),frames=[443,555],left_liftoff=500,joint_names=names,frequency_hz=frequency,fit_rmse_rad=np.sqrt(((features@coef-q)**2).mean(0)).tolist(),reset_states=int(selected.sum()),command_mps=1.,limitations='One approximately 1.1 m/s cycle, no claim of dynamic feasibility. Reference supplies reward targets/reset states only; muscle actuators remain native.')
(folder/'periodic_trunk_leg_targets.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
