"""Check dominant state-based rewards on projected references versus standing.

This deliberately excludes contact/event/action penalties and does not establish
full return preference or physical realizability of the reference trajectory.
"""
from pathlib import Path
import json
import mujoco,numpy as np,torch
from periodic_pose_reward import PeriodicPoseReward
ROOT=Path(__file__).resolve().parents[2];base=ROOT/'outputs/isaac_velocity_g1';folder=base/'reference_candidates'
m=mujoco.MjModel.from_xml_path(str(ROOT/'outputs/isaac_velocity/assets/model.xml'));d=mujoco.MjData(m)
initial=np.load(ROOT/'outputs/isaac_velocity/assets/initial.npz')['qpos'];r=np.load(folder/'projected_cycle_walking_states.npz');targets=np.load(folder/'periodic_trunk_leg_targets.npz')
ids=[int(m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)]) for n in targets['joint_names']];pose=PeriodicPoseReward(targets['coefficients'],initial[ids]);phase=torch.tensor(r['phase'],dtype=torch.float32);command=torch.ones(len(phase))
head=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'head');pelvis=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'pelvis');d.qpos[:]=initial;mujoco.mj_forward(m,d);gap0=float(d.xpos[head,2]-d.xpos[pelvis,2]);values={}
for label,qs,vs in [('standing',np.tile(initial,(len(phase),1)),np.zeros_like(r['qvel'])),('reference',r['qpos'],r['qvel'])]:
 gaps=[]
 for q in qs:
  d.qpos[:]=q;mujoco.mj_forward(m,d);gaps.append(d.xpos[head,2]-d.xpos[pelvis,2])
 gaps=np.array(gaps);quat=qs[:,3:7];yaw=np.arctan2(2*(quat[:,0]*quat[:,3]+quat[:,1]*quat[:,2]),1-2*(quat[:,2]**2+quat[:,3]**2));q0=initial[3:7];yaw0=np.arctan2(2*(q0[0]*q0[3]+q0[1]*q0[2]),1-2*(q0[2]**2+q0[3]**2));heading=yaw-yaw0
 forward=-np.cos(heading)*vs[:,0]-np.sin(heading)*vs[:,1];lateral=np.sin(heading)*vs[:,0]-np.cos(heading)*vs[:,1]
 pose_rate=10*pose(torch.tensor(qs[:,ids],dtype=torch.float32),phase,command).numpy();height_rate=5*np.exp(-((gaps/gap0-1)/.15)**2);velocity_rate=2*np.exp(-((forward-1)**2+lateral**2)/.25)
 total=pose_rate+height_rate+velocity_rate
 values[label]=dict(mean_pose_reward_per_second=float(pose_rate.mean()),mean_trunk_reward_per_second=float(height_rate.mean()),mean_velocity_reward_per_second=float(velocity_rate.mean()),mean_dominant_sum_per_second=float(total.mean()),minimum_head_gap=float(gaps.min()),trunk_failure_fraction=float((gaps<.6*gap0).mean()))
result=dict(conditions=dict(command_mps=1,pose_weight=10,trunk_height_weight=5),results=values,caveat=__doc__)
(base/'reference_reward_preference.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
