"""Build smooth pelvis-relative site targets from a full projected gait cycle."""
import json
from pathlib import Path
import mujoco,numpy as np,torch
from cartesian_reference_reward import aligned_relative_sites,CartesianReferenceReward
from project_joint_equalities import project_joint_equalities
root=Path(__file__).resolve().parents[2];folder=root/'outputs/isaac_velocity_g1/reference_candidates'
m=mujoco.MjModel.from_xml_path(str(root/'outputs/isaac_velocity/assets/model.xml'));d=mujoco.MjData(m)
initial=np.load(root/'outputs/isaac_velocity/assets/initial.npz')['qpos'];bank=np.load(folder/'projected_cycle_walking_states.npz')
source=Path('/home/k_miyazawa/.musclemimic/caches/AMASS/MyoFullBody/gmr/KIT/314/walking_medium09_poses.npz');raw=np.load(source)
assert raw['joint_names'].tolist()==[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j) for j in range(m.njnt)]
frames=np.arange(443,555);raw_q=raw['qpos'][frames].copy();raw_q[:,:2]=initial[:2]
q,v=project_joint_equalities(m,raw_q,raw['qvel'][frames])
retained=bank['source_frame']-443
assert np.max(np.abs(q[retained]-bank['qpos']))<1e-5 and np.max(np.abs(v[retained]-bank['qvel']))<1e-4
names=['upper_body_mimic','head_mimic']+[f'{side}_{part}_mimic' for side in ('left','right') for part in ('shoulder','elbow','hand','hip','knee','ankle','toes')]
ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,n) for n in names];pelvis=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'pelvis_mimic');assert min(ids+[pelvis])>=0
quat=initial[3:7];yaw0=np.arctan2(2*(quat[0]*quat[3]+quat[1]*quat[2]),1-2*(quat[2]**2+quat[3]**2))
def sites(qs):
 xyz=[];roots=[];heading=[]
 for q in qs:
  d.qpos[:]=q;mujoco.mj_kinematics(m,d);xyz.append(d.site_xpos[ids].copy());roots.append(d.site_xpos[pelvis].copy())
  quat=q[3:7];heading.append(np.arctan2(2*(quat[0]*quat[3]+quat[1]*quat[2]),1-2*(quat[2]**2+quat[3]**2))-yaw0)
 return aligned_relative_sites(torch.tensor(np.array(xyz)),torch.tensor(np.array(roots)),torch.tensor(heading)).numpy()
reference=sites(q);standing=sites(initial[None])[0];phase=np.interp(frames,[443,500,555],[0,np.pi,2*np.pi]);features=np.stack([np.ones_like(phase)]+[f(k*phase) for k in range(1,5) for f in (np.cos,np.sin)],axis=-1)
coeff=np.linalg.lstsq(features,reference.reshape(len(reference),-1),rcond=None)[0].reshape(9,len(names),3)
reward=CartesianReferenceReward(coeff,standing);ph=torch.tensor(phase,dtype=torch.float32);target=reward.target(ph).detach().numpy();error=reference-target
path=folder/'periodic_cartesian_fullcycle_targets.npz';np.savez_compressed(path,coefficients=coeff,standing_sites=standing,site_names=np.array(names),phase_frequency=np.array(100/112),sigma_m=np.array(.1),initial_yaw=np.array(yaw0))
result=dict(path=str(path),source=str(source),source_frame_range=[443,555],frames=len(phase),sites=names,fit_rmse_m=float(np.sqrt(np.mean(error**2))),maximum_site_error_m=float(np.linalg.norm(error,axis=-1).max()),coordinates='pelvis-relative sites rotated by negative root yaw delta from upright initial; world vertical preserved',reward='exp(-mean squared position error / 0.1^2), minus phase-matched standing score; gated off for zero command',training_integrated=False,walking_success=False,ready_for_training=False,review_reason="Full cycle fitted; inspect approximation error and native-physics behavior before integration.")
site_error=np.linalg.norm(error,axis=-1);feet=[i for i,name in enumerate(names) if 'ankle' in name or 'toes' in name]
seam_phase=torch.tensor([0.,2*np.pi],requires_grad=True);seam_target=reward.target(seam_phase)
seam_target.sum().backward()
result.update(maximum_foot_site_error_m=float(site_error[:,feet].max()),per_site_max_error_m=dict(zip(names,site_error.max(0).tolist())),cycle_seam_max_m=float((seam_target[0]-seam_target[1]).detach().abs().max()),cycle_derivative_seam_error=float((seam_phase.grad[0]-seam_phase.grad[1]).abs()))
path.with_suffix('.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
