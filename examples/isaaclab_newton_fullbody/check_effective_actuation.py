"""Static, equality-reduced muscle authority diagnostic; not a gait controller."""
from pathlib import Path
import json
import mujoco,numpy as np
from scipy.optimize import lsq_linear
from project_joint_equalities import project_joint_equalities
ROOT=Path(__file__).resolve().parents[2];base=ROOT/'outputs/isaac_velocity_g1'
m=mujoco.MjModel.from_xml_path(str(ROOT/'outputs/isaac_velocity/assets/model.xml'));d=mujoco.MjData(m)
initial=np.load(ROOT/'outputs/isaac_velocity/assets/initial.npz')['qpos']
bank=np.load(base/'reference_candidates/projected_cycle_walking_states.npz')
dependent={int(m.eq_obj1id[e]) for e in range(m.neq) if m.eq_active0[e] and int(m.eq_type[e])==int(mujoco.mjtEq.mjEQ_JOINT)}
independent=[j for j in range(1,m.njnt) if j not in dependent]
dofs=[int(m.jnt_dofadr[j]) for j in independent]
results=[]
for label,q in [('standing',initial),('reference_midcycle',bank['qpos'][len(bank['qpos'])//2])]:
 q,_=project_joint_equalities(m,q,np.zeros(m.nv));mujoco.mj_resetData(m,d);d.qpos[:]=q;mujoco.mj_forward(m,d)
 eye=np.zeros((len(dofs),m.nv));eye[np.arange(len(dofs)),dofs]=1
 _,tangent=project_joint_equalities(m,np.tile(q,(len(dofs),1)),eye)
 moment=np.zeros((m.nu,m.nv))
 for a in range(m.nu):
  start=int(d.moment_rowadr[a]);end=start+int(d.moment_rownnz[a]);moment[a,d.moment_colind[start:end]]=d.actuator_moment[start:end]
 d.act[:]=0;mujoco.mj_fwdActuation(m,d);f0=d.actuator_force.copy();base_force=d.qfrc_actuator.copy()
 np.testing.assert_allclose(moment.T@f0,base_force,atol=1e-9)
 d.act[:]=1;mujoco.mj_fwdActuation(m,d);gain=d.actuator_force.copy()-f0
 d.act[:]=.5;mujoco.mj_fwdActuation(m,d);np.testing.assert_allclose(d.actuator_force,f0+.5*gain,atol=1e-8)
 A=tangent@(moment.T*gain[None,:]);target=tangent@(d.qfrc_bias-d.qfrc_passive-base_force)
 scale=np.maximum(np.linalg.norm(A,axis=1),1.)
 sol=lsq_linear(A/scale[:,None],target/scale,bounds=(0,1),method='trf',lsq_solver='lsmr',tol=1e-8,max_iter=100)
 err=A@sol.x-target
 results.append(dict(pose=label,independent_joint_count=len(dofs),effective_rank=int(np.linalg.matrix_rank(A,tol=np.linalg.svd(A,compute_uv=False)[0]*1e-6)),optimizer_success=bool(sol.success),target_norm=float(np.linalg.norm(target)),residual_norm=float(np.linalg.norm(err)),relative_residual=float(np.linalg.norm(err)/max(np.linalg.norm(target),1e-9)),max_abs_joint_residual=float(np.abs(err).max()),activation_min=float(sol.x.min()),activation_max=float(sol.x.max()),activation_mean=float(sol.x.mean())))
out=dict(independent_joints=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j) for j in independent],results=results,limitations='Static zero-velocity internal gravity/passive load balancing only. Root balance/contact forces, activation dynamics and walking are not solved. Effective Jacobian is tangent to active joint equalities; this does not prove dynamic controllability.')
(base/'effective_actuation_diagnostic.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
