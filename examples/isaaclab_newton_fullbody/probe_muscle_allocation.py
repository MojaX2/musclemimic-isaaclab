"""CPU-only muscle allocation probe, not a learned Isaac Lab walking policy.

A joint-space acceleration target is converted to bounded muscle excitations.
No generalized joint forces or kinematic state overwrite are applied in stepping.
Pelvis support, if enabled, is explicitly recorded as an external wrench.
"""
import argparse,json,time
from pathlib import Path
import mujoco,numpy as np
from scipy.optimize import lsq_linear
from project_joint_equalities import project_joint_equalities

p=argparse.ArgumentParser();p.add_argument('--solver-iterations',type=int,default=4);p.add_argument('--kp',type=float,default=100.);p.add_argument('--seconds',type=float,default=3.);p.add_argument('--support',type=float,default=1.);p.add_argument('--acceleration-feedback-gain',type=float,default=0.);p.add_argument('--trunk-priority',type=float,default=1.);p.add_argument('--initialize-activation',action='store_true');p.add_argument('--support-ramp-seconds',type=float,default=0.);p.add_argument('--walk-delay',type=float,default=0.);p.add_argument('--motion-ramp-seconds',type=float,default=0.);p.add_argument('--mode',choices=['standing','walking'],default='standing');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
ROOT=Path(__file__).resolve().parents[2];a.output.mkdir(parents=True,exist_ok=False)
m=mujoco.MjModel.from_xml_path(str(ROOT/'outputs/isaac_velocity/assets/model.xml'));m.opt.timestep=.002;m.opt.integrator=mujoco.mjtIntegrator.mjINT_EULER;m.opt.iterations=a.solver_iterations;m.opt.ls_iterations=8
d=mujoco.MjData(m);initial=np.load(ROOT/'outputs/isaac_velocity/assets/initial.npz')['qpos'];initial,_=project_joint_equalities(m,initial,np.zeros(m.nv));d.qpos[:]=initial;mujoco.mj_forward(m,d)
pelvis=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'pelvis');head=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'head');initial_gap=d.xpos[head,2]-d.xpos[pelvis,2];mass=float(m.body_mass.sum())
dependent={int(m.eq_obj1id[e]) for e in range(m.neq) if m.eq_active0[e] and int(m.eq_type[e])==int(mujoco.mjtEq.mjEQ_JOINT)}
independent=[j for j in range(1,m.njnt) if j not in dependent];qids=m.jnt_qposadr[independent];vids=m.jnt_dofadr[independent];n=len(independent)
basis=np.zeros((n,m.nv));basis[np.arange(n),vids]=1.;M=np.zeros((m.nv,m.nv));previous=np.zeros(m.nu)
reference=np.load(ROOT/'outputs/isaac_velocity_g1/reference_candidates/periodic_trunk_leg_targets.npz')
reference_indices=[independent.index(mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name)) for name in reference['joint_names']]
poses=[];acts=[];wrenches=[];speeds=[];commands=[];heights=[];gaps=[];residuals=[];successes=[];start=time.monotonic();terminated=False;balance_errors=[];allocation_trace=[];target_trace=[];error_trace=[];excitation_trace=[]
for step in range(round(a.seconds/.02)):
    _,T=project_joint_equalities(m,np.tile(d.qpos,(n,1)),basis)
    moment=np.zeros((m.nu,m.nv))
    for j in range(m.nu):
        k=int(d.moment_rowadr[j]);end=k+int(d.moment_rownnz[j]);moment[j,d.moment_colind[k:end]]=d.actuator_moment[k:end]
    original_actuator_force=d.qfrc_actuator.copy()
    act=d.act.copy();d.act[:]=0;mujoco.mj_fwdActuation(m,d);f0=d.actuator_force.copy();base=d.qfrc_actuator.copy()
    d.act[:]=1;mujoco.mj_fwdActuation(m,d);gain=d.actuator_force.copy()-f0
    d.act[:]=act;mujoco.mj_fwdActuation(m,d)
    mujoco.mj_fullM(m,M,d.qM);target=initial[qids].copy();target_v=np.zeros(n);command=0.
    if a.mode=='walking' and step*.02>=a.walk_delay:
        elapsed=step*.02-a.walk_delay
        phase=elapsed*2*np.pi*float(reference['frequency']);features=np.array([1.]+[f(k*phase) for k in range(1,5) for f in [np.cos,np.sin]])
        derivative=np.array([0.]+[x for k in range(1,5) for x in [-k*np.sin(k*phase),k*np.cos(k*phase)]])*2*np.pi*float(reference['frequency'])
        ramp=min(1.,elapsed/a.motion_ramp_seconds) if a.motion_ramp_seconds>0 else 1.
        blend=ramp*ramp*(3-2*ramp);blend_rate=6*ramp*(1-ramp)/a.motion_ramp_seconds if a.motion_ramp_seconds>0 else 0.
        refpose=features@reference['coefficients'];startpose=initial[qids][reference_indices]
        target[reference_indices]=startpose+blend*(refpose-startpose)
        target_v[reference_indices]=blend*(derivative@reference['coefficients'])+blend_rate*(refpose-startpose);command=blend
    qacc=(a.kp*(target-d.qpos[qids])+2*np.sqrt(a.kp)*(target_v-d.qvel[vids])).clip(-100,100)
    desired=(T@M@T.T)@qacc+T@(d.qfrc_bias-d.qfrc_passive-base)
    A=T@(moment.T*gain[None,:])
    balance=T@(M@d.qacc+d.qfrc_bias-d.qfrc_passive-d.qfrc_constraint-d.qfrc_actuator)
    balance_errors.append(float(np.max(np.abs(balance))))
    if balance_errors[-1]>=1e-5:
        diagnostic={'step':step,'time':step*.02,'original_balance_error':balance_errors[-1],'actuator_force_restore_error':float(np.max(np.abs(d.qfrc_actuator-original_actuator_force))),'recomputed':{}}
        np.savez_compressed(a.output/'balance_failure_state.npz',qpos=d.qpos.copy(),qvel=d.qvel.copy(),act=d.act.copy(),ctrl=d.ctrl.copy(),xfrc_applied=d.xfrc_applied.copy())
        for iterations in [4,20,100]:
            m.opt.iterations=iterations;mujoco.mj_forward(m,d);mujoco.mj_fullM(m,M,d.qM)
            check=T@(M@d.qacc+d.qfrc_bias-d.qfrc_passive-d.qfrc_constraint-d.qfrc_actuator)
            diagnostic['recomputed'][str(iterations)]=float(np.max(np.abs(check)))
        (a.output/'balance_failure.json').write_text(json.dumps(diagnostic,indent=2))
        raise RuntimeError(diagnostic)
    if a.acceleration_feedback_gain>0:
        desired=A@act+a.acceleration_feedback_gain*(T@M@T.T)@(qacc-d.qacc[vids])
    scale=np.maximum(np.linalg.norm(A,axis=1),1.);scale[:3]/=a.trunk_priority
    # Tiny previous-action regularizer selects among redundant allocations.
    weight=1e-3
    sol=lsq_linear(np.vstack([A/scale[:,None],weight*np.eye(m.nu)]),np.r_[desired/scale,weight*previous],bounds=(0,1),lsq_solver='lsmr',tol=1e-6,max_iter=40)
    allocation_trace.append(np.stack([desired,A@sol.x]));target_trace.append(target.copy());error_trace.append(target-d.qpos[qids]);excitation_trace.append(sol.x.copy())
    previous=sol.x;d.ctrl[:]=previous
    if step==0 and a.initialize_activation:d.act[:]=previous
    residuals.append(float(np.linalg.norm(A@sol.x-desired)/max(np.linalg.norm(desired),1e-8)));successes.append(bool(sol.success))
    q=d.qpos[3:7];w0=initial[3];xyz0=initial[4:7];bw=q[0];bv=-q[1:];ew=w0*bw-xyz0@bv;ev=w0*bv+bw*xyz0+np.cross(xyz0,bv)
    omega=d.xmat[pelvis].reshape(3,3)@d.qvel[3:6]
    horizontal=mass*2*(np.array([-command,0.])-d.qvel[:2]);horizontal*=min(1.,.3*mass*9.81/max(np.linalg.norm(horizontal),1e-8))
    vertical=np.clip(mass*(25*(initial[2]-d.qpos[2])-8*d.qvel[2]+.8*9.81),-1.2*mass*9.81,1.2*mass*9.81)
    torque=200*2*ev*(1 if ew>=0 else -1)-40*omega;torque*=min(1.,120/max(np.linalg.norm(torque),1e-8));wrench=np.r_[horizontal,vertical,torque]*a.support
    if a.support_ramp_seconds>0:wrench*=min(1.,step*.02/a.support_ramp_seconds)
    d.xfrc_applied[:]=0.;d.xfrc_applied[pelvis]=wrench
    for _ in range(10):mujoco.mj_step(m,d)
    mujoco.mj_forward(m,d)
    gap=float(d.xpos[head,2]-d.xpos[pelvis,2]);up=1-2*(d.qpos[4]**2+d.qpos[5]**2)
    assert np.isfinite(d.qpos).all() and np.all(d.ctrl>=0) and np.all(d.ctrl<=1)
    if step%2==0:
        poses.append(d.qpos.copy());acts.append(d.act.copy());wrenches.append(wrench);speeds.append(-d.qvel[0]);commands.append(command);heights.append(d.qpos[2]);gaps.append(gap)
    terminated=bool(d.qpos[2]<.6 or d.qpos[2]>1.35 or up<.45 or gap<.6*initial_gap)
    if terminated:break
np.savez_compressed(a.output/'allocation_trace.npz',torque=allocation_trace,target=target_trace,error=error_trace,excitation=excitation_trace,joint_names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j) for j in independent],dt=.02)
np.savez_compressed(a.output/'rollout.npz',qpos=poses,act=acts,external_wrench=wrenches,velocity=speeds,command=commands,reset=np.zeros(len(poses),bool),dt=.04)
(a.output/'model.xml').write_text((ROOT/'outputs/isaac_velocity/assets/model.xml').read_text())
(a.output/'assistance.json').write_text(json.dumps(dict(body='pelvis',controller='pelvis_pd',strength=a.support,weight_support=.8,vertical_limit=1.2,fraction_body_weight=0.)))
result=dict(mode=a.mode,kp=a.kp,solver_iterations=a.solver_iterations,walk_delay=a.walk_delay,motion_ramp_seconds=a.motion_ramp_seconds,acceleration_feedback_gain=a.acceleration_feedback_gain,max_projected_dynamics_balance_error=max(balance_errors),trunk_priority=a.trunk_priority,support=a.support,initialize_activation=a.initialize_activation,support_ramp_seconds=a.support_ramp_seconds,simulated_seconds=(step+1)*.02,terminated=terminated,minimum_root_height=float(min(heights)),minimum_head_gap=float(min(gaps)),mean_allocation_relative_residual=float(np.mean(residuals)),solver_success_fraction=float(np.mean(successes)),wall_seconds=time.monotonic()-start,learned_policy=False,engine='CPU MuJoCo 3.4',caveat='Exploratory bounded excitation control. Not Isaac Lab validation or independent walking; inverse model neglects changing tangent acceleration and contact force prediction.')
(a.output/'metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
