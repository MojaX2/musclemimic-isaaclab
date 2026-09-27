"""Project reference dependent scalar joints onto active polynomial equalities.

Only reference/reset data is changed. Simulator constraints and independent joint
coordinates are untouched. Velocities use the derivative of the same polynomial.
"""
import mujoco
import numpy as np


def project_joint_equalities(model, qpos, qvel):
    q=np.array(qpos,dtype=np.float64,copy=True);v=np.array(qvel,dtype=np.float64,copy=True)
    if q.shape[-1]!=model.nq or v.shape[-1]!=model.nv or q.shape[:-1]!=v.shape[:-1]:
        raise ValueError('Reference shape mismatch')
    pending={}
    for e in range(model.neq):
        if not model.eq_active0[e]:continue
        if int(model.eq_type[e])!=int(mujoco.mjtEq.mjEQ_JOINT):
            raise ValueError('Only scalar joint equalities supported')
        j1,j2=int(model.eq_obj1id[e]),int(model.eq_obj2id[e])
        if j1 in pending:raise ValueError('Multiple constraints on dependent joint')
        if any(int(model.jnt_type[j]) not in (int(mujoco.mjtJoint.mjJNT_HINGE),int(mujoco.mjtJoint.mjJNT_SLIDE)) for j in [j1,j2] if j>=0):
            raise ValueError('Non-scalar equality')
        pending[j1]=(j2,model.eq_data[e,:5].copy())
    while pending:
        ready=[j for j,(parent,_) in pending.items() if parent not in pending]
        if not ready:raise ValueError('Cyclic joint equalities')
        for j1 in ready:
            j2,c=pending.pop(j1);a,b=int(model.jnt_qposadr[j1]),int(model.jnt_dofadr[j1])
            if j2<0:q[...,a]=model.qpos0[a]+c[0];v[...,b]=0.;continue
            a2,b2=int(model.jnt_qposadr[j2]),int(model.jnt_dofadr[j2]);x=q[...,a2]-model.qpos0[a2]
            q[...,a]=model.qpos0[a]+np.polynomial.polynomial.polyval(x,c)
            v[...,b]=np.polynomial.polynomial.polyval(x,np.arange(1,5)*c[1:])*v[...,b2]
    return q,v
