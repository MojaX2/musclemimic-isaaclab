"""Contact-force sensing and anatomical joint mapping for the G1 reward variant."""
import re
import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp


@wp.kernel
def sum_foot_normal(nacon: wp.array(dtype=int), world: wp.array(dtype=int),
                    geom: wp.array(dtype=wp.vec2i), foot: wp.array(dtype=int),
                    floor: wp.array(dtype=int), force: wp.array(dtype=wp.spatial_vector),
                    out: wp.array2d(dtype=float)):
    i=wp.tid()
    if i>=nacon[0]:
        return
    g0=geom[i][0];g1=geom[i][1]
    if g0<0 or g1<0:
        return
    side=int(-1)
    if floor[g0]!=0:
        side=foot[g1]
    elif floor[g1]!=0:
        side=foot[g0]
    if side>=0:
        wp.atomic_add(out, world[i], side, wp.max(force[i][0], 0.0))


class FootContactForces:
    def __init__(self, model, warp_model, data, feet, device):
        self.model=warp_model;self.data=data;self.device=device
        foot=np.full(model.ngeom,-1,dtype=np.int32)
        for side,(ids,_,_) in enumerate(feet.feet):
            foot[ids]=side
        self.foot=wp.array(foot,device=device)
        self.floor=wp.array((model.geom_type==int(mujoco.mjtGeom.mjGEOM_PLANE)).astype(np.int32),device=device)
        self.ids=wp.array(np.arange(data.naconmax,dtype=np.int32),device=device)
        self.force=wp.zeros(data.naconmax,dtype=wp.spatial_vector,device=device)
        self.normal=wp.zeros((data.qpos.shape[0],2),dtype=float,device=device)
        self.normal_t=wp.to_torch(self.normal)

    def __call__(self):
        self.normal.zero_()
        mjw.contact_force(self.model,self.data,self.ids,False,self.force)
        c=self.data.contact
        wp.launch(sum_foot_normal,dim=self.data.naconmax,
                  inputs=[self.data.nacon,c.worldid,c.geom,self.foot,self.floor,self.force,self.normal],device=self.device)
        return self.normal_t > 1.0


def joint_groups(model, device, joint_ids=None):
    names=[re.sub(r'_\d+$','',mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i).rsplit('/',1)[-1])
           for i in range(model.njnt)]
    groups={
        'hip':[f'hip_{kind}_{s}' for s in ['r','l'] for kind in ['adduction','rotation']],
        'arms':[f'{kind}_{s}' for s in ['r','l'] for kind in ['elv_angle','shoulder_elv','shoulder_rot','elbow_flex','pro_sup']],
        'torso':['flex_extension','lat_bending','axial_rotation'],
        'ankle':[f'{kind}_{s}' for s in ['r','l'] for kind in ['ankle_angle','subtalar_angle']],
        'leg':[f'{kind}_{s}' for s in ['r','l'] for kind in ['hip_flexion','hip_adduction','hip_rotation','knee_angle']],
    }
    result={}
    for key, wanted in groups.items():
        ids=[joint_ids[name] if joint_ids is not None else names.index(name) for name in wanted]
        result[key]=torch.tensor(model.jnt_qposadr[ids],device=device,dtype=torch.long)
        result[key+'_dof']=torch.tensor(model.jnt_dofadr[ids],device=device,dtype=torch.long)
        if key=='ankle':
            limits=torch.tensor(model.jnt_range[ids],device=device,dtype=torch.float32)
            mid=limits.mean(1);half=.9*(limits[:,1]-limits[:,0])/2
            result['ankle_low']=mid-half;result['ankle_high']=mid+half
    return result


def root_world_velocity(q, v):
    """MuJoCo free-joint linear velocity is world; angular velocity is body local."""
    quat=q[:,3:7];xyz=quat[:,1:];angular=v[:,3:6]
    t=2*torch.linalg.cross(xyz,angular)
    rotated=angular+quat[:,:1]*t+torch.linalg.cross(xyz,t)
    return torch.cat((v[:,:3],rotated),dim=1)
