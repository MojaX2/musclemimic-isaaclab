"""Bounded world-frame pelvis stabilization for explicitly assisted training.

Returns [Fx,Fy,Fz,Tx,Ty,Tz] at the pelvis center of mass. Velocity support can
transport the body: assisted speed tracking is never evidence of independent
walking. A zero strength returns exactly zero wrench.
"""
import torch


def pelvis_wrench(qpos, world_velocity, target_quaternion, target_height,
                  target_velocity_xy, mass, strength, weight_support=.3, vertical_limit=.6):
    quat=qpos[:,3:7]
    # target * conjugate(current): orientation correction in world coordinates.
    aw=target_quaternion[0];av=target_quaternion[1:]
    bw=quat[:,:1];bv=-quat[:,1:]
    ew=aw*bw-(av*bv).sum(-1,keepdim=True)
    ev=aw*bv+bw*av+torch.linalg.cross(av.expand_as(bv),bv)
    rotation_error=2*ev*torch.where(ew>=0,1.,-1.)
    horizontal=mass*2*(target_velocity_xy-world_velocity[:,:2])
    limit=.3*mass*9.81
    horizontal*= (limit/horizontal.norm(dim=-1,keepdim=True).clamp(min=limit)).clamp(max=1)
    vertical=mass*(25*(target_height-qpos[:,2])-8*world_velocity[:,2]+weight_support*9.81)
    vertical=vertical.clamp(-vertical_limit*mass*9.81,vertical_limit*mass*9.81)
    torque=200*rotation_error-40*world_velocity[:,3:6]
    torque*=120/torque.norm(dim=-1,keepdim=True).clamp(min=120)
    return torch.cat([horizontal,vertical[:,None],torque],dim=-1)*strength[:,None]
