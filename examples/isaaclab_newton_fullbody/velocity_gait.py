"""GPU foot clearance from the authored capsule and ellipsoid collision shapes."""
import re
import mujoco
import torch
import warp as wp

class FootClearance:
    def __init__(self,model,data,device):
        self.positions=wp.to_torch(data.geom_xpos)
        self.rotations=wp.to_torch(data.geom_xmat)
        names=[re.sub(r'_\d+$','',mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i).rsplit('/',1)[-1]) for i in range(model.ngeom)]
        self.feet=[]
        for side in ['r','l']:
            expected=[side+'_foot_col1',side+'_foot_col3',side+'_foot_col4',side+'_bofoot_col1',side+'_bofoot_col2']
            ids=[names.index(name) for name in expected]
            types=model.geom_type[ids]
            assert all(t in [int(mujoco.mjtGeom.mjGEOM_CAPSULE),int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)] for t in types), (expected, types)
            self.feet.append((ids,torch.tensor(model.geom_size[ids],dtype=torch.float32,device=device),torch.tensor(types==int(mujoco.mjtGeom.mjGEOM_CAPSULE),device=device)))

    def __call__(self):
        clearances=[]
        for ids,size,capsule in self.feet:
            row_z=self.rotations[:,ids,2,:]
            capsule_extent=size[:,0]+row_z[:,:,2].abs()*size[:,1]
            ellipsoid_extent=((row_z*size).square().sum(-1)).sqrt()
            extent=torch.where(capsule,capsule_extent,ellipsoid_extent)
            clearances.append((self.positions[:,ids,2]-extent).min(-1).values)
        return torch.stack(clearances,dim=-1)
