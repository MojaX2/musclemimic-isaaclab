"""Write a new constraint-consistent reset bank and physical residual diagnostics."""
import argparse,json
from pathlib import Path
import mujoco,numpy as np
from project_joint_equalities import project_joint_equalities
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
root=Path(__file__).resolve().parents[2];m=mujoco.MjModel.from_xml_path(str(root/'outputs/isaac_velocity/assets/model.xml'));d=mujoco.MjData(m)
r=dict(np.load(a.source,allow_pickle=False));q,v=project_joint_equalities(m,r['qpos'],r['qvel']);results={}
for label,qs,vs in [('before',r['qpos'],r['qvel']),('after',q,v)]:
 vals=[]
 for qi,vi in zip(qs,vs):
  mujoco.mj_resetData(m,d);d.qpos[:]=qi;d.qvel[:]=vi;mujoco.mj_forward(m,d)
  mask=d.efc_type==int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
  vals.append([np.max(np.abs(d.efc_pos[mask])),np.max(np.abs(d.efc_vel[mask])),min([c.dist for c in d.contact],default=0.)])
 vals=np.array(vals);results[label]=dict(position_residual_max=float(vals[:,0].max()),velocity_residual_max=float(vals[:,1].max()),penetration_max_m=float(-vals[:,2].min()))
assert results['after']['position_residual_max']<1e-9 and results['after']['velocity_residual_max']<1e-9
r.update(qpos=q,qvel=v);np.savez_compressed(a.output,**r)
results.update(source=str(a.source.resolve()),output=str(a.output.resolve()),count=len(q),note='Dependent scalar qpos/qvel corrected only; independent joints, root, phase and commands preserved. This is not evidence of gait learning.')
a.output.with_suffix('.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
