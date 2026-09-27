"""8192-world full-body mjlab GPU simulation benchmark (no policy/training/render)."""
import json,time,subprocess
from pathlib import Path
from importlib.metadata import version
import mujoco,numpy as np,torch,warp as wp
from mjlab.actuator import XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import Entity,EntityCfg,EntityArticulationInfoCfg
from mjlab.sim import Simulation,SimulationCfg,MujocoCfg
wp.config.kernel_cache_dir='/tmp/muscle-feasibility/warp-cache'
wp.init()
assert torch.cuda.is_available()
N,DT,DECIMATION,CONTROLS=8192,0.002,5,100
path=Path('/home/k_miyazawa/musclemimic/.venv/lib/python3.11/site-packages/musclemimic_models/model/body/myofullbody.xml')
source=mujoco.MjSpec.from_file(str(path))
entity=Entity(EntityCfg(spec_fn=lambda:mujoco.MjSpec.from_file(str(path)),articulation=EntityArticulationInfoCfg(actuators=(XmlActuatorCfg(target_names_expr=tuple(a.target for a in source.actuators),transmission_type=TransmissionType.TENDON),))))
model=entity.compile()
print('BUILD',N,'worlds',model.nu,'muscles',flush=True)
t0=time.perf_counter()
sim=Simulation(num_envs=N,model=model,device='cuda:0',cfg=SimulationCfg(nconmax=96,njmax=768,mujoco=MujocoCfg(timestep=DT,integrator='implicitfast',iterations=50,ls_iterations=20)))
entity.initialize(model,sim.model,sim.data,'cuda:0')
sim.reset();entity.reset();sim.forward();wp.synchronize()
build=time.perf_counter()-t0
print('READY build_seconds',build,flush=True)
ids=torch.tensor(model.actuator_trnid[:,0].copy(),device='cuda:0',dtype=torch.long)
rng=np.random.default_rng(42)
targets=torch.tensor(rng.beta(.8,2.,size=(7,N,model.nu)).astype(np.float32)*.55,device='cuda:0');targets[0]=0
seen_overflow=torch.zeros(N,device='cuda:0',dtype=torch.int32)
max_nefc=torch.zeros((),device='cuda:0',dtype=torch.int32)
max_contacts=torch.zeros((),device='cuda:0',dtype=torch.int32)
overflow=wp.to_torch(sim.wp_data.overflow);nefc=wp.to_torch(sim.wp_data.nefc);nacon=wp.to_torch(sim.wp_data.nacon)
def block(first,count):
    for control in range(first,first+count):
        i,t=divmod(control,20);f=t/20.;blend=f*f*(3-2*f)
        excitation=(1-blend)*targets[i]+blend*targets[i+1]
        entity.set_tendon_effort_target(excitation,tendon_ids=ids)
        entity.write_data_to_sim()
        for _ in range(DECIMATION):
            sim.step()
            seen_overflow.bitwise_or_(overflow)
            torch.maximum(max_nefc,nefc.max(),out=max_nefc)
            torch.maximum(max_contacts,nacon.max(),out=max_contacts)
print('WARMUP',flush=True)
block(0,10);wp.synchronize()
results={'gpu':torch.cuda.get_device_name(0),'versions':{p:version(p) for p in ['mjlab','mujoco','mujoco-warp','torch','warp-lang']},'num_envs':N,'nu':model.nu,'nv':model.nv,'ntendon':model.ntendon,'dt':DT,'decimation':DECIMATION,'physics_steps_per_trial':CONTROLS*DECIMATION,'sim_seconds_per_env_per_trial':DT*CONTROLS*DECIMATION,'build_seconds':build,'solver':'Newton','iterations':50,'ls_iterations':20,'nconmax_per_world':96,'njmax':768,'trials':[],'timing_scope':'simulation, random excitation interpolation and Entity control write, per-step GPU overflow/contact/constraint monitoring; excludes initialization, reset, validation, policy, rewards, rendering, learning'}
for trial in range(3):
    sim.reset();entity.reset();sim.forward();wp.synchronize()
    seen_overflow.zero_();max_nefc.zero_();max_contacts.zero_();wp.synchronize()
    elapsed=0.
    for start in range(0,CONTROLS,20):
        wp.synchronize();t0=time.perf_counter();block(start,20);wp.synchronize();elapsed+=time.perf_counter()-t0
        print('PROGRESS trial',trial+1,'control_steps',start+20,'timed_seconds',round(elapsed,3),flush=True)
    finite={k:bool(torch.isfinite(wp.to_torch(getattr(sim.wp_data,k))).all().item()) for k in ['qpos','qvel','act','actuator_force']}
    times=sim.wp_data.time.numpy()
    item={'elapsed_seconds':elapsed,'physics_env_steps_per_second':N*CONTROLS*DECIMATION/elapsed,'control_env_steps_per_second':N*CONTROLS/elapsed,'per_env_realtime_factor':DT*CONTROLS*DECIMATION/elapsed,'all_finite':finite,'overflow_bitmask':int(seen_overflow.max().item()),'peak_constraints_per_world':int(max_nefc.item()),'peak_contacts_all_worlds':int(max_contacts.item()),'time_min':float(times.min()),'time_max':float(times.max()),'max_abs_muscle_force':float(wp.to_torch(sim.wp_data.actuator_force).abs().max().item())}
    results['trials'].append(item)
    results['gpu_memory']=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
    Path('/tmp/muscle-feasibility/fullbody_8192_benchmark.json').write_text(json.dumps(results,indent=2))
    print('TRIAL',json.dumps(item),flush=True)
    assert all(finite.values()) and item['overflow_bitmask']==0,item
    np.testing.assert_allclose(times,CONTROLS*DECIMATION*DT,atol=1e-4)
print('RESULT',json.dumps(results),flush=True)
