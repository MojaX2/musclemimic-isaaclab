"""External Isaac Lab/Newton adapter smoke test; no Isaac Lab source patches."""
import json,time
from pathlib import Path
import numpy as np,warp as wp,newton,mujoco
from importlib.metadata import version
wp.config.kernel_cache_dir='/tmp/muscle-feasibility/warp-cache'
from isaaclab.sim import SimulationCfg,build_simulation_context
from isaaclab_newton.physics import NewtonCfg,MJWarpSolverCfg,NewtonManager,NewtonMJWarpManager
from newton_fullbody_adapter import FullBodySolver,make_builder,REFERENCE
from newton.solvers import SolverMuJoCo
class MuscleManager(NewtonMJWarpManager):
    @classmethod
    def _create_solver(cls,model,solver_cfg):
        return FullBodySolver(model,**cls._filter_solver_kwargs(SolverMuJoCo,solver_cfg))
cfg=NewtonCfg(solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=True,integrator='implicitfast',iterations=100,ls_iterations=50,nconmax=512,njmax=2048),use_cuda_graph=True,num_substeps=1)
cfg.class_type=MuscleManager
sim_cfg=SimulationCfg(dt=.002,device='cuda:0',physics=cfg)
with build_simulation_context(sim_cfg=sim_cfg) as sim:
    template=make_builder()
    b=newton.ModelBuilder();SolverMuJoCo.register_custom_attributes(b)
    for _ in range(4): b.add_world(template)
    NewtonManager.set_builder(b)
    print('ISAAC_RESET',flush=True)
    sim.reset()
    s=NewtonManager._solver;m=s.mj_model
    print('ISAAC_READY',sim.physics_manager.__name__,m.nq,m.nv,m.nu,m.na,m.ntendon,m.neq,flush=True)
    ctrl=NewtonManager.get_control().mujoco.ctrl
    print('CTRL',ctrl.shape,'STATE',s.mjw_data.qpos.shape,flush=True)
    assert (m.nu,m.na,m.ntendon,m.neq)==(416,416,424,51)
    initial=s.mjw_data.qpos.numpy().copy()
    # Check that native muscle parameters and tendon targets survived conversion.
    for field in ['gainprm','biasprm','dynprm','lengthrange','trnid']:
        np.testing.assert_allclose(getattr(m,'actuator_'+field),getattr(REFERENCE,'actuator_'+field))
    native_data=mujoco.MjData(REFERENCE)
    mujoco.mj_forward(REFERENCE,native_data)
    length_error=float(np.max(np.abs(s.mjw_data.actuator_length.numpy()[0]-native_data.actuator_length)))
    # Report geometry conversion error; this smoke test does not certify equivalence.
    assert np.isfinite(length_error)
    rng=np.random.default_rng(7)
    start=time.perf_counter()
    for step in range(500):
        if step%20==0:
            values=rng.uniform(.02,.25,size=ctrl.shape).astype(np.float32)
            ctrl.assign(values)
        sim.step(render=False)
        assert not s.mjw_data.overflow.numpy().any(),('overflow',step,s.mjw_data.overflow.numpy().tolist())
        if (step+1)%100==0:
            wp.synchronize()
            for field in ['qpos','qvel','act','actuator_force']:
                assert np.isfinite(getattr(s.mjw_data,field).numpy()).all(),(step,field)
            print('STEP',step+1,'force',float(np.abs(s.mjw_data.actuator_force.numpy()).max()),'qvel',float(np.abs(s.mjw_data.qvel.numpy()).max()),flush=True)
    wp.synchronize()
    result={'backend':sim.physics_manager.__name__,'solver':type(s).__name__,'device':str(s.mjw_data.qpos.device),'nu':m.nu,'na':m.na,'ntendon':m.ntendon,'neq':m.neq,'num_envs':4,'cuda_graph':True,'steps':500,'dt':.002,'wall_seconds':time.perf_counter()-start,'max_qpos_change':float(np.abs(s.mjw_data.qpos.numpy()-initial).max()),'activation_min':float(s.mjw_data.act.numpy().min()),'activation_max':float(s.mjw_data.act.numpy().max()),'max_force':float(np.abs(s.mjw_data.actuator_force.numpy()).max()),'all_finite':True}
    assert result['max_qpos_change']>1e-5 and result['max_force']>0 and result['activation_max']>0
    result['versions']={p:version(p) for p in ['newton','mujoco','mujoco-warp','warp-lang']}
    result['initial_actuator_length_max_error_m']=length_error
    result['scope']='Isaac Lab SimulationContext + custom Newton manager/solver; no Articulation/ManagerBasedRLEnv, rendering, rewards, or learning'
    result['solver_options']={'iterations':100,'ls_iterations':50,'nconmax':512,'njmax':2048}
    final_act=s.mjw_data.act.numpy().copy()
    assert not np.allclose(final_act[0],final_act[1])
    mask=wp.array([False,True,False,False,False],dtype=wp.bool,device='cuda:0')
    sim.physics_manager._reset_solver_internals(mask)
    wp.synchronize()
    reset_act=s.mjw_data.act.numpy()
    np.testing.assert_allclose(reset_act[1],0,atol=1e-7)
    np.testing.assert_array_equal(reset_act[[0,2,3]],final_act[[0,2,3]])
    result['selective_activation_reset']=True
    result['overflow_detected']=False
    Path('/tmp/muscle-feasibility/isaaclab_fullbody_results.json').write_text(json.dumps(result,indent=2))
    print('RESULT',json.dumps(result),flush=True)
