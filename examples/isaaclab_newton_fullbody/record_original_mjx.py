"""Record original MJX policy rollout with the same seeded reset-frame sampler."""
import os,sys,json
from pathlib import Path
import numpy as np
import jax
import mujoco
from omegaconf import OmegaConf
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from fullbody import eval as evaluation
OUT=Path(os.environ.get('MM_OUTPUT_DIR','/tmp/muscle-feasibility/original_mjx_30s'));OUT.mkdir(parents=True,exist_ok=True)
load=evaluation.load_checkpoint
def load_checkpoint(*args,**kwargs):
    c,s,m=load(*args,**kwargs);OmegaConf.set_struct(c,False);c.experiment.env_params.nconmax=1024
    return c,s,m
evaluation.load_checkpoint=load_checkpoint
original=evaluation.PPOJax.play_policy
FIELDS=['qpos','qvel','act','ctrl','actuator_force','actuator_length','actuator_velocity','sensordata','xpos','xquat','xmat','site_xpos','site_xmat','cvel','subtree_com','qacc','qacc_warmstart','xfrc_applied']
def arrays(data):
    return {k:np.asarray(getattr(data._impl,k) if hasattr(data._impl,k) else getattr(data,k)) for k in FIELDS}
def play(env,*args,**kwargs):
    rng=np.random.RandomState(42);resets=[];records=[];poses=[];acts=[];observations=[];actions=[]
    spec=env._mjspec.copy()
    import musclemimic_models
    assets=Path(musclemimic_models.__file__).parent/'model/body'
    spec.meshdir=str((assets/spec.meshdir).resolve());spec.texturedir=str((assets/spec.texturedir).resolve())
    (OUT/'model.xml').write_text(spec.to_xml())
    support_fraction=float(os.environ.get('MM_SUPPORT_FRACTION','0'))
    assert 0 <= support_fraction < 1
    support_force=float(np.sum(env._model.body_mass)*abs(env._model.opt.gravity[2])*support_fraction)
    support_body=os.environ.get('MM_SUPPORT_BODY','pelvis')
    support_body_id=mujoco.mj_name2id(env._model,mujoco.mjtObj.mjOBJ_BODY,support_body)
    assert support_body_id>0,support_body
    if support_force:
        original_pre_step=env._mjx_simulation_pre_step
        def assisted_pre_step(model,data,carry):
            model,data,carry=original_pre_step(model,data,carry)
            return model,data.replace(xfrc_applied=data.xfrc_applied.at[support_body_id,2].set(support_force)),carry
        env._mjx_simulation_pre_step=assisted_pre_step
    (OUT/'assistance.json').write_text(json.dumps(dict(body=support_body,fraction_body_weight=support_fraction,force_world_N=[0.,0.,support_force],torque_Nm=[0.,0.,0.],point='body center of mass'),indent=2))
    rawreset=env.mjx_reset;rawstep=jax.jit(env.mjx_step)
    def reset(key):
        frame=int(rng.randint(0,env.th.len_trajectory(0)))
        env.th.random_start=False;env.th.use_fixed_start=True;env.th.fixed_start_conf=(0,frame)
        state=rawreset(key)
        resets.append(dict(control_step=len(records),start_frame=frame));print('RESET',resets[-1],flush=True)
        if not records:np.savez(OUT/'initial.npz',**arrays(state.data),obs=np.asarray(state.observation))
        return state
    def step(state,action):
        result=rawstep(state,action)
        d=arrays(result.data)
        if support_force:assert np.isclose(d['xfrc_applied'][support_body_id,2],support_force,rtol=1e-6)
        if not records:np.savez(OUT/'first_step.npz',**d,obs=np.asarray(result.observation),action=np.asarray(action))
        poses.append(d['qpos']);acts.append(d['act']);observations.append(np.asarray(result.observation));actions.append(np.asarray(action))
        records.append(dict(time=float(result.data.time),root=d['qpos'][:3].tolist(),done=bool(result.done),absorbing=bool(result.absorbing),reward=float(result.reward)))
        if len(records)%100==0: print('MJX',len(records),records[-1],flush=True)
        return result
    env.mjx_reset=reset;env.mjx_step=step;kwargs['sequential_mjx']=True;kwargs['n_envs']=1
    try:return original(env,*args,**kwargs)
    finally:
        np.savez_compressed(OUT/'rollout.npz',qpos=poses,act=acts,obs=observations,action=actions,dt=env.dt)
        (OUT/'records.json').write_text(json.dumps(records,indent=2));(OUT/'resets.json').write_text(json.dumps(resets,indent=2))
evaluation.PPOJax.play_policy=staticmethod(play)
raise SystemExit(evaluation.main())
