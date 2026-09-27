"""Use unchanged MuscleMimic inference with optional Isaac Lab physics IPC."""
import os, sys, json, subprocess
from pathlib import Path
from multiprocessing.connection import Listener
import numpy as np
import mujoco
ROOT=Path(os.environ.get('MM_OUTPUT_DIR','/tmp/muscle-feasibility/pretrained')); ROOT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from fullbody import eval as evaluation
from musclemimic.algorithms import PPOJax
os.environ.setdefault('MM_MATCH_MJX','1')
os.environ.setdefault('MM_MATCH_PHYSICS','1')
os.environ.setdefault('MM_GPU_OBSERVATIONS','1')
os.environ.setdefault('MM_SYNC_INTERVAL','0')
original_play=PPOJax.play_policy_mujoco
if os.environ.get('MM_MATCH_MJX')=='1':
    original_load=evaluation.load_checkpoint
    def load_checkpoint(*args,**kwargs):
        from omegaconf import OmegaConf
        config,state,metadata=original_load(*args,**kwargs)
        OmegaConf.set_struct(config,False)
        # MjxMyoFullBody uses these defaults; MyoFullBody otherwise uses MuJoCo defaults.
        if 'model_option_conf' not in config.experiment.env_params:
            config.experiment.env_params.model_option_conf=dict(iterations=4,ls_iterations=8,disableflags=int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP))
        return config,state,metadata
    evaluation.load_checkpoint=load_checkpoint
backend=os.environ.get('MM_BACKEND','baseline')
def play(env, *args, **kwargs):
    records=[]; resets=[]; poses=[]; acts=[]; observations=[]; actions=[]; conn=None; proc=None
    if "MM_RESET_SEED" in os.environ:
        np.random.seed(int(os.environ["MM_RESET_SEED"]))
    original_reset=env.reset
    def reset(*reset_args, **reset_kwargs):
        obs=original_reset(*reset_args, **reset_kwargs)
        if conn and os.environ.get('MM_GPU_OBSERVATIONS')=='1':
            gpu_step(env._model,env._data,0)
            obs,env._additional_carry=env._create_observation(env._model,env._data,env._additional_carry)
            env._obs=obs
        state=env._additional_carry.traj_state
        resets.append(dict(control_step=len(records),traj_no=int(state.traj_no),start_frame=int(state.subtraj_step_no),root=env._data.qpos[:3].tolist()))
        if not records:
            np.savez(ROOT/"initial.npz",qpos=env._data.qpos,qvel=env._data.qvel,act=env._data.act,obs=obs)
        print("RESET",resets[-1],flush=True)
        return obs
    env.reset=reset
    print("TRAJECTORY SAMPLING",env.th.random_start,env.th.start_from_random_step,env.th.fixed_start_conf,flush=True)
    original_mjstep=mujoco.mj_step
    (ROOT/"model_info.json").write_text(json.dumps(dict(nq=env._model.nq,nv=env._model.nv,nu=env._model.nu,integrator=int(env._model.opt.integrator),timestep=env._model.opt.timestep,observation_dim=env.info.observation_space.shape[0])))
    if backend=='isaaclab':
        path=ROOT/'trained_model.xml'
        path.write_text(env._mjspec.to_xml())
        # Resolve asset directories of the installed source model.
        import xml.etree.ElementTree as ET
        tree=ET.parse(path); compiler=tree.getroot().find('compiler')
        import musclemimic_models
        assets=Path(musclemimic_models.__file__).parent/'model/body'
        for key in ('meshdir','texturedir'):
            value=compiler.get(key)
            if value and not Path(value).is_absolute(): compiler.set(key,str((assets/value).resolve()))
        tree.write(path)
        listener=Listener(('127.0.0.1',0),authkey=b'muscle-local-test')
        worker=Path(__file__).with_name('pretrained_physics_worker.py')
        log=open(ROOT/'worker.log','w')
        childenv=os.environ.copy(); childenv['MM_MODEL_XML']=str(path)
        proc=subprocess.Popen(['uv','run','--no-sync','--project','/home/k_miyazawa/IsaacLab','python',str(worker),str(listener.address[1])],env=childenv,stdout=log,stderr=subprocess.STDOUT)
        listener._listener._socket.settimeout(180)
        conn=listener.accept(); print('WORKER',conn.recv(),flush=True)
        original_mjstep=mujoco.mj_step
        def gpu_step(model,data,nstep=1):
            if model is not env._model: return original_mjstep(model,data,nstep)
            conn.send(dict(qpos=data.qpos.copy(),qvel=data.qvel.copy(),act=data.act.copy(),ctrl=data.ctrl.copy(),time=data.time,nstep=nstep))
            result=conn.recv()
            for name in ('qpos','qvel','act'): getattr(data,name)[:]=result[name]
            data.time=result['time']
            if os.environ.get('MM_GPU_OBSERVATIONS')=='1':
                for name in ['xpos','xquat','xmat','xipos','ximat','cvel','subtree_com','site_xpos','site_xmat','sensordata']:
                    target=getattr(data,name);target[:]=result[name].reshape(target.shape)
            else:
                mujoco.mj_forward(model,data)
            for name in ('actuator_force','actuator_length','actuator_velocity'):
                getattr(data,name)[:]=result[name]
        mujoco.mj_step=gpu_step
    original_step=env.step
    def step(action):
        out=original_step(action)
        poses.append(env._data.qpos.copy());acts.append(env._data.act.copy());observations.append(np.asarray(out[0]));actions.append(np.asarray(action).reshape(-1))
        if not records:
            np.savez(ROOT/"first_step.npz",qpos=env._data.qpos,qvel=env._data.qvel,act=env._data.act,obs=out[0],action=actions[-1],actuator_force=env._data.actuator_force,actuator_length=env._data.actuator_length,actuator_velocity=env._data.actuator_velocity,sensordata=env._data.sensordata)
        records.append(dict(time=float(env._data.time),root=env._data.qpos[:3].tolist(),reward=float(out[1]),absorbing=bool(out[2]),done=bool(out[3])))
        if len(records)%100==0: print(backend,len(records),records[-1],flush=True)
        return out
    env.step=step
    print('MODEL',env._model.nq,env._model.nv,env._model.nu,'OBS',env.info.observation_space.shape,flush=True)
    try: return original_play(env,*args,**kwargs)
    finally:
        mujoco.mj_step=original_mjstep
        (ROOT/f'{backend}.json').write_text(json.dumps(records,indent=2))
        (ROOT/f'{backend}_resets.json').write_text(json.dumps(resets,indent=2))
        np.savez_compressed(ROOT/"rollout.npz",qpos=poses,act=acts,obs=observations,action=actions,dt=env.dt)
        env.reset=original_reset; env.step=original_step
        if conn: conn.send(None); conn.close(); proc.wait(timeout=60)
PPOJax.play_policy_mujoco=staticmethod(play)
raise SystemExit(evaluation.main())
