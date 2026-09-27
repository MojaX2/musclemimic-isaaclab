"""Unassisted G1-inspired muscle velocity-tracking PPO (separate experiment).

Checkpoints, hourly evaluations and videos are local. No reference motion enters
observations or rewards; a saved upright pose is used only for episode reset.
"""
import argparse, json, math, os, time, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser()
p.add_argument('--num-envs',type=int,default=1024)
p.add_argument('--iterations',type=int,default=100000)
p.add_argument('--run-dir',type=Path,default=ROOT/'outputs/isaac_velocity_g1/run_001')
p.add_argument('--eval-seconds',type=float,default=20.)
p.add_argument('--eval-interval',type=float,default=3600.)
p.add_argument('--video-interval',type=float,default=3600.)
p.add_argument('--first-eval-seconds',type=float,default=300.)
p.add_argument('--support',type=float,choices=[0.],default=0.)
p.add_argument('--entropy-coef',type=float,default=0.)
p.add_argument('--exploration-std-final',type=float,help='Anneal the latent Gaussian std cap; no added physical assistance')
p.add_argument('--exploration-anneal-updates',type=int,default=600)
p.add_argument('--resume',type=Path)
p.add_argument('--smoke',action='store_true')
p.add_argument('--eval-only',action='store_true')
p.add_argument('--stochastic-eval',action='store_true')
p.add_argument('--mean-excitation-eval',action='store_true')
p.add_argument('--eval-phase',choices=['zero','random'],default='zero')
p.add_argument('--eval-support',type=float,choices=[0.])
p.add_argument('--eval-velocity-perturbation',type=float,default=0.)
p.add_argument('--eval-command-sequence',help='Comma-separated targets, equal-duration segments; eval-only')
p.add_argument('--legacy-support-normalization',action='store_true')
a=p.parse_args()
if a.exploration_std_final is not None and not .05<=a.exploration_std_final<=1.:p.error('--exploration-std-final must be in [.05,1]')
if a.exploration_anneal_updates<=0:p.error('--exploration-anneal-updates must be positive')
if a.eval_command_sequence and not a.eval_only:p.error('--eval-command-sequence requires --eval-only')
a.run_dir.mkdir(parents=True,exist_ok=True)
os.environ['MM_MODEL_XML']=str(ROOT/'outputs/isaac_velocity/assets/model.xml')
os.environ['MM_MATCH_PHYSICS']='1'
os.environ['MM_GPU_OBSERVATIONS']='0'
os.environ.pop('DISPLAY',None)
import mujoco, newton, mujoco_warp as mjw, warp as wp
import numpy as np
import torch
from tensordict import TensorDict
from velocity_normalization import VelocityNormalization
from velocity_exploration import cap_exploration
from velocity_gait import FootClearance
from g1_muscle_rewards import G1MuscleRewards, WEIGHTS, TOUCHDOWN_WEIGHT, forward_velocity
from g1_muscle_sensors import FootContactForces, joint_groups, root_world_velocity
from g1_muscle_posture import torso_collapsed
from torch.utils.tensorboard import SummaryWriter
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from isaaclab.sim import SimulationCfg,build_simulation_context
from isaaclab_newton.physics import NewtonCfg,MJWarpSolverCfg,NewtonManager,NewtonMJWarpManager
from newton.solvers import SolverMuJoCo
from newton_fullbody_adapter import FullBodySolver,make_builder,REFERENCE,JOINT_DOF
import newton_fullbody_adapter as adapter
# Keep generated importer files and compiled kernels separate from other runs.
adapter.ROOT=a.run_dir.resolve()/'adapter'
adapter.ROOT.mkdir(parents=True,exist_ok=True)
wp.config.kernel_cache_dir=str(ROOT/'outputs/isaac_velocity_g1/warp_cache')
torch.set_num_threads(8);torch.manual_seed(42);np.random.seed(42)

class NewtonVelocityManager(NewtonMJWarpManager):
    @classmethod
    def _create_solver(cls,model,solver_cfg):
        solver=FullBodySolver(model,**cls._filter_solver_kwargs(SolverMuJoCo,solver_cfg))
        # Inspect capacity flags explicitly instead of printing per-world iteration warnings.
        solver.mjw_model.opt.warn_overflow=False
        return solver

class VelocityEnv:
    def __init__(self,sim,n):
        self.sim=sim;self.n=n;self.device='cuda:0';self.dt=.02;self.decimation=10
        self.support=wp.full(n,a.support,dtype=float,device=self.device)
        self.support_t=wp.to_torch(self.support);self.support_level=a.support
        template=make_builder();b=newton.ModelBuilder();SolverMuJoCo.register_custom_attributes(b)
        for _ in range(n):b.add_world(template)
        NewtonManager.set_builder(b)
        # Deliberately register NO assistance force callback.
        sim.reset();self.s=NewtonManager._solver;self.d=self.s.mjw_data;self.m=self.s.mj_model
        self.q=wp.to_torch(self.d.qpos);self.v=wp.to_torch(self.d.qvel);self.act=wp.to_torch(self.d.act)
        self.ctrl=wp.to_torch(NewtonManager.get_control().mujoco.ctrl).view(n,-1)
        self.feet=FootClearance(self.m,self.d,self.device)
        self.contact_sensor=FootContactForces(self.m,self.s.mjw_model,self.d,self.feet,self.device)
        foot_bodies=[int(self.m.geom_bodyid[ids[0]]) for ids,_,_ in self.feet.feet]
        self.foot_bodies=torch.tensor(foot_bodies,device=self.device)
        self.body_positions=wp.to_torch(self.d.xpos)
        body_names=[mujoco.mj_id2name(self.m,mujoco.mjtObj.mjOBJ_BODY,i) or '' for i in range(self.m.nbody)]
        head_ids=[i for i,name in enumerate(body_names) if name=='head' or name.endswith('_head') or name.endswith('/head')]
        assert len(head_ids)==1,body_names
        self.head_id=head_ids[0]
        self.reward_model=G1MuscleRewards(n,self.device,self.dt)
        self.joint_acceleration=wp.to_torch(self.d.qacc)
        self.joint_torque=wp.to_torch(self.d.qfrc_actuator)
        self.num_actions=self.m.nu;self.num_envs=n
        dofs=self.s.mjc_jnt_to_newton_dof.numpy().reshape(n,-1)[0]
        self.src=list(range(7));self.dst=list(range(7))
        anatomical_joint_ids={}
        for j in range(REFERENCE.njnt):
            if REFERENCE.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE:continue
            name=mujoco.mj_id2name(REFERENCE,mujoco.mjtObj.mjOBJ_JOINT,j).replace('-','_')
            k=np.flatnonzero(dofs==JOINT_DOF[name]);assert len(k)==1
            anatomical_joint_ids[name]=int(k[0])
            self.src.append(int(REFERENCE.jnt_qposadr[j]));self.dst.append(int(self.m.jnt_qposadr[int(k[0])]))
        self.groups=joint_groups(self.m,self.device,anatomical_joint_ids)
        pose=np.load(ROOT/'outputs/isaac_velocity/assets/initial.npz')['qpos']
        initial=self.m.qpos0.copy();initial[self.dst]=pose[self.src]
        self.initial=torch.tensor(initial,dtype=torch.float32,device=self.device)
        initial_data=mujoco.MjData(self.m);initial_data.qpos[:]=initial;mujoco.mj_forward(self.m,initial_data)
        self.initial_head_gap=float(initial_data.xpos[self.head_id,2]-initial[2])
        assert self.initial_head_gap>0.
        q0=self.initial[3:7]
        self.initial_yaw=torch.atan2(2*(q0[0]*q0[3]+q0[1]*q0[2]),1-2*(q0[2]**2+q0[3]**2))
        self.steps=torch.zeros(n,dtype=torch.long,device=self.device)
        self.command=torch.zeros(n,device=self.device);self.phase=torch.zeros(n,device=self.device)
        self.previous=torch.zeros(n,self.num_actions,device=self.device)
        self.max_steps=500;self.speed_max=1.;self.autoreset=True
        self.last_metrics={};self.fall_count=0;self.completed=[];self.control_steps=0
        self.training_episodes=0
        self.return_buf=torch.zeros(n,device=self.device)
        self.pose_target=self.initial[7:].clone()
        self.reset(torch.ones(n,dtype=torch.bool,device=self.device))

    def reset(self,mask,randomize=True):
        ids=mask.nonzero().flatten()
        if ids.numel()==0:return
        world_mask=torch.cat([mask,torch.zeros(1,dtype=torch.bool,device=self.device)])
        self.s.reset(NewtonManager.get_state_0(),wp.from_torch(world_mask,dtype=wp.bool),flags=0)
        self.q[ids]=self.initial;self.v[ids]=0.;self.act[ids]=0.;self.ctrl[ids]=0.
        if randomize:
            self.v[ids,:6]=.01*torch.randn(ids.numel(),6,device=self.device)
            self.command[ids]=torch.rand(ids.numel(),device=self.device)*self.speed_max
            self.command[ids]=torch.where(torch.rand(ids.numel(),device=self.device)<.1,0.,self.command[ids])
            self.phase[ids]=2*math.pi*torch.rand(ids.numel(),device=self.device)
        self.reward_model.reset(mask)
        self.steps[ids]=0;self.previous[ids]=0.;self.return_buf[ids]=0.
        mjw.forward(self.s.mjw_model,self.d)
        state=NewtonManager.get_state_0();self.s._update_newton_state(self.s.model,state,self.d,state_prev=state)

    def heading_error(self):
        q=self.q[:,3:7]
        yaw=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
        delta=yaw-self.initial_yaw
        return torch.atan2(torch.sin(delta),torch.cos(delta))

    def observation(self):
        q=self.q[:,3:7]
        gravity=torch.stack([2*(q[:,1]*q[:,3]-q[:,0]*q[:,2]),2*(q[:,2]*q[:,3]+q[:,0]*q[:,1]),1-2*(q[:,1]**2+q[:,2]**2)],dim=-1)
        obs=torch.cat([self.q[:,2:3],gravity,self.q[:,7:]-self.pose_target,
                       self.v[:,:6],self.v[:,6:]*.1,self.act,self.previous,
                       self.command[:,None],torch.sin(self.phase)[:,None],torch.cos(self.phase)[:,None],torch.cos(self.heading_error())[:,None],torch.sin(self.heading_error())[:,None],self.support_t[:,None]],dim=1)
        return TensorDict({'policy':torch.nan_to_num(obs).clamp(-20,20)},batch_size=[self.n])

    def step(self,action):
        # Excitation is bounded without hard action clipping in the PPO distribution.
        excitation=torch.sigmoid(action*1.5-2.2)
        foot_before=self.body_positions[:,self.foot_bodies,:2].clone()
        self.ctrl.copy_(excitation)
        for _ in range(self.decimation):self.sim.step(render=False)
        if bool((wp.to_torch(self.d.overflow).to(torch.int64)&511).any()):
            raise RuntimeError('MuJoCo Warp capacity overflow')
        self.steps+=1;self.phase+=2*math.pi*1.1*self.dt
        finite=torch.isfinite(self.q).all(1)&torch.isfinite(self.v).all(1)&torch.isfinite(self.act).all(1)
        q=self.q[:,3:7];up=1-2*(q[:,1]**2+q[:,2]**2)
        root_velocity=root_world_velocity(self.q,self.v)
        heading_error=self.heading_error()
        velocity_xy=forward_velocity(root_velocity,heading_error)
        velocity_error=(velocity_xy[:,0]-self.command).abs()
        clearance=self.feet()
        contact=self.contact_sensor()
        foot_velocity=(self.body_positions[:,self.foot_bodies,:2]-foot_before)/self.dt
        gravity_xy=torch.stack([2*(q[:,1]*q[:,3]-q[:,0]*q[:,2]),
                                2*(q[:,2]*q[:,3]+q[:,0]*q[:,1])],dim=1)
        head_gap=self.body_positions[:,self.head_id,2]-self.q[:,2]
        collapsed=torso_collapsed(head_gap,self.initial_head_gap)
        fallen=(self.q[:,2]<.60)|(self.q[:,2]>1.35)|(up<.45)|collapsed|(~finite)
        timeout=self.steps>=self.max_steps;done=fallen|timeout
        g=self.groups
        ankle=self.q[:,g['ankle']]
        reward,terms,weighted=self.reward_model(
            velocity_xy=velocity_xy,command=self.command,root_velocity=root_velocity,
            gravity_xy=gravity_xy,fallen=fallen,contact=contact,clearance=clearance,
            foot_velocity_xy=foot_velocity,
            ankle_violation=(g['ankle_low']-ankle).clamp(min=0)+(ankle-g['ankle_high']).clamp(min=0),
            hip_deviation=self.q[:,g['hip']]-self.initial[g['hip']],
            arm_deviation=self.q[:,g['arms']]-self.initial[g['arms']],
            torso_deviation=self.q[:,g['torso']]-self.initial[g['torso']],
            joint_acceleration=self.joint_acceleration[:,g['leg_dof']],
            joint_torque=self.joint_torque[:,g['leg_dof']],
            excitation=excitation,previous=self.previous,phase=self.phase)
        reward=torch.where(finite,reward,torch.full_like(reward,-4.))
        self.previous.copy_(excitation);self.return_buf+=reward
        # Log once per PPO rollout; avoid per-term GPU synchronizations each step.
        self.control_steps+=1
        if self.control_steps%32==0 or not self.last_metrics:
            values=torch.stack([value.mean() for value in weighted.values()]).detach().cpu().tolist()
            self.last_metrics={f'reward/{key}':value for key,value in zip(weighted,values)}
            self.last_metrics.update(velocity_error=float(velocity_error.nan_to_num().mean()),
                height=float(self.q[:,2].nan_to_num().mean()),upright=float(up.nan_to_num().mean()),
                mean_excitation=float(excitation.mean()),foot_clearance_mean=float(clearance.mean()),
                single_support_fraction=float((contact.sum(1)==1).float().mean()),
                head_height_gap=float(head_gap.mean()),torso_collapse_fraction=float(collapsed.float().mean()),
                nonfinite=int((~finite).sum()))
        if done.any():
            if self.autoreset:self.training_episodes+=int(done.sum())
            for i in done.nonzero().flatten().tolist():
                self.completed.append({'seconds':float(self.steps[i]*self.dt),'return':float(self.return_buf[i]),'fall':bool(fallen[i])})
            self.completed=self.completed[-5000:]
            self.fall_count+=int(fallen.sum())
            if self.autoreset:self.reset(done)
        return self.observation(),reward,done,{'time_outs':timeout&~fallen}

    def evaluate(self,actor,label,seconds=20.,support=None):
        if support not in (None,0.):raise ValueError('G1 variant is strictly unassisted')
        old_completed=self.completed.copy()
        was_training=actor.training;actor.eval();self.autoreset=False
        old_support=self.support_t.clone();self.support_t.fill_(self.support_level if support is None else support)
        self.reset(torch.ones(self.n,dtype=torch.bool,device=self.device),randomize=False)
        targets=torch.tensor([0.,.4,.8,1.2],device=self.device)
        self.command.copy_(targets[torch.arange(self.n,device=self.device)%4]);self.phase.zero_()
        sequence=[float(x) for x in a.eval_command_sequence.split(',')] if a.eval_command_sequence else None
        if sequence:self.command.fill_(sequence[0])
        if a.eval_phase=='random':self.phase.uniform_(0.,2*math.pi)
        if a.eval_velocity_perturbation:
            generator=torch.Generator(device=self.device).manual_seed(1729)
            self.v[:,:6]+=a.eval_velocity_perturbation*torch.randn(self.n,6,device=self.device,generator=generator)
            mjw.forward(self.s.mjw_model,self.d)
            state=NewtonManager.get_state_0();self.s._update_newton_state(self.s.model,state,self.d,state_prev=state)
        self.max_steps=int(seconds/self.dt)+1
        alive=torch.ones(self.n,dtype=torch.bool,device=self.device);lifetime=torch.zeros(self.n,device=self.device)
        nodes,weights=np.polynomial.hermite.hermgauss(9)
        nodes=torch.tensor(nodes,dtype=torch.float32,device=self.device)
        weights=torch.tensor(weights/math.sqrt(math.pi),dtype=torch.float32,device=self.device)
        expected_force=float(self.support_t[0])*float(np.sum(REFERENCE.body_mass))*9.81
        force_error=torch.zeros((),device=self.device);wrench_max=torch.zeros((),device=self.device)
        extra_force_max=torch.zeros((),device=self.device)
        segment_errors=torch.zeros(len(sequence or [0]),device=self.device)
        segment_counts=torch.zeros_like(segment_errors);segment_survival=[]
        errors=torch.zeros(self.n,device=self.device);poses=[];acts=[];resets=[];speeds=[];commands=[];foot_heights=[];phases=[];pending_reset=False;view=min(1,self.n-1)
        with torch.inference_mode():
            for i in range(int(seconds/self.dt)):
                stage=i*len(sequence)//int(seconds/self.dt) if sequence else 0
                if sequence:self.command.fill_(sequence[stage])
                action=actor(self.observation(),stochastic_output=a.stochastic_eval)
                if a.mean_excitation_eval:
                    if a.stochastic_eval:raise ValueError('Select stochastic or mean excitation evaluation, not both')
                    actor.distribution.update(action)
                    std=actor.distribution.std
                    expected=sum(w*torch.sigmoid(1.5*(action+math.sqrt(2)*x*std)-2.2) for x,w in zip(nodes,weights))
                    action=(torch.logit(expected.clamp(1e-6,1-1e-6))+2.2)/1.5
                obs,_,done,_=self.step(action)
                error=(forward_velocity(self.v,self.heading_error())[:,0]-self.command).abs().nan_to_num()
                errors+=error*alive
                segment_errors[stage]+=(error*alive).sum();segment_counts[stage]+=alive.sum()
                lifetime+=alive*self.dt;alive&=~done
                if sequence and (i+1)*len(sequence)//int(seconds/self.dt)!=stage:segment_survival.append(float(alive.float().mean()))
                force=wp.to_torch(self.d.xfrc_applied)
                force_error=torch.maximum(force_error,(force[:,:,2].sum(1)-expected_force).abs().max())
                wrench_max=torch.maximum(wrench_max,force.abs().max())
                extra_force_max=torch.maximum(extra_force_max,force[:,:,[0,1,3,4,5]].abs().max())
                if i%2==0:
                    pose=REFERENCE.qpos0.copy();pose[self.src]=self.q[view,self.dst].cpu().numpy()
                    poses.append(pose);acts.append(self.act[view].cpu().numpy());resets.append(pending_reset);pending_reset=False
                    speeds.append(float(forward_velocity(self.v,self.heading_error())[view,0]));commands.append(float(self.command[view]))
                    foot_heights.append(self.feet()[view].cpu().numpy());phases.append(float(self.phase[view]))
                # Freeze failed environments by resetting them, but never restore alive.
                pending_reset=pending_reset or bool(done[view])
                if done.any():self.reset(done,randomize=False)
        result={'label':label,'termination_version':'g1_muscle_v2_torso','minimum_head_gap_m':.6*self.initial_head_gap,'stochastic':a.stochastic_eval,'mean_excitation':a.mean_excitation_eval,'initial_phase':a.eval_phase,'initial_velocity_perturbation_std':a.eval_velocity_perturbation,'support':float(self.support_t[0]),'commands':{}}
        for j,target in enumerate([0.,.4,.8,1.2]):
            mask=torch.arange(self.n,device=self.device)%4==j
            if mask.any():result['commands'][str(target)]={'survival':float(alive[mask].float().mean()),'mean_seconds':float(lifetime[mask].mean()),'velocity_mae':float((errors[mask]/(lifetime[mask]/self.dt).clamp(min=1)).mean())}
        assert float(force_error)<.001 and float(extra_force_max)<.001,(float(force_error),float(extra_force_max))
        if expected_force==0.:assert float(wrench_max)==0.,float(wrench_max)
        result['external_force_check']={'expected_upward_N':expected_force,'max_sum_error_N':float(force_error),'max_external_wrench_component':float(wrench_max),'max_other_component':float(extra_force_max)}
        if sequence:
            result.pop('commands')
            result['command_schedule']={'targets_mps':sequence,'segment_seconds':seconds/len(sequence)}
            result['sequence_metrics']={'survival':float(alive.float().mean()),'mean_seconds':float(lifetime.mean()),'velocity_mae':float((errors/(lifetime/self.dt).clamp(min=1)).mean()),'segment_survival':segment_survival,'segment_velocity_mae':[float(e/c) if c>0 else None for e,c in zip(segment_errors,segment_counts)]}
        folder=a.run_dir/'evaluations'/label;folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'rollout.npz',qpos=poses,act=acts,reset=resets,velocity=speeds,command=commands,foot_clearance=foot_heights,phase=phases,dt=.04)
        (folder/'model.xml').write_text(Path(os.environ['MM_MODEL_XML']).read_text())
        (folder/'metrics.json').write_text(json.dumps(result,indent=2))
        fraction=float(self.support_t[0])
        (folder/'assistance.json').write_text(json.dumps(dict(body='head',fraction_body_weight=fraction,force_world_N=[0.,0.,fraction*float(np.sum(REFERENCE.body_mass))*9.81])))
        self.support_t.copy_(old_support);self.max_steps=500;self.autoreset=True
        self.reset(torch.ones(self.n,dtype=torch.bool,device=self.device));actor.train(was_training)
        self.completed=old_completed
        return result,folder


def main():
    (a.run_dir/'reward_config.json').write_text(json.dumps(dict(version='g1_muscle_v4_phase_clearance',rate_weights=WEIGHTS,touchdown_event_weight=TOUCHDOWN_WEIGHT,support=0.,minimum_head_gap_fraction=.6),indent=2))
    (a.run_dir/'config.json').write_text(json.dumps(vars(a),default=str,indent=2))
    cfg=NewtonCfg(solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=True,update_data_interval=0,integrator='euler',iterations=4,ls_iterations=8,nconmax=256,njmax=1024),use_cuda_graph=True,num_substeps=1)
    cfg.class_type=NewtonVelocityManager
    with build_simulation_context(sim_cfg=SimulationCfg(dt=.002,device='cuda:0',physics=cfg)) as sim:
        env=VelocityEnv(sim,a.num_envs);obs=env.observation()
        actor=MLPModel(obs,{'actor':['policy']},'actor',env.num_actions,hidden_dims=[512,256,256],obs_normalization=True,distribution_cfg={'class_name':'rsl_rl.modules.GaussianDistribution','init_std':.5,'std_type':'log','std_range':[.05,1.]}).to('cuda:0')
        critic=MLPModel(obs,{'critic':['policy']},'critic',1,hidden_dims=[512,256,256],obs_normalization=True).to('cuda:0')
        if not a.legacy_support_normalization:
            for network in [actor,critic]:
                network.obs_normalizer=VelocityNormalization(obs['policy'].shape[-1]).to('cuda:0')
        horizon=32;storage=RolloutStorage('rl',env.n,horizon,obs,[env.num_actions],device='cuda:0')
        alg=PPO(actor,critic,storage,device='cuda:0',num_learning_epochs=4,num_mini_batches=4,learning_rate=3e-4,desired_kl=.02,entropy_coef=a.entropy_coef,gamma=.99,lam=.95)
        start_it=0;schedule=None
        if a.resume:
            saved=torch.load(a.resume,weights_only=False)
            if saved.get('reward_version') not in ('g1_muscle_v1','g1_muscle_v2_torso','g1_muscle_v3_movement','g1_muscle_v4_phase_clearance') or saved.get('support')!=0.:
                raise ValueError('Resume requires an unassisted G1 reward checkpoint; start this experiment from scratch')
            alg.load(saved,None,True);start_it=saved['iteration']+1
            env.speed_max=saved.get('speed_max',.8)
            env.support_level=saved.get('support',a.support);env.support_t.fill_(env.support_level)
            schedule=saved.get('exploration_schedule')
        if a.exploration_std_final is not None and schedule is None:
            initial=float(actor.distribution.log_std_param.exp().clamp(.05,1.).max())
            if a.exploration_std_final>initial:raise ValueError('Exploration annealing must not increase initial std')
            schedule=dict(start_iteration=start_it,initial=initial,final=a.exploration_std_final,updates=a.exploration_anneal_updates)
        elif a.exploration_std_final is not None and schedule['final']!=a.exploration_std_final:
            raise ValueError('Resume preserves the saved exploration schedule; do not silently replace it')
        if a.eval_only:
            if not a.resume:raise ValueError('--eval-only requires --resume')
            result,folder=env.evaluate(actor,'evaluation',a.eval_seconds,support=a.eval_support)
            print('EVALUATION',json.dumps(result),flush=True)
            subprocess.run([str(ROOT/'.venv/bin/python'),str(Path(__file__).with_name('render_velocity.py')),str(folder)],env={**os.environ,'MUJOCO_GL':'egl'},check=True)
            return
        writer=SummaryWriter(str(a.run_dir/'tensorboard'));start=time.monotonic();last_eval=start;last_save=start;last_video=start;eval_count=0
        def save(it,name):
            state=alg.save();state.update(reward_version='g1_muscle_v4_phase_clearance',iteration=it,support=env.support_level,speed_max=env.speed_max,config=vars(a),exploration_schedule=schedule,training_episodes_this_run=env.training_episodes);tmp=a.run_dir/(name+'.tmp');torch.save(state,tmp);tmp.replace(a.run_dir/name)
        def render(folder):
            with (folder/'render.log').open('w') as log:
                subprocess.Popen([str(ROOT/'.venv/bin/python'),str(Path(__file__).with_name('render_velocity.py')),str(folder)],env={**os.environ,'MUJOCO_GL':'egl'},stdout=log,stderr=subprocess.STDOUT)
        print('READY',json.dumps({'n':env.n,'obs':obs['policy'].shape[1],'actions':env.num_actions}),flush=True)
        if a.smoke:
            before=env.q.clone()
            for _ in range(100):env.step(torch.zeros(env.n,env.num_actions,device='cuda:0'))
            assert torch.isfinite(env.q).all();assert (env.q-before).abs().max()>1e-5
            reset_mask=torch.arange(env.n,device='cuda:0')%2==0;unselected=env.q[~reset_mask].clone()
            env.reset(reset_mask);assert torch.allclose(env.q[reset_mask],env.initial.expand(int(reset_mask.sum()),-1));assert torch.equal(env.q[~reset_mask],unselected)
            print('SMOKE_RESET_PASS',flush=True)
            cpu=mujoco.MjData(env.m);cpu.qpos[:]=env.q[0].cpu().numpy();mujoco.mj_forward(env.m,cpu)
            floor=int(np.flatnonzero(env.m.geom_type==int(mujoco.mjtGeom.mjGEOM_PLANE))[0])
            expected=[min(mujoco.mj_geomDistance(env.m,cpu,floor,g,10.,None) for g in ids) for ids,_,_ in env.feet.feet]
            actual=env.feet()[0].cpu().numpy()
            assert np.max(np.abs(actual-expected))<5e-5,(actual,expected)
            print('SMOKE_FOOT_CLEARANCE_PASS',actual.tolist(),flush=True)
        obs=env.observation()
        env.training_episodes=0
        for it in range(start_it,start_it+a.iterations):
            std_cap=cap_exploration(actor.distribution,schedule,it)
            with torch.inference_mode():
                for _ in range(horizon):
                    actions=alg.act(obs);obs,rewards,dones,extras=env.step(actions)
                    if not torch.isfinite(obs['policy']).all():raise RuntimeError('Nonfinite observations')
                    alg.process_env_step(obs,rewards,dones,extras)
                alg.compute_returns(obs)
            loss=alg.update()
            # Save/evaluate the same capped distribution used for the next rollout.
            cap_exploration(actor.distribution,schedule,it)
            now=time.monotonic()
            if it%10==0:
                recent=env.completed[-1000:]
                record={'iteration':it,'wall_seconds':now-start,'env_steps':(it-start_it+1)*horizon*env.n,'support':env.support_level,'learning_rate':alg.learning_rate,'loss':loss,**env.last_metrics,
                        'training_episodes_this_run':env.training_episodes,'exploration_std_cap':std_cap,'exploration_std_mean':float(actor.distribution.log_std_param.exp().clamp(.05,1.).mean()),
                        'episode_seconds':np.mean([r['seconds'] for r in recent]) if recent else 0.,'episode_return':np.mean([r['return'] for r in recent]) if recent else 0.}
                print('TRAIN',json.dumps(record),flush=True)
                with (a.run_dir/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                for k,v in record.items():
                    if isinstance(v,(int,float)):writer.add_scalar(k,v,it)
                (a.run_dir/'status.json').write_text(json.dumps(record,indent=2))
            if now-last_save>=300 or it==start_it:
                save(it,'latest.pt');last_save=now
            requested_eval=(a.run_dir/'EVALUATE').exists()
            if requested_eval or now-last_eval>=(a.first_eval_seconds if eval_count==0 else a.eval_interval) or (a.smoke and it==start_it+a.iterations-1):
                save(it,f'model_{it}.pt')
                result,folder=env.evaluate(actor,f'iter_{it:06d}',a.eval_seconds)
                print('EVALUATION',json.dumps(result),flush=True)
                with (a.run_dir/'evaluations.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
                render_this=eval_count==0 or requested_eval or now-last_video>=a.video_interval
                if render_this:render(folder)
                if render_this:last_video=time.monotonic()
                if requested_eval:(a.run_dir/'EVALUATE').unlink(missing_ok=True)
                save(it,'latest.pt');last_save=time.monotonic()
                eval_count+=1;last_eval=time.monotonic();obs=env.observation()
            if (a.run_dir/'STOP').exists():
                save(it,'stopped.pt');print('STOP_REQUESTED',flush=True);break
        save(it,'latest.pt');writer.close()

if __name__=='__main__':main()
