"""Experimental assistance/task curriculum for G1 muscle PPO (separate runs).

Checkpoints, hourly evaluations and videos are local. Optional reference states
initialize training episodes; evaluation always begins from the upright pose.
"""
import argparse, json, math, os, time, subprocess, re, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser()
p.add_argument('--num-envs',type=int,default=1024)
p.add_argument('--solver-iterations',type=int,default=4)
p.add_argument('--iterations',type=int,default=100000)
p.add_argument('--run-dir',type=Path,default=ROOT/'outputs/isaac_velocity_g1/run_001')
p.add_argument('--eval-seconds',type=float,default=20.)
p.add_argument('--episode-seconds',type=float,default=10.,help='Training episode time limit; evaluation uses --eval-seconds')
p.add_argument('--eval-interval',type=float,default=3600.)
p.add_argument('--video-interval',type=float,default=3600.)
p.add_argument('--first-eval-seconds',type=float,default=300.)
p.add_argument('--support',type=float,default=0.)
p.add_argument('--root-assistance',type=float,default=0.,help='Bounded pelvis PD strength, 0..1; not a bodyweight fraction')
p.add_argument('--eval-root-assistance',type=float,default=0.)
p.add_argument('--root-weight-support',type=float,default=.3,help='Vertical gravity feedforward as bodyweight fraction')
p.add_argument('--root-vertical-limit',type=float,default=.6,help='Absolute vertical force cap as bodyweight fraction')
p.add_argument('--command-min',type=float,default=.4)
p.add_argument('--command-max',type=float,default=.4)
p.add_argument('--standing-probability',type=float,default=0.)
p.add_argument('--reset-velocity-noise',type=float,default=.01)
p.add_argument('--reference-states',type=Path)
p.add_argument('--reference-reset-probability',type=float,default=0.)
p.add_argument('--phase-harmonics',type=int,default=0,help='Optional direct periodic branch in the Gaussian muscle-action mean')
p.add_argument('--warm-start-phase',action='store_true',help='Add a zero periodic branch to a full-action checkpoint; reset PPO optimizer')
p.add_argument('--action-basis',type=Path,help='Orthonormal muscle-logit residual basis with actuator_names')
p.add_argument('--baseline-checkpoint',type=Path,help='Frozen full-action G1 policy, used only with --action-basis')
p.add_argument('--pose-targets',type=Path,help='Periodic joint-pose coefficients used only for reward guidance')
p.add_argument('--pose-weight',type=float,default=0.)
p.add_argument('--cartesian-targets',type=Path)
p.add_argument('--cartesian-weight',type=float,default=0.)
p.add_argument('--cartesian-positive-score',action='store_true',help='Use nonnegative reference similarity instead of subtracting a standing baseline; changes survival incentives')
p.add_argument('--phase-frequency',type=float,default=1.1)
p.add_argument('--trunk-height-weight',type=float,default=0.,help='Dense upright trunk reward; independent of pelvis height')
p.add_argument('--eval-at-end',action='store_true')
p.add_argument('--entropy-coef',type=float,default=0.)
p.add_argument('--reset-exploration-std',type=float,help='Explicitly replace resumed Gaussian std and schedule; preserve mean policy')
p.add_argument('--exploration-std-final',type=float,help='Anneal the latent Gaussian std cap; no added physical assistance')
p.add_argument('--exploration-anneal-updates',type=int,default=600)
p.add_argument('--resume',type=Path)
p.add_argument('--smoke',action='store_true')
p.add_argument('--ppo-diagnostics',action='store_true')
p.add_argument('--desired-kl',type=float,default=.02)
p.add_argument('--eval-only',action='store_true')
p.add_argument('--no-render',action='store_true',help='Skip video rendering; retain evaluation metrics and recorded states')
p.add_argument('--eval-reference-start',action='store_true',help='Diagnostic eval-only initialization from the reference bank; not an upright-start test')
p.add_argument('--stochastic-eval',action='store_true')
p.add_argument('--mean-excitation-eval',action='store_true')
p.add_argument('--eval-phase',choices=['zero','random'],default='zero')
p.add_argument('--eval-support',type=float,default=0.)
p.add_argument('--eval-velocity-perturbation',type=float,default=0.)
p.add_argument('--eval-command-sequence',help='Comma-separated targets, equal-duration segments; eval-only')
p.add_argument('--legacy-support-normalization',action='store_true')
a=p.parse_args()
if not math.isfinite(a.episode_seconds) or not 1<=a.episode_seconds<=120:p.error('Training episode duration must be between 1 and 120 seconds')
if not 0<=a.phase_harmonics<=8:p.error('Invalid phase harmonics')
if not math.isfinite(a.desired_kl) or a.desired_kl<=0:p.error('Invalid desired KL')
if a.phase_harmonics and (a.action_basis or a.legacy_support_normalization):p.error('Phase branch requires full actions and fixed support normalization')
if a.warm_start_phase and not (a.phase_harmonics and a.resume):p.error('Phase warm start requires harmonics and resume checkpoint')
if not 1<=a.solver_iterations<=200:p.error('Solver iterations must be in [1,200]')
if not 0<=a.support<=.5 or not 0<=a.eval_support<=.5:p.error('Support must be in [0,.5]')
if not 0<=a.eval_root_assistance<=a.root_assistance<=1:p.error('Invalid root assistance strength')
if not math.isfinite(a.root_weight_support) or not 0<=a.root_weight_support<=1.5:p.error('Invalid root weight support')
if not math.isfinite(a.root_vertical_limit) or not 0<a.root_vertical_limit<=2:p.error('Invalid vertical force limit')
if a.root_assistance>0 and a.support>0:p.error('Use either head support or root assistance')
if a.eval_support>0 and a.support==0:p.error('Positive evaluation support requires a positive --support to register forces')
if not 0<=a.command_min<=a.command_max<=1.2:p.error('Invalid command range')
if not 0<=a.standing_probability<=1 or a.reset_velocity_noise<0:p.error('Invalid reset randomization')
if not 0<=a.reference_reset_probability<=1:p.error('Invalid reference reset probability')
if a.reference_reset_probability>0 and a.reference_states is None:p.error('Reference reset requires --reference-states')
if a.baseline_checkpoint and not a.action_basis:p.error('Baseline requires --action-basis')
if a.action_basis and not (a.baseline_checkpoint or a.resume):p.error('Basis requires baseline or basis-policy resume')
if a.action_basis and a.mean_excitation_eval:p.error('Mean-excitation quadrature is not implemented for the residual composite policy')
if not math.isfinite(a.pose_weight) or a.pose_weight<0:p.error('Invalid pose reward weight')
if not math.isfinite(a.phase_frequency) or not .1<=a.phase_frequency<=3:p.error('Invalid phase frequency')
if not math.isfinite(a.trunk_height_weight) or a.trunk_height_weight<0:p.error('Invalid trunk height weight')
if a.pose_weight>0 and a.pose_targets is None:p.error('Pose reward requires --pose-targets')
if not math.isfinite(a.cartesian_weight) or a.cartesian_weight<0:p.error('Invalid Cartesian reward weight')
if a.cartesian_weight>0 and a.cartesian_targets is None:p.error('Cartesian reward requires --cartesian-targets')
if a.exploration_std_final is not None and not .05<=a.exploration_std_final<=1.:p.error('--exploration-std-final must be in [.05,1]')
if a.reset_exploration_std is not None and (not a.resume or not math.isfinite(a.reset_exploration_std) or not .05<=a.reset_exploration_std<=1.):p.error('Exploration reset requires resume and std in [.05,1]')
if a.exploration_anneal_updates<=0:p.error('--exploration-anneal-updates must be positive')
if a.eval_command_sequence and not a.eval_only:p.error('--eval-command-sequence requires --eval-only')
if a.eval_reference_start and not (a.eval_only and a.reference_reset_probability==1. and a.eval_command_sequence and a.eval_phase=='zero'):p.error('Reference-start diagnostic requires eval-only, reference reset probability1, command sequence, and eval-phase zero (bank phase is retained)')
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
from velocity_exploration import cap_exploration,reset_exploration
from muscle_action_basis import MuscleActionBasis
from periodic_pose_reward import PeriodicPoseReward
from cartesian_reference_reward import CartesianReferenceReward
from phase_residual_actor import PhaseResidualActor
from ppo_update_diagnostics import update_with_diagnostics
from root_assistance import pelvis_wrench
from project_joint_equalities import project_joint_equalities
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

@wp.kernel
def curriculum_support_kernel(f:wp.array(dtype=wp.spatial_vector),support:wp.array(dtype=float),body:int,stride:int,mass_g:float):
    world=wp.tid();i=world*stride+body
    f[i]=f[i]+wp.spatial_vector(0.,0.,support[world]*mass_g,0.,0.,0.)

@wp.kernel
def pelvis_assistance_kernel(f:wp.array(dtype=wp.spatial_vector),wrench:wp.array(dtype=wp.spatial_vector),body:int,stride:int):
    world=wp.tid();i=world*stride+body
    f[i]=f[i]+wrench[world]

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
        self.root_strength=torch.full((n,),a.root_assistance,device=self.device)
        self.root_wrench=wp.zeros(n,dtype=wp.spatial_vector,device=self.device)
        self.root_wrench_t=wp.to_torch(self.root_wrench)
        template=make_builder();b=newton.ModelBuilder();SolverMuJoCo.register_custom_attributes(b)
        for _ in range(n):b.add_world(template)
        NewtonManager.set_builder(b)
        if a.support>0:
            head=[i for i,x in enumerate(template.body_label) if x.endswith('/head')]
            assert len(head)==1
            stride=len(template.body_label)
            def force_callback(state):
                wp.launch(curriculum_support_kernel,dim=n,inputs=[state.body_f,self.support,head[0],stride,float(np.sum(REFERENCE.body_mass)*9.81)],device=self.device)
            NewtonManager.register_state_force_callback(force_callback)
        if a.root_assistance>0:
            pelvis=[i for i,x in enumerate(template.body_label) if x.endswith('/pelvis')]
            assert len(pelvis)==1
            stride=len(template.body_label)
            def root_force_callback(state):
                wp.launch(pelvis_assistance_kernel,dim=n,inputs=[state.body_f,self.root_wrench,pelvis[0],stride],device=self.device)
            NewtonManager.register_state_force_callback(root_force_callback)
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
        pelvis_ids=[i for i,name in enumerate(body_names) if name=='pelvis' or name.endswith('_pelvis') or name.endswith('/pelvis')]
        assert len(pelvis_ids)==1
        self.pelvis_id=pelvis_ids[0]
        self.reward_model=G1MuscleRewards(n,self.device,self.dt)
        self.joint_acceleration=wp.to_torch(self.d.qacc)
        self.joint_torque=wp.to_torch(self.d.qfrc_actuator)
        self.num_actions=self.m.nu;self.num_envs=n
        self.action_transform=None;self.baseline_actor=None
        if a.action_basis:
            basis=np.load(a.action_basis,allow_pickle=False)
            expected=[mujoco.mj_id2name(REFERENCE,mujoco.mjtObj.mjOBJ_ACTUATOR,j) for j in range(REFERENCE.nu)]
            if basis['actuator_names'].tolist()!=expected:raise ValueError('Action basis actuator order mismatch')
            actual=[mujoco.mj_id2name(self.m,mujoco.mjtObj.mjOBJ_ACTUATOR,j) for j in range(self.m.nu)]
            if actual!=expected:raise ValueError('Native simulator actuator order differs from action basis')
            self.action_transform=MuscleActionBasis(basis['basis']).to(self.device)
            if self.action_transform.basis.shape[1]!=self.m.nu:raise ValueError('Wrong muscle count in basis')
            self.num_actions=self.action_transform.basis.shape[0]
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
        self.previous=torch.zeros(n,self.m.nu,device=self.device)
        self.training_max_steps=round(a.episode_seconds/self.dt)
        self.max_steps=self.training_max_steps;self.speed_max=1.;self.autoreset=True
        self.last_metrics={};self.fall_count=0;self.completed=[];self.control_steps=0
        self.training_episodes=0
        self.return_buf=torch.zeros(n,device=self.device)
        self.pose_target=self.initial[7:].clone()
        self.periodic_pose=None
        if a.pose_weight>0:
            targets=np.load(a.pose_targets,allow_pickle=False)
            target_ids=[int(self.m.jnt_qposadr[anatomical_joint_ids[name]]) for name in targets['joint_names'].tolist()]
            self.periodic_pose_ids=torch.tensor(target_ids,device=self.device,dtype=torch.long)
            self.periodic_pose=PeriodicPoseReward(targets['coefficients'],self.initial[self.periodic_pose_ids]).to(self.device)
        self.cartesian_reward=None
        if a.cartesian_weight>0:
            targets=np.load(a.cartesian_targets,allow_pickle=False)
            if abs(float(targets['phase_frequency'])-a.phase_frequency)>1e-6:raise ValueError('Cartesian phase frequency mismatch')
            site_names=[mujoco.mj_id2name(self.m,mujoco.mjtObj.mjOBJ_SITE,i) or '' for i in range(self.m.nsite)]
            def site_id(name):
                matches=[i for i,n in enumerate(site_names) if re.sub(r'_\d+$','',n.rsplit('/',1)[-1])==name]
                if len(matches)!=1:raise ValueError(f'Expected one mapped site for {name}: {matches}')
                return matches[0]
            mapped=[site_id(name) for name in targets['site_names'].tolist()]
            self.cartesian_site_ids=torch.tensor(mapped,dtype=torch.long,device=self.device)
            self.cartesian_pelvis_site=site_id('pelvis_mimic')
            standing=initial_data.site_xpos[mapped]-initial_data.site_xpos[self.cartesian_pelvis_site]
            if np.max(np.abs(standing-targets['standing_sites']))>1e-5:raise ValueError('Cartesian standing site mapping mismatch')
            self.cartesian_reward=CartesianReferenceReward(targets['coefficients'],standing,float(targets['sigma_m']),subtract_standing_baseline=not a.cartesian_positive_score).to(self.device)
            self.site_positions=wp.to_torch(self.d.site_xpos)
            (a.run_dir/'cartesian_mapping.json').write_text(json.dumps(dict(site_names=targets['site_names'].tolist(),site_ids=mapped,standing_max_error=float(np.max(np.abs(standing-targets['standing_sites'])))),indent=2))
        self.reference_bank=None
        if a.reference_reset_probability>0:
            bank=np.load(a.reference_states,allow_pickle=False)
            names=[mujoco.mj_id2name(REFERENCE,mujoco.mjtObj.mjOBJ_JOINT,j) for j in range(REFERENCE.njnt)]
            if bank['joint_names'].tolist()!=names:raise ValueError('Reference reset joint order mismatch')
            projected_q,projected_v=project_joint_equalities(REFERENCE,bank['qpos'],bank['qvel'])
            if np.max(np.abs(projected_q-bank['qpos']))>1e-5 or np.max(np.abs(projected_v-bank['qvel']))>1e-4:
                raise ValueError('Reference reset violates model joint equalities; prepare_projected_resets.py must project dependent qpos and qvel first')
            count=len(bank['qpos'])
            assert count>0 and bank['qpos'].shape==(count,REFERENCE.nq) and bank['qvel'].shape==(count,REFERENCE.nv)
            assert bank['phase'].shape==(count,) and bank['command'].shape==(count,)
            assert all(np.isfinite(bank[k]).all() for k in ['qpos','qvel','phase','command'])
            q=np.tile(initial,(count,1));q[:,self.dst]=bank['qpos'][:,self.src]
            vel=np.zeros((count,self.m.nv),dtype=np.float32)
            src_v=list(range(6))+[i-1 for i in self.src[7:]]
            dst_v=list(range(6))+[i-1 for i in self.dst[7:]]
            vel[:,dst_v]=bank['qvel'][:,src_v]
            self.reference_bank={k:torch.tensor(value,dtype=torch.float32,device=self.device)
                                 for k,value in dict(qpos=q,qvel=vel,phase=bank['phase'],command=bank['command']).items()}
        self.reset(torch.ones(n,dtype=torch.bool,device=self.device))

    def reset(self,mask,randomize=True):
        ids=mask.nonzero().flatten()
        if ids.numel()==0:return
        world_mask=torch.cat([mask,torch.zeros(1,dtype=torch.bool,device=self.device)])
        self.s.reset(NewtonManager.get_state_0(),wp.from_torch(world_mask,dtype=wp.bool),flags=0)
        self.q[ids]=self.initial;self.v[ids]=0.;self.act[ids]=0.;self.ctrl[ids]=0.
        if randomize:
            self.v[ids,:6]=a.reset_velocity_noise*torch.randn(ids.numel(),6,device=self.device)
            self.command[ids]=a.command_min+torch.rand(ids.numel(),device=self.device)*(a.command_max-a.command_min)
            self.command[ids]=torch.where(torch.rand(ids.numel(),device=self.device)<a.standing_probability,0.,self.command[ids])
            self.phase[ids]=2*math.pi*torch.rand(ids.numel(),device=self.device)
            if self.reference_bank is not None:
                selected=ids[torch.rand(ids.numel(),device=self.device)<a.reference_reset_probability]
                frames=torch.randint(len(self.reference_bank['qpos']),(selected.numel(),),device=self.device)
                self.q[selected]=self.reference_bank['qpos'][frames]
                self.v[selected]=self.reference_bank['qvel'][frames]
                self.command[selected]=self.reference_bank['command'][frames]
                self.phase[selected]=self.reference_bank['phase'][frames]
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
                       self.command[:,None],torch.sin(self.phase)[:,None],torch.cos(self.phase)[:,None],torch.cos(self.heading_error())[:,None],torch.sin(self.heading_error())[:,None],(self.support_t+self.root_strength)[:,None]],dim=1)
        return TensorDict({'policy':torch.nan_to_num(obs).clamp(-20,20)},batch_size=[self.n])

    def step(self,action):
        if self.action_transform is not None:
            if self.baseline_actor is None:raise RuntimeError('Frozen baseline not initialized')
            with torch.no_grad():
                baseline_obs=self.observation()
                if a.root_assistance>0:
                    # The frozen unassisted baseline never learned this new
                    # controller-strength feature. Only the residual sees it.
                    baseline_obs['policy'][:,-1]=0.
                baseline=self.baseline_actor(baseline_obs,stochastic_output=self.autoreset or a.stochastic_eval)
                action=self.action_transform(action,baseline)
        # Excitation is bounded without hard action clipping in the PPO distribution.
        excitation=torch.sigmoid(action*1.5-2.2)
        foot_before=self.body_positions[:,self.foot_bodies,:2].clone()
        self.ctrl.copy_(excitation)
        if a.root_assistance>0:
            heading=self.heading_error()
            target_xy=torch.stack([-heading.cos()*self.command,-heading.sin()*self.command],dim=-1)
            self.root_wrench_t.copy_(pelvis_wrench(self.q,root_world_velocity(self.q,self.v),self.initial[3:7],self.initial[2],target_xy,float(np.sum(REFERENCE.body_mass)),self.root_strength,weight_support=a.root_weight_support,vertical_limit=a.root_vertical_limit))
        for _ in range(self.decimation):self.sim.step(render=False)
        if bool((wp.to_torch(self.d.overflow).to(torch.int64)&511).any()):
            raise RuntimeError('MuJoCo Warp capacity overflow')
        self.steps+=1;self.phase+=2*math.pi*a.phase_frequency*self.dt
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
        self.touchdown_event=terms['touchdown'].bool()
        self.failure_components={'low_pelvis':self.q[:,2]<.60,'high_pelvis':self.q[:,2]>1.35,'root_tilt':up<.45,'trunk_fold':collapsed,'nonfinite':~finite}
        if self.periodic_pose is not None:
            weighted['reference_pose']=a.pose_weight*self.dt*self.periodic_pose(self.q[:,self.periodic_pose_ids],self.phase,self.command)
            reward=reward+weighted['reference_pose']
        if self.cartesian_reward is not None:
            weighted['reference_cartesian']=a.cartesian_weight*self.dt*self.cartesian_reward(self.site_positions[:,self.cartesian_site_ids],self.site_positions[:,self.cartesian_pelvis_site],self.heading_error(),self.phase,self.command)
            reward=reward+weighted['reference_cartesian']
        if a.trunk_height_weight>0:
            weighted['trunk_height']=a.trunk_height_weight*self.dt*torch.exp(-((head_gap/self.initial_head_gap-1.)/.15).square())
            reward=reward+weighted['trunk_height']
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
            if self.autoreset:
                for i in done.nonzero().flatten().tolist():
                    self.completed.append({'seconds':float(self.steps[i]*self.dt),'return':float(self.return_buf[i]),'fall':bool(fallen[i])})
            self.completed=self.completed[-5000:]
            self.fall_count+=int(fallen.sum())
            if self.autoreset:self.reset(done)
        return self.observation(),reward,done,{'time_outs':timeout&~fallen}

    def evaluate(self,actor,label,seconds=20.,support=None,root_strength=0.):
        if not 0<=root_strength<=a.root_assistance:raise ValueError('Invalid evaluation root assistance')
        if support is not None and not 0<=support<=a.support:raise ValueError('Evaluation support exceeds configured training maximum')
        old_completed=self.completed.copy()
        was_training=actor.training;actor.eval();self.autoreset=False
        old_support=self.support_t.clone();self.support_t.fill_(self.support_level if support is None else support)
        old_root_strength=self.root_strength.clone();self.root_strength.fill_(root_strength)
        self.reset(torch.ones(self.n,dtype=torch.bool,device=self.device),randomize=a.eval_reference_start)
        reference_start_check=None
        if a.eval_reference_start:
            distance=(self.q[:,None,:]-self.reference_bank['qpos'][None,:,:]).abs().amax(-1)
            closest=distance.argmin(1)
            q_error=float(distance.min(1).values.max());v_error=float((self.v-self.reference_bank['qvel'][closest]).abs().max());phase_error=float((self.phase-self.reference_bank['phase'][closest]).abs().max())
            assert max(q_error,v_error,phase_error)<1e-6
            reference_start_check=dict(qpos_max_error=q_error,qvel_max_error=v_error,phase_max_error=phase_error,bank_indices=closest.cpu().tolist())
        else:
            targets=torch.tensor([0.,.4,.8,1.2],device=self.device)
            self.command.copy_(targets[torch.arange(self.n,device=self.device)%4]);self.phase.zero_()
        sequence=[float(x) for x in a.eval_command_sequence.split(',')] if a.eval_command_sequence else None
        if sequence:self.command.fill_(sequence[0])
        if a.eval_phase=='random' and not a.eval_reference_start:self.phase.uniform_(0.,2*math.pi)
        if a.eval_velocity_perturbation:
            generator=torch.Generator(device=self.device).manual_seed(1729)
            self.v[:,:6]+=a.eval_velocity_perturbation*torch.randn(self.n,6,device=self.device,generator=generator)
            mjw.forward(self.s.mjw_model,self.d)
            state=NewtonManager.get_state_0();self.s._update_newton_state(self.s.model,state,self.d,state_prev=state)
        self.max_steps=int(seconds/self.dt)+1
        initial_eval_command=self.command.clone()
        alive=torch.ones(self.n,dtype=torch.bool,device=self.device);lifetime=torch.zeros(self.n,device=self.device)
        landing_count=torch.zeros(self.n,device=self.device)
        landing_chain=torch.zeros_like(landing_count);max_chain=torch.zeros_like(landing_count)
        last_landing=torch.full_like(landing_count,-100.)
        failure_counts={}
        nodes,weights=np.polynomial.hermite.hermgauss(9)
        nodes=torch.tensor(nodes,dtype=torch.float32,device=self.device)
        weights=torch.tensor(weights/math.sqrt(math.pi),dtype=torch.float32,device=self.device)
        expected_force=float(self.support_t[0])*float(np.sum(REFERENCE.body_mass))*9.81
        force_error=torch.zeros((),device=self.device);wrench_max=torch.zeros((),device=self.device)
        extra_force_max=torch.zeros((),device=self.device)
        segment_errors=torch.zeros(len(sequence or [0]),device=self.device)
        segment_counts=torch.zeros_like(segment_errors);segment_survival=[]
        errors=torch.zeros(self.n,device=self.device);poses=[];acts=[];resets=[];speeds=[];commands=[];foot_heights=[];phases=[];wrenches=[];pending_reset=False;view=min(1,self.n-1)
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
                event=self.touchdown_event & alive
                landing_count+=event
                landing_chain=torch.where(event,torch.where(i*self.dt-last_landing<=1.2,landing_chain+1,1.),landing_chain)
                last_landing=torch.where(event,i*self.dt,last_landing)
                max_chain=torch.maximum(max_chain,landing_chain)
                for key,mask in self.failure_components.items():
                    failure_counts[key]=failure_counts.get(key,torch.zeros_like(landing_count))+(mask & alive & done)
                lifetime+=alive*self.dt;alive&=~done
                if sequence and (i+1)*len(sequence)//int(seconds/self.dt)!=stage:segment_survival.append(float(alive.float().mean()))
                force=wp.to_torch(self.d.xfrc_applied)
                if a.root_assistance>0:
                    force_error=torch.maximum(force_error,(force[:,self.pelvis_id,:]-self.root_wrench_t).abs().max())
                    extra_force_max=torch.maximum(extra_force_max,(force.abs().sum(1)-force[:,self.pelvis_id,:].abs()).abs().max())
                else:
                    force_error=torch.maximum(force_error,(force[:,:,2].sum(1)-expected_force).abs().max())
                    extra_force_max=torch.maximum(extra_force_max,force[:,:,[0,1,3,4,5]].abs().max())
                wrench_max=torch.maximum(wrench_max,force.abs().max())
                if i%2==0:
                    pose=REFERENCE.qpos0.copy();pose[self.src]=self.q[view,self.dst].cpu().numpy()
                    poses.append(pose);acts.append(self.act[view].cpu().numpy());resets.append(pending_reset);pending_reset=False
                    speeds.append(float(forward_velocity(self.v,self.heading_error())[view,0]));commands.append(float(self.command[view]))
                    foot_heights.append(self.feet()[view].cpu().numpy());phases.append(float(self.phase[view]))
                    wrenches.append(force[view].sum(0).cpu().numpy().copy())
                # Freeze failed environments by resetting them, but never restore alive.
                pending_reset=pending_reset or bool(done[view])
                if done.any():self.reset(done,randomize=a.eval_reference_start)
        result={'label':label,'termination_version':'g1_muscle_v2_torso','minimum_head_gap_m':.6*self.initial_head_gap,'stochastic':a.stochastic_eval,'mean_excitation':a.mean_excitation_eval,'initial_phase':a.eval_phase,'initial_velocity_perturbation_std':a.eval_velocity_perturbation,'support':float(self.support_t[0]),'commands':{}}
        if self.action_transform is not None:
            result['action_transform']={'type':'frozen_baseline_plus_muscle_basis','learned_outputs':self.num_actions,'muscles':self.m.nu,'baseline_stochastic':a.stochastic_eval,'note':'Baseline retains its fixed Gaussian noise when stochastic; only residual coefficients are optimized.'}
        result['initial_state']='reference' if a.eval_reference_start else 'upright'
        if reference_start_check is not None:
            result['initial_phase']='reference_bank'
            result['reference_start_check']=reference_start_check
        result['phase_harmonics']=a.phase_harmonics
        result['solver_iterations']=int(self.m.opt.iterations)
        assert result['solver_iterations']==a.solver_iterations
        result['root_assistance_strength']=root_strength
        result['root_controller_parameters']={'weight_support':a.root_weight_support,'vertical_limit':a.root_vertical_limit}
        for j,target in enumerate([0.,.4,.8,1.2]):
            mask=torch.arange(self.n,device=self.device)%4==j
            if mask.any():result['commands'][str(target)]={'survival':float(alive[mask].float().mean()),'mean_seconds':float(lifetime[mask].mean()),'velocity_mae':float((errors[mask]/(lifetime[mask]/self.dt).clamp(min=1)).mean())}
        result['first_episode_gait']={'qualified_landings_mean':float(landing_count.mean()),'longest_alternating_chain_mean':float(max_chain.mean()),'fraction_with_chain_at_least_4':float((max_chain>=4).float().mean()),'failure_fractions':{key:float(value.mean()) for key,value in failure_counts.items()},'note':'Force-contact touchdown criterion, before first termination; chain gap <=1.2s. Failure causes may overlap.'}
        assert float(force_error)<.001 and float(extra_force_max)<.001,(float(force_error),float(extra_force_max))
        if expected_force==0. and root_strength==0.:assert float(wrench_max)==0.,float(wrench_max)
        result['external_force_check']={'expected_upward_N':expected_force,'max_sum_error_N':float(force_error),'max_external_wrench_component':float(wrench_max),'max_other_component':float(extra_force_max)}
        if a.root_assistance>0:
            result['external_force_check']={'controller':'pelvis_pd','strength':root_strength,'max_requested_wrench_error':float(force_error),'max_external_wrench_component':float(wrench_max),'max_unrequested_body_wrench':float(extra_force_max)}
        if sequence:
            result.pop('commands')
            result['command_schedule']={'targets_mps':sequence,'segment_seconds':seconds/len(sequence)}
            result['sequence_metrics']={'survival':float(alive.float().mean()),'mean_seconds':float(lifetime.mean()),'velocity_mae':float((errors/(lifetime/self.dt).clamp(min=1)).mean()),'segment_survival':segment_survival,'segment_velocity_mae':[float(e/c) if c>0 else None for e,c in zip(segment_errors,segment_counts)]}
        folder=a.run_dir/'evaluations'/label;folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'first_episode_outcomes.npz',lifetime_seconds=lifetime.cpu().numpy(),survived=alive.cpu().numpy(),initial_command=initial_eval_command.cpu().numpy())
        np.savez_compressed(folder/'rollout.npz',qpos=poses,act=acts,reset=resets,velocity=speeds,command=commands,foot_clearance=foot_heights,phase=phases,external_wrench=wrenches,dt=.04)
        (folder/'model.xml').write_text(Path(os.environ['MM_MODEL_XML']).read_text())
        (folder/'metrics.json').write_text(json.dumps(result,indent=2))
        fraction=float(self.support_t[0])
        assistance=dict(body='head',fraction_body_weight=fraction,force_world_N=[0.,0.,fraction*float(np.sum(REFERENCE.body_mass))*9.81])
        if a.root_assistance>0:assistance=dict(body='pelvis',controller='pelvis_pd',strength=root_strength,fraction_body_weight=0.,weight_support=a.root_weight_support,vertical_limit=a.root_vertical_limit)
        (folder/'assistance.json').write_text(json.dumps(assistance))
        self.root_strength.copy_(old_root_strength)
        self.support_t.copy_(old_support);self.max_steps=self.training_max_steps;self.autoreset=True
        self.reset(torch.ones(self.n,dtype=torch.bool,device=self.device));actor.train(was_training)
        self.completed=old_completed
        return result,folder


def main():
    version='g1_curriculum_v5_root' if a.root_assistance>0 else ('g1_curriculum_v4_pose' if a.pose_weight>0 else ('g1_curriculum_v3_basis' if a.action_basis else ('g1_curriculum_v2_reference' if a.reference_reset_probability>0 else 'g1_curriculum_v1')))
    if a.cartesian_weight>0:version='g1_curriculum_v6_cartesian'
    if a.cartesian_weight>0 and a.cartesian_positive_score:version='g1_curriculum_v7_positive_cartesian'
    (a.run_dir/'reward_config.json').write_text(json.dumps(dict(version=version,rate_weights=WEIGHTS,touchdown_event_weight=TOUCHDOWN_WEIGHT,support=a.support,root_assistance_strength=a.root_assistance,minimum_head_gap_fraction=.6,reference_pose_weight=a.pose_weight,reference_cartesian_weight=a.cartesian_weight,cartesian_positive_score=a.cartesian_positive_score,trunk_height_weight=a.trunk_height_weight,phase_frequency=a.phase_frequency),indent=2))
    (a.run_dir/'config.json').write_text(json.dumps(vars(a),default=str,indent=2))
    cfg=NewtonCfg(solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=True,update_data_interval=0,integrator='euler',iterations=a.solver_iterations,ls_iterations=8,nconmax=256,njmax=1024),use_cuda_graph=True,num_substeps=1)
    cfg.class_type=NewtonVelocityManager
    with build_simulation_context(sim_cfg=SimulationCfg(dt=.002,device='cuda:0',physics=cfg)) as sim:
        env=VelocityEnv(sim,a.num_envs);obs=env.observation()
        actor_type=PhaseResidualActor if a.phase_harmonics else MLPModel
        actor_options={'harmonics':a.phase_harmonics} if a.phase_harmonics else {}
        actor=actor_type(obs,{'actor':['policy']},'actor',env.num_actions,**actor_options,hidden_dims=[512,256,256],obs_normalization=True,distribution_cfg={'class_name':'rsl_rl.modules.GaussianDistribution','init_std':.15 if a.action_basis else .5,'std_type':'log','std_range':[.05,1.]}).to('cuda:0')
        critic=MLPModel(obs,{'critic':['policy']},'critic',1,hidden_dims=[512,256,256],obs_normalization=True).to('cuda:0')
        if not a.legacy_support_normalization:
            for network in [actor,critic]:
                network.obs_normalizer=VelocityNormalization(obs['policy'].shape[-1]).to('cuda:0')
        horizon=32;storage=RolloutStorage('rl',env.n,horizon,obs,[env.num_actions],device='cuda:0')
        alg=PPO(actor,critic,storage,device='cuda:0',num_learning_epochs=4,num_mini_batches=4,learning_rate=3e-4,desired_kl=a.desired_kl,entropy_coef=a.entropy_coef,gamma=.99,lam=.95)
        start_it=0;schedule=None
        saved=torch.load(a.resume,weights_only=False) if a.resume else None
        if a.action_basis:
            if a.legacy_support_normalization:raise ValueError('Basis experiment requires fixed support/heading normalization')
            if saved is not None:
                if 'action_basis' not in saved:raise ValueError('Basis resume requires a basis-policy checkpoint')
                if not torch.equal(saved['action_basis'].cpu(),env.action_transform.basis.cpu()):raise ValueError('Resume basis differs from saved basis')
                baseline_state=saved['baseline_actor_state_dict']
            else:
                baseline_saved=torch.load(a.baseline_checkpoint,weights_only=False)
                if not baseline_saved.get('reward_version','').startswith('g1_') or 'action_basis' in baseline_saved:
                    raise ValueError('Frozen baseline must be a full-action G1 policy')
                baseline_state=baseline_saved['actor_state_dict']
                critic.load_state_dict(baseline_saved['critic_state_dict'])
            baseline=MLPModel(obs,{'actor':['policy']},'actor',env.m.nu,hidden_dims=[512,256,256],obs_normalization=True,distribution_cfg={'class_name':'rsl_rl.modules.GaussianDistribution','init_std':.15,'std_type':'log','std_range':[.05,1.]}).to('cuda:0')
            baseline.obs_normalizer=VelocityNormalization(obs['policy'].shape[-1]).to('cuda:0')
            baseline.load_state_dict(baseline_state);baseline.eval();baseline.requires_grad_(False)
            env.baseline_actor=baseline
            if saved is None:
                actor.obs_normalizer.load_state_dict(baseline.obs_normalizer.state_dict())
                # Start with zero mean correction and modest correlated noise.
                head=[layer for layer in actor.mlp.modules() if isinstance(layer,torch.nn.Linear)][-1]
                torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias)
        if a.resume:
            if 'action_basis' in saved and not a.action_basis:raise ValueError('Basis checkpoint requires --action-basis')
            if saved.get('reward_version') not in ('g1_muscle_v1','g1_muscle_v2_torso','g1_muscle_v3_movement','g1_muscle_v4_phase_clearance','g1_curriculum_v1','g1_curriculum_v2_reference','g1_curriculum_v3_basis','g1_curriculum_v4_pose','g1_curriculum_v5_root','g1_curriculum_v6_cartesian','g1_curriculum_v7_positive_cartesian'):
                raise ValueError('Resume requires a G1-lineage checkpoint; old assisted experiment is not used')
            saved_harmonics=saved.get('phase_harmonics',0)
            if a.warm_start_phase:
                if saved_harmonics or 'action_basis' in saved:raise ValueError('Phase warm start requires a base full-action checkpoint')
                actor.load_base_state_dict(saved['actor_state_dict']);critic.load_state_dict(saved['critic_state_dict'])
                (a.run_dir/'phase_initialization.json').write_text(json.dumps({'source':str(a.resume),'optimizer_reset':True,'initial_periodic_weights_zero':True,'note':'Actor/critic and normalization restored; optimizer starts fresh for the expanded parameter set.'},indent=2))
            else:
                if saved_harmonics!=a.phase_harmonics:raise ValueError('Checkpoint phase harmonics mismatch; adding a branch requires --warm-start-phase')
                alg.load(saved,None,True)
            start_it=saved['iteration']+1
            env.speed_max=saved.get('speed_max',.8)
            env.support_level=a.support;env.support_t.fill_(a.support)
            schedule=saved.get('exploration_schedule')
        if a.reset_exploration_std is not None:
            schedule,reset_metadata=reset_exploration(actor.distribution,alg.optimizer,a.reset_exploration_std,start_it,a.exploration_std_final,a.exploration_anneal_updates)
            reset_metadata['source']=str(a.resume)
            (a.run_dir/'exploration_reset.json').write_text(json.dumps(reset_metadata,indent=2))
        if a.exploration_std_final is not None and schedule is None:
            initial=float(actor.distribution.log_std_param.exp().clamp(.05,1.).max())
            if a.exploration_std_final>initial:raise ValueError('Exploration annealing must not increase initial std')
            schedule=dict(start_iteration=start_it,initial=initial,final=a.exploration_std_final,updates=a.exploration_anneal_updates)
        elif a.exploration_std_final is not None and schedule['final']!=a.exploration_std_final:
            raise ValueError('Resume preserves the saved exploration schedule; do not silently replace it')
        if a.eval_only:
            if not a.resume:raise ValueError('--eval-only requires --resume')
            result,folder=env.evaluate(actor,'evaluation',a.eval_seconds,support=a.eval_support,root_strength=a.eval_root_assistance)
            print('EVALUATION',json.dumps(result),flush=True)
            if not a.no_render:
                subprocess.run([os.environ.get('MM_RENDER_PYTHON',sys.executable),str(Path(__file__).with_name('render_velocity.py')),str(folder)],env={**os.environ,'MUJOCO_GL':'egl'},check=True)
            return
        writer=SummaryWriter(str(a.run_dir/'tensorboard'));start=time.monotonic();last_eval=start;last_save=start;last_video=start;eval_count=0
        def save(it,name):
            state=alg.save();state.update(reward_version=version,phase_harmonics=a.phase_harmonics,iteration=it,support=env.support_level,speed_max=env.speed_max,config=vars(a),exploration_schedule=schedule,training_episodes_this_run=env.training_episodes)
            if env.action_transform is not None:
                state.update(action_basis=env.action_transform.basis.detach().cpu(),baseline_actor_state_dict=env.baseline_actor.state_dict())
            tmp=a.run_dir/(name+'.tmp');torch.save(state,tmp);tmp.replace(a.run_dir/name)
        render_jobs=[]
        def render(folder):
            if a.no_render:return
            with (folder/'render.log').open('w') as log:
                render_jobs.append(subprocess.Popen([os.environ.get('MM_RENDER_PYTHON',sys.executable),str(Path(__file__).with_name('render_velocity.py')),str(folder)],env={**os.environ,'MUJOCO_GL':'egl'},stdout=log,stderr=subprocess.STDOUT))
        print('READY',json.dumps({'n':env.n,'obs':obs['policy'].shape[1],'actions':env.num_actions}),flush=True)
        if a.smoke:
            if env.action_transform is not None:
                with torch.no_grad():
                    baseline_logits=env.baseline_actor(env.observation(),stochastic_output=False)
                    zero=torch.zeros(env.n,env.num_actions,device=env.device)
                    assert torch.equal(env.action_transform(zero,baseline_logits),baseline_logits)
                    untouched=env.action_transform.basis.abs().sum(0)==0
                    probe=torch.arange(env.num_actions,device=env.device,dtype=torch.float32).expand(env.n,-1)/max(env.num_actions,1)
                    changed=env.action_transform(probe,baseline_logits)
                    assert torch.equal(changed[:,untouched],baseline_logits[:,untouched])
                    assert all(not p.requires_grad for p in env.baseline_actor.parameters())
                print('SMOKE_BASIS_PASS',json.dumps(dict(learned_actions=env.num_actions,untouched_muscles=int(untouched.sum()),native_actuator_order_verified=True)),flush=True)
            if env.reference_bank is not None and a.reference_reset_probability==1.:
                distance=(env.q[:,None,:]-env.reference_bank['qpos'][None,:,:]).abs().amax(-1)
                closest=distance.argmin(1)
                assert distance.min(1).values.max()<1e-6
                assert torch.allclose(env.v,env.reference_bank['qvel'][closest])
                assert torch.allclose(env.phase,env.reference_bank['phase'][closest])
                assert torch.allclose(env.command,env.reference_bank['command'][closest])
                print('SMOKE_REFERENCE_RESET_PASS',flush=True)
            before=env.q.clone()
            for _ in range(100):env.step(torch.zeros(env.n,env.num_actions,device='cuda:0'))
            assert torch.isfinite(env.q).all();assert (env.q-before).abs().max()>1e-5
            reset_mask=torch.arange(env.n,device='cuda:0')%2==0;unselected=env.q[~reset_mask].clone()
            env.reset(reset_mask,randomize=False);assert torch.allclose(env.q[reset_mask],env.initial.expand(int(reset_mask.sum()),-1));assert torch.equal(env.q[~reset_mask],unselected)
            print('SMOKE_RESET_PASS',flush=True)
            cpu=mujoco.MjData(env.m);cpu.qpos[:]=env.q[0].cpu().numpy();mujoco.mj_forward(env.m,cpu)
            floor=int(np.flatnonzero(env.m.geom_type==int(mujoco.mjtGeom.mjGEOM_PLANE))[0])
            expected=[min(mujoco.mj_geomDistance(env.m,cpu,floor,g,10.,None) for g in ids) for ids,_,_ in env.feet.feet]
            actual=env.feet()[0].cpu().numpy()
            assert np.max(np.abs(actual-expected))<5e-5,(actual,expected)
            print('SMOKE_FOOT_CLEARANCE_PASS',actual.tolist(),flush=True)
            if env.cartesian_reward is not None:
                expected_sites=[]
                for world in range(env.n):
                    cpu.qpos[:]=env.q[world].cpu().numpy();mujoco.mj_forward(env.m,cpu)
                    expected_sites.append(cpu.site_xpos.copy())
                expected_sites=torch.tensor(np.array(expected_sites),dtype=torch.float32,device=env.device)
                error=(env.site_positions[:,env.cartesian_site_ids]-expected_sites[:,env.cartesian_site_ids]).abs().max()
                assert float(error)<5e-5,float(error)
                score=env.cartesian_reward(env.site_positions[:,env.cartesian_site_ids],env.site_positions[:,env.cartesian_pelvis_site],env.heading_error(),env.phase,env.command)
                expected_score=env.cartesian_reward(expected_sites[:,env.cartesian_site_ids],expected_sites[:,env.cartesian_pelvis_site],env.heading_error(),env.phase,env.command)
                assert torch.isfinite(score).all() and torch.allclose(score,expected_score,atol=1e-5)
                if a.cartesian_positive_score:
                    assert ((score>=0)&(score<=1)).all()
                print('SMOKE_CARTESIAN_PASS',json.dumps(dict(site_max_error_m=float(error),score_max_error=float((score-expected_score).abs().max()))),flush=True)
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
            if a.ppo_diagnostics:
                loss,diagnostic=update_with_diagnostics(alg)
                diagnostic.update(iteration=it,num_envs=env.n)
                with (a.run_dir/'ppo_diagnostics.jsonl').open('a') as f:f.write(json.dumps(diagnostic)+'\n')
            else:loss=alg.update()
            # Save/evaluate the same capped distribution used for the next rollout.
            cap_exploration(actor.distribution,schedule,it)
            now=time.monotonic()
            if it%10==0:
                recent=env.completed[-1000:]
                record={'iteration':it,'wall_seconds':now-start,'env_steps':(it-start_it+1)*horizon*env.n,'support':env.support_level,'root_assistance_strength':float(env.root_strength.mean()),'learning_rate':alg.learning_rate,'loss':loss,**env.last_metrics,
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
            if requested_eval or now-last_eval>=(a.first_eval_seconds if eval_count==0 else a.eval_interval) or ((a.smoke or a.eval_at_end) and it==start_it+a.iterations-1):
                save(it,f'model_{it}.pt')
                result,folder=env.evaluate(actor,f'iter_{it:06d}',a.eval_seconds,support=a.eval_support)
                print('EVALUATION',json.dumps(result),flush=True)
                with (a.run_dir/'evaluations.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
                render_this=eval_count==0 or requested_eval or now-last_video>=a.video_interval
                if render_this:render(folder)
                if render_this:last_video=time.monotonic()
                if requested_eval:(a.run_dir/'EVALUATE').unlink(missing_ok=True)
                if a.support>0 and a.eval_support==0.:
                    assisted,assisted_folder=env.evaluate(actor,f'iter_{it:06d}_assisted',a.eval_seconds,support=a.support)
                    print('EVALUATION',json.dumps(assisted),flush=True)
                    with (a.run_dir/'evaluations.jsonl').open('a') as f:f.write(json.dumps(assisted)+'\n')
                    if render_this:render(assisted_folder)
                if a.root_assistance>0:
                    assisted,assisted_folder=env.evaluate(actor,f'iter_{it:06d}_root_assisted',a.eval_seconds,support=0.,root_strength=a.root_assistance)
                    print('EVALUATION',json.dumps(assisted),flush=True)
                    with (a.run_dir/'evaluations.jsonl').open('a') as f:f.write(json.dumps(assisted)+'\n')
                    if render_this:render(assisted_folder)
                save(it,'latest.pt');last_save=time.monotonic()
                eval_count+=1;last_eval=time.monotonic();obs=env.observation()
            if (a.run_dir/'STOP').exists():
                save(it,'stopped.pt');print('STOP_REQUESTED',flush=True);break
        save(it,'latest.pt');writer.close()
        for job in render_jobs:
            if job.wait()!=0:raise RuntimeError('Video rendering failed; inspect evaluation render.log')

if __name__=='__main__':main()
