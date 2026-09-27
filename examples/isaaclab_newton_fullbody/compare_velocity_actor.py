"""Paired open-loop actor probes on upright reset observations, not gait evidence."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from velocity_normalization import VelocityNormalization
from phase_residual_actor import PhaseResidualActor
p=argparse.ArgumentParser();p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path);p.add_argument('--support',type=float,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(4);root=Path(__file__).resolve().parents[2];q=np.load(root/'outputs/isaac_velocity/assets/initial.npz')['qpos'];quat=q[3:7]
gravity=np.array([2*(quat[1]*quat[3]-quat[0]*quat[2]),2*(quat[2]*quat[3]+quat[0]*quat[1]),1-2*(quat[1]**2+quat[2]**2)])
phase=np.arange(32)*2*np.pi/32;values=np.zeros((32,888),np.float32);values[:,0]=q[2];values[:,1:4]=gravity;values[:,-6:]=np.stack([np.ones(32),np.sin(phase),np.cos(phase),np.ones(32),np.zeros(32),np.full(32,a.support)],axis=1)
obs=TensorDict({'policy':torch.from_numpy(values)},batch_size=[32]);outputs=[];records=[]
for path in [a.baseline,a.candidate]:
 s=torch.load(path,map_location='cpu',weights_only=False)
 if 'action_basis' in s:raise ValueError('Probe supports full354-action checkpoints only')
 harmonics=s.get('phase_harmonics',0);actor_type=PhaseResidualActor if harmonics else MLPModel;options={'harmonics':harmonics} if harmonics else {}
 actor=actor_type(obs,{'actor':['policy']},'actor',354,**options,hidden_dims=[512,256,256],obs_normalization=True,distribution_cfg={'class_name':'rsl_rl.modules.GaussianDistribution','init_std':.5,'std_type':'log','std_range':[.05,1.]})
 actor.obs_normalizer=VelocityNormalization(888);actor.load_state_dict(s['actor_state_dict']);actor.eval()
 with torch.inference_mode():
  logits=actor(obs,stochastic_output=False);excitation=torch.sigmoid(1.5*logits-2.2).numpy()
 outputs.append(excitation);records.append(dict(checkpoint=str(path.resolve()),iteration=s['iteration'],mean_excitation=float(excitation.mean()),mean_phase_std=float(excitation.std(0).mean()),maximum_phase_std=float(excitation.std(0).max())))
delta=outputs[1]-outputs[0];result=dict(probes=records,mean_absolute_excitation_change=float(np.abs(delta).mean()),maximum_excitation_change=float(np.abs(delta).max()),support=a.support,command_mps=1.,observations='Upright reset,zero velocity,zero activation/previous excitation,32 phases. Relative scalar joint positions allzero, so solver joint reordering does not affect this probe.',caveat='Open-loop conditional network outputs only. Changes do not prove useful learning, periodic gait, stability, or velocity tracking.')
a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
