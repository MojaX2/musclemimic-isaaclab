"""Conditional excitation distribution probe, not a simulation or gait test."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from phase_residual_actor import PhaseResidualActor
from velocity_normalization import VelocityNormalization
p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(4);torch.manual_seed(42)
root=Path(__file__).resolve().parents[2];q=np.load(root/'outputs/isaac_velocity/assets/initial.npz')['qpos'];w,x,y,z=q[3:7]
phase=np.arange(32)*2*np.pi/32;values=np.zeros((32,888),np.float32);values[:,0]=q[2];values[:,1:4]=[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)];values[:,-6:]=np.stack([np.ones(32),np.sin(phase),np.cos(phase),np.ones(32),np.zeros(32),np.full(32,.3)],axis=1)
obs=TensorDict({'policy':torch.from_numpy(values)},batch_size=[32]);saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);assert 'action_basis' not in saved
harmonics=saved.get('phase_harmonics',0);cls=PhaseResidualActor if harmonics else MLPModel;options={'harmonics':harmonics} if harmonics else {}
actor=cls(obs,{'actor':['policy']},'actor',354,**options,hidden_dims=[512,256,256],obs_normalization=True,distribution_cfg={'class_name':'rsl_rl.modules.GaussianDistribution','init_std':.5,'std_type':'log','std_range':[.05,1.]});actor.obs_normalizer=VelocityNormalization(888);actor.load_state_dict(saved['actor_state_dict']);actor.eval()
with torch.inference_mode():
 mean=actor(obs,stochastic_output=False);current_std=actor.distribution.log_std_param.exp().clamp(.05,1.);noise=torch.randn(1024,32,354);deterministic=torch.sigmoid(1.5*mean-2.2);cases={}
 for name,std in [('checkpoint',current_std),('hypothetical_std_0.5',torch.full_like(current_std,.5)),('hypothetical_std_1.0',torch.ones_like(current_std))]:
  samples=torch.sigmoid(1.5*(mean+noise*std)-2.2);conditional_mean=samples.mean(0);conditional_std=samples.std(0)
  cases[name]=dict(latent_std_mean=float(std.mean()),mean_excitation=float(samples.mean()),mean_conditional_excitation_std=float(conditional_std.mean()),fraction_samples_above_0_5=float((samples>.5).float().mean()),mean_absolute_shift_from_deterministic=float((conditional_mean-deterministic).abs().mean()),mean_probability_of_delta_above_0_1=float(((samples-deterministic).abs()>.1).float().mean()))
result=dict(checkpoint=str(a.checkpoint.resolve()),iteration=saved['iteration'],cases=cases,observations='32 phases of upright reset, zero velocity/activation/previous control, command1m/s,headsupport0.3',limitations='Hypothetical noise widths retain the same Gaussian mean. Sigmoid nonlinearity changes mean excitation as well as spread. No activation dynamics, torque response, learning or gait are tested; not evidence that larger noise is better.')
a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
