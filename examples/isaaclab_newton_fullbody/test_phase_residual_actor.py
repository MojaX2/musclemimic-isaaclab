import unittest
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from phase_residual_actor import PhaseResidualActor
from velocity_normalization import VelocityNormalization


class PhaseActorTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11);torch.set_num_threads(2)
        raw=torch.zeros(8,888);phase=torch.arange(8)*torch.pi/4
        raw[:,-6]=1;raw[:,-5]=phase.sin();raw[:,-4]=phase.cos()
        self.obs=TensorDict({'policy':raw},batch_size=[8])
        self.kw=dict(hidden_dims=[16,16],obs_normalization=True,distribution_cfg={'class_name':'rsl_rl.modules.GaussianDistribution','init_std':.15,'std_type':'log','std_range':[.05,1.]})
        self.base=MLPModel(self.obs,{'actor':['policy']},'actor',354,**self.kw).eval()
        self.base.obs_normalizer=VelocityNormalization(888)
        self.base.update_normalization(self.obs)
        self.actor=PhaseResidualActor(self.obs,{'actor':['policy']},'actor',354,**self.kw).eval()
        self.actor.load_base_state_dict(self.base.state_dict())

    def test_zero_branch_preserves_mean_and_sample(self):
        torch.testing.assert_close(self.base(self.obs),self.actor(self.obs),rtol=0,atol=0)
        torch.manual_seed(9);a=self.base(self.obs,stochastic_output=True)
        torch.manual_seed(9);b=self.actor(self.obs,stochastic_output=True)
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_distribution_and_gradient_include_phase_branch(self):
        with torch.no_grad():self.actor.phase_head.weight[0,0]=.2
        mean=self.actor(self.obs);expected=self.base(self.obs).detach();expected[:,0]+=.2*self.obs['policy'][:,-5]
        torch.testing.assert_close(mean,expected)
        self.actor(self.obs,stochastic_output=True)
        actions=mean.detach()+.1
        actual=self.actor.get_output_log_prob(actions)
        reference=torch.distributions.Normal(mean,self.actor.output_std).log_prob(actions).sum(-1)
        torch.testing.assert_close(actual,reference)
        (-actual[:2].mean()).backward()
        self.assertGreater(float(self.actor.phase_head.weight.grad.norm()),0)

    def test_standing_gate_and_checkpoint_roundtrip(self):
        with torch.no_grad():self.actor.phase_head.weight.fill_(.1)
        stopped=self.obs.clone();stopped['policy'][:,-6]=0
        torch.testing.assert_close(self.actor(stopped),self.base(stopped),rtol=0,atol=0)
        other=PhaseResidualActor(self.obs,{'actor':['policy']},'actor',354,**self.kw).eval()
        other.load_state_dict(self.actor.state_dict())
        torch.testing.assert_close(other(self.obs),self.actor(self.obs),rtol=0,atol=0)
        with self.assertRaises(NotImplementedError):self.actor.as_jit()

if __name__=='__main__':unittest.main()
