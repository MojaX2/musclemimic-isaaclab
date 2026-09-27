import copy,unittest
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.algorithms import PPO
from ppo_update_diagnostics import update_with_diagnostics

class DiagnosticsTest(unittest.TestCase):
    def test_same_update_with_and_without_observer(self):
        torch.manual_seed(91)
        obs=TensorDict({'policy':torch.randn(4,5)},batch_size=[4])
        actor=MLPModel(obs,{'actor':['policy']},'actor',3,hidden_dims=[8],obs_normalization=False,distribution_cfg={'class_name':'rsl_rl.modules.GaussianDistribution','init_std':.5,'std_type':'log'})
        critic=MLPModel(obs,{'critic':['policy']},'critic',1,hidden_dims=[8],obs_normalization=False)
        storage=RolloutStorage('rl',4,4,obs,[3],device='cpu')
        alg=PPO(actor,critic,storage,device='cpu',num_learning_epochs=2,num_mini_batches=2,desired_kl=.02)
        with torch.inference_mode():
            for _ in range(4):
                alg.act(obs);obs=TensorDict({'policy':torch.randn(4,5)},batch_size=[4]);alg.process_env_step(obs,torch.randn(4),torch.zeros(4,dtype=torch.bool),{})
            alg.compute_returns(obs)
        observed=copy.deepcopy(alg)
        torch.manual_seed(92);expected=alg.update()
        torch.manual_seed(92);actual,diagnostics=update_with_diagnostics(observed)
        self.assertEqual(expected,actual)
        for key,value in alg.actor.state_dict().items():self.assertTrue(torch.equal(value,observed.actor.state_dict()[key]),key)
        self.assertEqual(alg.learning_rate,observed.learning_rate)
        self.assertNotIn('get_kl_divergence',observed.actor.__dict__)
        self.assertEqual(diagnostics['minibatches'],4)
        self.assertGreater(diagnostics['kl_max'],0.)

if __name__=='__main__':unittest.main()
