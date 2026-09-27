import copy,unittest
from types import SimpleNamespace
import torch
from velocity_exploration import reset_exploration,cap_exploration

class ResetTest(unittest.TestCase):
    def test_reset_preserves_mean_optimizer_and_schedule_roundtrip(self):
        mean=torch.nn.Parameter(torch.ones(3));std=torch.nn.Parameter(torch.full((3,),-2.))
        optimizer=torch.optim.Adam([mean,std]);(mean.square().sum()+std.square().sum()).backward();optimizer.step()
        old_mean=mean.detach().clone();old_state=copy.deepcopy(optimizer.state[mean])
        schedule,meta=reset_exploration(SimpleNamespace(log_std_param=std),optimizer,.5,100,.15,600)
        self.assertTrue(torch.equal(mean,old_mean));self.assertNotIn(std,optimizer.state)
        for k,v in old_state.items():self.assertTrue(torch.equal(v,optimizer.state[mean][k]))
        self.assertTrue(meta['std_optimizer_state_cleared']);self.assertTrue(torch.allclose(std.exp(),torch.full((3,),.5)))
        restored=copy.deepcopy(schedule);cap_exploration(SimpleNamespace(log_std_param=std),restored,700)
        self.assertTrue(torch.allclose(std.exp(),torch.full((3,),.15)))

    def test_invalid_reset_does_not_mutate(self):
        std=torch.nn.Parameter(torch.full((3,),-2.));opt=torch.optim.Adam([std]);before=std.detach().clone()
        with self.assertRaises(ValueError):reset_exploration(SimpleNamespace(log_std_param=std),opt,.5,0,.8)
        self.assertTrue(torch.equal(before,std))

if __name__=='__main__':unittest.main()
