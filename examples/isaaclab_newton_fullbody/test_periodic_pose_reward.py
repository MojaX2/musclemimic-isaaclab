import math
import unittest
import torch
from periodic_pose_reward import PeriodicPoseReward


class PeriodicPoseTest(unittest.TestCase):
    def setUp(self):
        self.reward=PeriodicPoseReward([[.1,-.1],[.2,-.2],[.1,.1]],[0.,0.])

    def test_cycle_seam_and_gradient(self):
        phase=torch.tensor([0.,2*math.pi],requires_grad=True)
        values=self.reward.target(phase)
        self.assertTrue(torch.allclose(values[0],values[1],atol=1e-6))
        values.sum().backward()
        self.assertTrue(torch.allclose(phase.grad[0],phase.grad[1],atol=1e-6))

    def test_standing_zero_target_positive_stationary_disabled(self):
        phase=torch.tensor([0.,1.,2.]);command=torch.ones(3)*.4
        self.assertTrue(torch.equal(self.reward(torch.zeros(3,2),phase,command),torch.zeros(3)))
        self.assertTrue((self.reward(self.reward.target(phase),phase,command)>0).all())
        self.assertTrue(torch.equal(self.reward(self.reward.target(phase),phase,torch.zeros(3)),torch.zeros(3)))

    def test_local_improvement_before_liftoff(self):
        phase=torch.tensor([.5]);q=torch.zeros(1,2,requires_grad=True)
        self.reward(q,phase,torch.tensor([.4])).sum().backward()
        self.assertGreater(float((q.grad*self.reward.target(phase)).sum()),0.)


if __name__=='__main__':unittest.main()
