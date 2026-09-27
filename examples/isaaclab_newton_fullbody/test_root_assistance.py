import math
import unittest
import torch
from root_assistance import pelvis_wrench


class RootAssistanceTest(unittest.TestCase):
    def setUp(self):
        self.q=torch.tensor([[0.,0.,1.,1.,0.,0.,0.]])
        self.v=torch.zeros(1,6);self.target=torch.tensor([1.,0.,0.,0.])

    def force(self,strength=1.,command=-.4):
        return pelvis_wrench(self.q,self.v,self.target,1.,torch.tensor([[command,0.]]),80.,torch.tensor([strength]))

    def test_zero_and_nominal_force(self):
        self.assertTrue(torch.equal(self.force(0),torch.zeros(1,6)))
        f=self.force()[0];self.assertAlmostEqual(float(f[0]),-64.,places=4)
        self.assertAlmostEqual(float(f[2]),.3*80*9.81,places=3)
        self.assertTrue(torch.equal(f[3:],torch.zeros(3)))

    def test_restoring_torque_and_quaternion_sign(self):
        self.q[0,3:7]=torch.tensor([math.cos(.1),math.sin(.1),0.,0.])
        f=self.force();self.assertLess(float(f[0,3]),0.)
        self.q[:,3:7]*=-1
        self.assertTrue(torch.allclose(f,self.force()))

    def test_stronger_support_and_zero(self):
        args=(self.q,self.v,self.target,1.,torch.zeros(1,2),80.)
        f=pelvis_wrench(*args,torch.ones(1),weight_support=.8,vertical_limit=1.2)
        self.assertAlmostEqual(float(f[0,2]),.8*80*9.81,places=3)
        self.assertTrue(torch.equal(pelvis_wrench(*args,torch.zeros(1),weight_support=.8,vertical_limit=1.2),torch.zeros(1,6)))
        self.q[:,2]=0.
        f=pelvis_wrench(*args,torch.ones(1),weight_support=.8,vertical_limit=1.2)
        self.assertAlmostEqual(float(f[0,2]),1.2*80*9.81,places=3)

    def test_caps_and_scaling(self):
        self.v.fill_(100.)
        f=self.force();self.assertLessEqual(float(f[0,:2].norm()),.3*80*9.81+1e-3)
        self.assertLessEqual(abs(float(f[0,2])),.6*80*9.81+1e-3)
        self.assertLessEqual(float(f[0,3:].norm()),120.001)
        self.assertTrue(torch.allclose(self.force(.25),f*.25))


if __name__=='__main__':unittest.main()
