import unittest
import torch
from muscle_action_basis import MuscleActionBasis


class ActionBasisTest(unittest.TestCase):
    def test_tonic_baseline_and_covariance(self):
        # Two disjoint coordinated pairs: directions are orthogonal but output
        # muscles are correlated, not four independent Gaussian actions.
        b=torch.tensor([[1.,1.,0.,0.],[0.,0.,1.,-1.]])/2**.5
        transform=MuscleActionBasis(b)
        baseline=torch.tensor([.2,.3,.4,.5])
        self.assertTrue(torch.allclose(transform(torch.zeros(2),baseline),baseline))
        self.assertTrue(torch.allclose(transform(torch.tensor([1.,2.])),torch.tensor([1.,1.,2.,-2.])))
        std=transform.muscle_std(torch.tensor([.2,.4]))
        self.assertTrue(torch.allclose(std,torch.tensor([.2,.2,.4,.4])))
        self.assertAlmostEqual(float(transform.muscle_std(torch.ones(2)).square().sum()),4.,places=5)

    def test_basis_persists_and_gradients_reach_coefficients(self):
        transform=MuscleActionBasis(torch.eye(3)[:2]);x=torch.tensor([.3,.7],requires_grad=True)
        transform(x).square().sum().backward()
        self.assertTrue(torch.allclose(x.grad,3*x.detach()))
        self.assertIn('basis',transform.state_dict())
        self.assertEqual(len(list(transform.parameters())),0)

    def test_rejects_degenerate_basis(self):
        with self.assertRaises(ValueError):MuscleActionBasis(torch.ones(2,4))


if __name__=='__main__':unittest.main()
