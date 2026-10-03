"""Check matching conditions in the infant standing bias audit."""

import unittest

import torch

from evaluate_infant_stand_equilibrium_bias import interpolate_bias


class StandEquilibriumBiasTests(unittest.TestCase):
    def test_interpolation_endpoints(self):
        previous = torch.tensor([[-1., 0.]])
        equilibrium = torch.tensor([[1., 0.5]])
        torch.testing.assert_close(interpolate_bias(previous, equilibrium, 0), previous)
        torch.testing.assert_close(interpolate_bias(previous, equilibrium, 0.5),
                                   torch.tensor([[0., 0.25]]))
        torch.testing.assert_close(interpolate_bias(previous, equilibrium, 1), equilibrium)
        with self.assertRaises(ValueError):
            interpolate_bias(previous, equilibrium, -0.1)


if __name__ == "__main__":
    unittest.main()
