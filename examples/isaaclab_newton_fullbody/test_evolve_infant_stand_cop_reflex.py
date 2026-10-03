"""Verify candidate grouping in pressure-reflex evolution."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from evolve_infant_stand_cop_reflex import score_population


class InfantStandCopEvolutionTests(unittest.TestCase):
    def test_population_uses_identical_poses_for_each_gain_candidate(self):
        environment = SimpleNamespace(worlds=6)
        gains = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
        poses = torch.tensor([[10., 11.], [20., 21.]])

        def fake_evaluate(env, bias, expanded_gains, positions, control_steps):
            self.assertIs(env, environment)
            torch.testing.assert_close(positions, poses.repeat(3, 1))
            torch.testing.assert_close(expanded_gains, gains.repeat_interleave(2, dim=0))
            self.assertEqual(control_steps, 40)
            return (torch.tensor([0., 1., 2., 3., 4., 5.]),
                    torch.tensor([0., 0., 1., 0., 0., 1.]), None)

        with (patch("evolve_infant_stand_cop_reflex.aligned_positions", return_value=poses),
              patch("evolve_infant_stand_cop_reflex.evaluate", side_effect=fake_evaluate)):
            fraction, final = score_population(environment, None, gains, 17, 2, 40)
        torch.testing.assert_close(fraction, torch.tensor([0.5, 2.5, 4.5]))
        torch.testing.assert_close(final, torch.tensor([0., 0.5, 0.5]))


if __name__ == "__main__":
    unittest.main()
