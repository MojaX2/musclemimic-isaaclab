"""Verify candidate alignment and late standing reward in hip-reflex evolution."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from evolve_infant_stand_hip_reflex import score_population


class InfantStandHipEvolutionTests(unittest.TestCase):
    def test_candidates_share_poses_and_report_late_fraction(self):
        environment = SimpleNamespace(worlds=4)
        gains = torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]])
        poses = torch.tensor([[10., 11.], [20., 21.]])

        def fake_evaluate(env, bias, expanded, positions, steps, return_late):
            self.assertIs(env, environment)
            self.assertTrue(return_late)
            self.assertEqual(steps, 40)
            torch.testing.assert_close(expanded, gains.repeat_interleave(2, 0))
            torch.testing.assert_close(positions, poses.repeat(2, 1))
            return (torch.tensor([0., 1., 2., 3.]), torch.zeros(4), None,
                    torch.tensor([0.1, 0.3, 0.5, 0.7]))

        with (patch("evolve_infant_stand_hip_reflex.aligned_positions",
                    return_value=poses),
              patch("evolve_infant_stand_hip_reflex.evaluate", side_effect=fake_evaluate)):
            fraction, late, final = score_population(environment, None, gains, 17, 2, 40)
        torch.testing.assert_close(fraction, torch.tensor([0.5, 2.5]))
        torch.testing.assert_close(late, torch.tensor([0.2, 0.6]))
        torch.testing.assert_close(final, torch.zeros(2))


if __name__ == "__main__":
    unittest.main()
