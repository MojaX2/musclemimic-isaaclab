"""Validate the optional standing weight-support control."""

from types import SimpleNamespace
import unittest

import torch

from evaluate_infant_newton_crawl import InfantNewtonCrawlEnv
from infant_crawl_env import InfantCrawlEnv, planar_tether_force


class InfantStandUnloadTests(unittest.TestCase):
    def test_planar_tether_restores_position_and_damps_velocity(self):
        force = planar_tether_force(torch.tensor([[0.1, -0.2]]),
                                    torch.tensor([[0.5, -0.5]]),
                                    torch.zeros((1, 2)), 100, 10)
        torch.testing.assert_close(force, torch.tensor([[-15., 25.]]))

    def test_both_backends_reject_invalid_unload(self):
        for environment_type in (InfantCrawlEnv, InfantNewtonCrawlEnv):
            environment = environment_type.__new__(environment_type)
            environment.worlds = 1
            environment.source = SimpleNamespace(nu=1)
            for fraction in (-0.1, 1.1):
                with self.subTest(backend=environment_type.__name__, fraction=fraction):
                    with self.assertRaisesRegex(ValueError, "Root vertical unload"):
                        environment.step(torch.zeros((1, 2)), 1,
                                         root_vertical_unload=fraction)
            with self.assertRaisesRegex(ValueError, "Planar tether gains"):
                environment.step(torch.zeros((1, 2)), 1,
                                 root_planar_stiffness=-1)
            with self.assertRaisesRegex(ValueError, "Planar tether gains"):
                environment.step(torch.zeros((1, 2)), 1,
                                 root_planar_damping=float("nan"))


if __name__ == "__main__":
    unittest.main()
