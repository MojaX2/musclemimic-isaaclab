"""Check that Newton standing joint feedback matches the direct infant controller."""

from types import SimpleNamespace
import unittest

import torch

from evaluate_infant_newton_crawl import InfantNewtonCrawlEnv
from infant_crawl_env import InfantCrawlEnv


class NewtonStandFeedbackTests(unittest.TestCase):
    def test_feedback_matches_direct_controller(self):
        environment = SimpleNamespace(worlds=2,
                                      controlled_joint_names=("hip", "ankle"),
                                      initial=torch.tensor([[0., 0.3, -0.2],
                                                            [0., 0.3, -0.2]]),
                                      position=torch.tensor([[0., 0.4, -0.1],
                                                             [0., 0.2, -0.3]]),
                                      velocity=torch.tensor([[0., 0.2, -0.1],
                                                             [0., -0.2, 0.1]]),
                                      crawl_qpos_ids=torch.tensor([1, 2]),
                                      crawl_qvel_ids=torch.tensor([1, 2]))
        bias = torch.tensor([[0.1, -0.1], [0.1, -0.1]])
        actual = InfantNewtonCrawlEnv.feedback_drives(environment, bias, 1.5, 1.0)
        expected = InfantCrawlEnv.feedback_drives(environment, bias, 1.5, 1.0)
        torch.testing.assert_close(actual, expected)
        with self.assertRaises(ValueError):
            InfantNewtonCrawlEnv.feedback_drives(environment, bias[:, :1], 1.5, 1.0)


if __name__ == "__main__":
    unittest.main()
