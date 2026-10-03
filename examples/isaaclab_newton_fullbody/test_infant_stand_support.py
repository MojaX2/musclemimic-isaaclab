"""Check that standing reward requires elevation and floor foot support."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from developmental_skin import REGIONS
from train_infant_crawl_ppo import SupportPolicy
from train_infant_stand_support import stand_score
from train_infant_stand_ppo import configure_stand_exploration, step_stand_control


class InfantStandSupportTests(unittest.TestCase):
    def test_zero_actor_head_starts_from_exact_reflex_and_low_noise(self):
        policy = SupportPolicy(5, 22)
        configure_stand_exploration(policy, -2.3, zero_actor_head=True)
        torch.testing.assert_close(policy.actor(torch.randn((3, 5))), torch.zeros((3, 22)))
        torch.testing.assert_close(policy.log_std, torch.full((22,), -2.3))
        with self.assertRaises(ValueError):
            configure_stand_exploration(policy, -4)

    def test_pressure_reflex_updates_every_physics_step(self):
        environment = SimpleNamespace(
            worlds=1, feedback_drives=Mock(return_value=torch.zeros((1, 22))),
            action_from_drives=Mock(return_value=torch.zeros((1, 180))),
            step=Mock(side_effect=lambda action, count: {"step": count}))
        sensed = []
        with patch("sweep_infant_cop_reflex.cop_reflex_drives",
                   side_effect=lambda env, measure, drives, gains:
                   sensed.append(measure) or drives):
            result = step_stand_control(
                environment, {"initial": True}, torch.zeros(22), None, 0,
                3, torch.zeros((1, 6)), 0.4, 0.3)
        self.assertEqual(environment.step.call_count, 3)
        self.assertEqual([call.args[1] for call in environment.step.call_args_list],
                         [1, 1, 1])
        self.assertEqual(sensed, [{"initial": True}, {"step": 1}, {"step": 1}])
        self.assertEqual(result, {"step": 1})

    def test_collapsed_or_hovering_pose_scores_lower(self):
        initial = torch.tensor([[0, 0, 0.33, 1, 0, 0, 0]], dtype=torch.float32)
        environment = SimpleNamespace(root_address=0, position=initial.clone(), initial=initial)
        touch = torch.zeros((1, len(REGIONS)))
        touch[0, REGIONS.index("right_foot")] = 10
        touch[0, REGIONS.index("left_foot")] = 10
        standing = {"chest_height": torch.tensor([0.58]),
                    "head_height": torch.tensor([0.63]),
                    "ground_touch": touch}
        collapsed = {**standing, "chest_height": torch.tensor([0.15]),
                     "head_height": torch.tensor([0.2])}
        hovering = {**standing, "ground_touch": torch.zeros_like(touch)}
        dragging_touch = touch.clone()
        dragging_touch[0, REGIONS.index("pelvis")] = 10
        dragging = {**standing, "ground_touch": dragging_touch}
        self.assertGreater(float(stand_score(environment, standing)),
                           float(stand_score(environment, collapsed)))
        self.assertGreater(float(stand_score(environment, standing)),
                           float(stand_score(environment, hovering)))
        self.assertGreater(float(stand_score(environment, standing)),
                           float(stand_score(environment, dragging)))
        self.assertAlmostEqual(float(stand_score(environment, collapsed)), 0.0)


if __name__ == "__main__":
    unittest.main()
