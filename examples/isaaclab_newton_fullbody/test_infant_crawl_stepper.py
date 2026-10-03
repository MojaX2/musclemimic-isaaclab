"""Check tactile stance and swing transitions for infant stepping."""

import unittest

import torch

from infant_crawl_env import PALM_JOINTS
from infant_crawl_oscillator import LIMB_IDS, SHOULDER_LIFT_IDS, WRIST_IDS
from infant_crawl_stepper import TactileStepper
from train_infant_crawl_stepper_ppo import (future_supported_forward_reward,
                                            supported_hand_reach_reward,
                                            sustained_support_reward)


class TactileStepperTests(unittest.TestCase):
    def test_future_forward_reward_requires_continued_support(self):
        speed = torch.ones((5, 1))
        support = torch.tensor([[True], [True], [False], [False], [True]])
        reward = future_supported_forward_reward(speed, support, 2, 1.0)
        torch.testing.assert_close(reward[:, 0], torch.tensor([0.5, 0., 0., 0., 0.]))
        maintained = future_supported_forward_reward(speed, torch.ones_like(support), 2, 1.0)
        torch.testing.assert_close(maintained[:, 0], torch.tensor([1., 1., 1., 0., 0.]))

    def test_sustained_support_reward_requires_elevation_and_both_limb_groups(self):
        measure = {"palm_shin_force": torch.tensor([[5., 0., 0., 5.],
                                                    [0., 0., 5., 5.],
                                                    [5., 0., 0., 5.]]),
                   "chest_height": torch.tensor([0.2, 0.2, 0.1]),
                   "head_height": torch.tensor([0.2, 0.2, 0.2])}
        early, supported = sustained_support_reward(measure, 5, 20, 0.3, 0.5)
        late, _ = sustained_support_reward(measure, 15, 20, 0.3, 0.5)
        torch.testing.assert_close(early, torch.tensor([0.3, 0., 0.]))
        torch.testing.assert_close(late, torch.tensor([0.8, 0., 0.]))
        torch.testing.assert_close(supported, torch.tensor([True, False, False]))

    def test_hand_reach_reward_requires_stance_and_shin_support(self):
        previous = {"palm_relative_x": torch.tensor([[0., 0.], [0., 0.]])}
        current = {"palm_relative_x": torch.tensor([[0.04, 0.], [0.04, 0.]]),
                   "palm_shin_force": torch.tensor([[0., 5., 5., 0.],
                                                     [0., 0., 5., 0.]]),
                   "chest_height": torch.tensor([0.2, 0.2]),
                   "head_height": torch.tensor([0.2, 0.2])}
        reward = supported_hand_reach_reward(
            previous, current, torch.zeros(2, dtype=torch.long),
            torch.ones(2, dtype=torch.bool), 10.)
        torch.testing.assert_close(reward, torch.tensor([0.4, 0.]))
        self.assertTrue(torch.equal(supported_hand_reach_reward(
            previous, current, torch.zeros(2, dtype=torch.long),
            torch.ones(2, dtype=torch.bool), 0.), torch.zeros(2)))

    def setUp(self):
        parameters = torch.tensor([[0.5, 0.3, 0.4, 0.2, 0.1, -0.1]])
        self.stepper = TactileStepper(parameters, lift_steps=2, reach_steps=2,
                                     plant_steps=1, max_plant_steps=3, cooldown_steps=1)
        self.base = torch.zeros((1, len(PALM_JOINTS)))

    def measure(self, right_hand=5., left_hand=5.):
        return {"palm_shin_force": torch.tensor([[right_hand, left_hand, 5., 5.]])}

    def test_observation_exposes_stage_side_and_unload(self):
        initial = self.stepper.observation()
        self.assertEqual(initial.shape, (1, self.stepper.observation_size))
        torch.testing.assert_close(initial[0, :6], torch.tensor([1., 0., 0., 0., 1., 0.]))
        self.stepper(self.base, self.measure())
        lifting = self.stepper.observation()
        self.assertEqual(float(lifting[0, 1]), 1.)
        self.stepper(self.base, self.measure(right_hand=0.))
        self.assertEqual(float(self.stepper.observation()[0, 8]), 1.)
        self.stepper.reset()
        torch.testing.assert_close(self.stepper.observation(), initial)

    def test_lift_reach_plant_and_alternate_after_contact(self):
        first = self.stepper(self.base, self.measure())
        self.assertEqual(int(self.stepper.stage[0]), 1)
        self.assertAlmostEqual(float(first[0, LIMB_IDS[0]]), -0.5)
        self.assertAlmostEqual(float(first[0, SHOULDER_LIFT_IDS[0]]), -0.3)
        self.stepper(self.base, self.measure(right_hand=0.))
        reach = self.stepper(self.base, self.measure(right_hand=0.))
        self.assertEqual(int(self.stepper.stage[0]), 2)
        self.assertAlmostEqual(float(reach[0, LIMB_IDS[0]]), 0.4)
        self.stepper(self.base, self.measure(right_hand=0.))
        self.stepper(self.base, self.measure())
        self.assertEqual(int(self.stepper.stage[0]), 3)
        self.stepper(self.base, self.measure())
        self.assertEqual(int(self.stepper.completed[0]), 1)
        self.assertEqual(int(self.stepper.unload_events[0]), 1)
        self.assertEqual(int(self.stepper.swing_side[0]), 1)

    def test_never_counts_contact_without_prior_lift_as_a_step(self):
        for _ in range(12):
            self.stepper(self.base, self.measure())
        self.assertEqual(int(self.stepper.completed[0]), 0)
        self.assertEqual(int(self.stepper.unload_events[0]), 0)
        self.assertGreaterEqual(int(self.stepper.aborted[0]), 1)
        self.assertEqual(int(self.stepper.aborted[0]), int(self.stepper.plant_timeouts[0]))

    def test_recontact_requires_forward_hand_reach_when_enabled(self):
        controller = TactileStepper(self.stepper.parameters, lift_steps=1,
                                    reach_steps=1, plant_steps=1,
                                    max_plant_steps=2, cooldown_steps=1,
                                    min_reach_m=0.03)
        with self.assertRaises(ValueError):
            controller(self.base, self.measure())
        def observation(right_force, right_x):
            return {"palm_shin_force": torch.tensor([[right_force, 5., 5., 5.]]),
                    "palm_relative_x": torch.tensor([[right_x, 0.]])}
        controller(self.base, observation(5., 0.))
        controller(self.base, observation(0., 0.))
        controller(self.base, observation(0., 0.))
        controller(self.base, observation(5., 0.01))
        self.assertEqual(int(controller.completed[0]), 0)
        controller(self.base, observation(5., 0.04))
        self.assertEqual(int(controller.completed[0]), 1)
        self.assertEqual(int(controller.forward_qualified_completions[0]), 1)
        self.assertAlmostEqual(float(controller.metrics(5)["mean_completed_reach_m"][0]),
                               0.04)

    def test_aborts_when_stance_hand_loses_contact(self):
        self.stepper(self.base, self.measure())
        recovered = self.stepper(self.base, self.measure(left_hand=0.))
        self.assertEqual(int(self.stepper.stage[0]), 0)
        self.assertEqual(int(self.stepper.aborted[0]), 1)
        self.assertEqual(int(self.stepper.stance_loss_aborts[0]), 1)
        self.assertEqual(int(self.stepper.plant_timeouts[0]), 0)
        torch.testing.assert_close(recovered, self.base)

    def test_stance_contact_grace_resets_after_recovery(self):
        controller = TactileStepper(self.stepper.parameters, lift_steps=2,
                                    reach_steps=2, cooldown_steps=1,
                                    stance_loss_grace_steps=1)
        controller(self.base, self.measure())
        controller(self.base, self.measure(left_hand=0.))
        self.assertEqual(int(controller.stage[0]), 1)
        controller(self.base, self.measure())
        self.assertEqual(int(controller.stance_gap[0]), 0)
        controller(self.base, self.measure(left_hand=0.))
        self.assertEqual(int(controller.aborted[0]), 0)
        controller(self.base, self.measure(left_hand=0.))
        self.assertEqual(int(controller.stage[0]), 0)
        self.assertEqual(int(controller.stance_loss_aborts[0]), 1)

    def test_stance_brace_strength_changes_only_support_arm(self):
        controller = TactileStepper(self.stepper.parameters, cooldown_steps=1,
                                    stance_brace_strength=0.5)
        base = torch.zeros_like(self.base)
        drive = controller(base, self.measure())
        self.assertAlmostEqual(float(drive[0, LIMB_IDS[1]]), 0.5)
        self.assertAlmostEqual(float(drive[0, SHOULDER_LIFT_IDS[1]]), 0.5)
        self.assertAlmostEqual(float(drive[0, LIMB_IDS[0]]), -0.5)
        self.assertAlmostEqual(float(drive[0, LIMB_IDS[2]]), 0.)

    def test_abort_can_switch_next_swing_side(self):
        controller = TactileStepper(self.stepper.parameters, cooldown_steps=1,
                                    alternate_on_abort=True)
        controller(self.base, self.measure())
        controller(self.base, self.measure(left_hand=0.))
        self.assertEqual(int(controller.swing_side[0]), 1)
        controller(self.base, self.measure())
        next_drive = controller(self.base, self.measure())
        self.assertEqual(int(controller.stage[0]), 1)
        self.assertAlmostEqual(float(next_drive[0, LIMB_IDS[1]]), -0.5)

    def test_absolute_swing_targets_override_support_policy(self):
        controller = TactileStepper(self.stepper.parameters, cooldown_steps=1,
                                    absolute_targets=True)
        base = torch.full_like(self.base, 0.7)
        drive = controller(base, self.measure())
        self.assertAlmostEqual(float(drive[0, LIMB_IDS[0]]), -0.5)
        self.assertAlmostEqual(float(drive[0, SHOULDER_LIFT_IDS[0]]), -0.3)
        self.assertAlmostEqual(float(drive[0, LIMB_IDS[1]]), 0.7)

    def test_forward_reach_keeps_hand_lifted_then_places_it_forward(self):
        controller = TactileStepper(self.stepper.parameters, lift_steps=1, reach_steps=1,
                                    plant_steps=1, cooldown_steps=1,
                                    absolute_targets=True, forward_reach=True)
        base = torch.full_like(self.base, 0.7)
        controller(base, self.measure())
        reach = controller(base, self.measure(right_hand=0.))
        self.assertEqual(int(controller.stage[0]), 2)
        self.assertAlmostEqual(float(reach[0, LIMB_IDS[0]]), -0.5)
        self.assertAlmostEqual(float(reach[0, SHOULDER_LIFT_IDS[0]]), 0.3)
        plant = controller(base, self.measure())
        self.assertEqual(int(controller.stage[0]), 3)
        self.assertAlmostEqual(float(plant[0, LIMB_IDS[0]]), 0.4)
        self.assertAlmostEqual(float(plant[0, SHOULDER_LIFT_IDS[0]]), 0.3)

    def test_strong_plant_replaces_weak_plant_target(self):
        controller = TactileStepper(self.stepper.parameters, lift_steps=1, reach_steps=1,
                                    cooldown_steps=1, absolute_targets=True,
                                    forward_reach=True, strong_plant=True)
        base = torch.full_like(self.base, 0.7)
        controller(base, self.measure())
        controller(base, self.measure(right_hand=0.))
        planted = controller(base, self.measure(right_hand=0.))
        self.assertEqual(int(controller.stage[0]), 3)
        self.assertAlmostEqual(float(planted[0, LIMB_IDS[0]]), 1.)
        self.assertAlmostEqual(float(planted[0, SHOULDER_LIFT_IDS[0]]), 1.)

    def test_absolute_swing_reports_overridden_policy_actions(self):
        controller = TactileStepper(self.stepper.parameters, cooldown_steps=1,
                                    absolute_targets=True, forward_reach=True,
                                    strong_plant=True)
        controller(self.base, self.measure())
        overridden = controller.override_mask[0]
        self.assertTrue(bool(overridden[LIMB_IDS[0]]))
        self.assertTrue(bool(overridden[SHOULDER_LIFT_IDS[0]]))
        self.assertTrue(bool(overridden[WRIST_IDS[0]]))
        self.assertFalse(bool(overridden[LIMB_IDS[1]]))
        self.assertFalse(bool(overridden[LIMB_IDS[5]]))
        controller(self.base, self.measure(left_hand=0.))
        self.assertFalse(bool(controller.override_mask.any()))

    def test_residual_swing_keeps_actor_control_and_ppo_credit(self):
        controller = TactileStepper(self.stepper.parameters, lift_steps=1, reach_steps=1,
                                    cooldown_steps=1, absolute_targets=True,
                                    forward_reach=True, strong_plant=True,
                                    residual_strength=0.25)
        base = torch.full_like(self.base, 0.7)
        reference = torch.ones_like(base)
        with self.assertRaises(ValueError):
            controller(base, self.measure())
        lift = controller(base, self.measure(), reference)
        self.assertAlmostEqual(float(lift[0, LIMB_IDS[0]]),
                               -0.5 + 0.25 * (0.7 - 1.0))
        self.assertFalse(bool(controller.override_mask.any()))
        controller(base, self.measure(right_hand=0.), reference)
        plant = controller(base, self.measure(right_hand=0.), reference)
        self.assertAlmostEqual(float(plant[0, LIMB_IDS[0]]),
                               1.0 + 0.25 * (0.7 - 1.0))
        self.assertFalse(bool(controller.override_mask.any()))
        controller.reset()
        unchanged = controller(base, self.measure(), base)
        self.assertAlmostEqual(float(unchanged[0, LIMB_IDS[0]]), -0.5)

    def test_residual_strength_rejects_out_of_range_values(self):
        with self.assertRaises(ValueError):
            TactileStepper(self.stepper.parameters, residual_strength=1.1)

    def test_diagonal_leg_push_reverses_swing_hip_drive_at_plant(self):
        controller = TactileStepper(self.stepper.parameters, lift_steps=1, reach_steps=1,
                                    cooldown_steps=1, leg_push=True)
        first = controller(self.base, self.measure())
        self.assertAlmostEqual(float(first[0, LIMB_IDS[5]]), 0.1)
        controller(self.base, self.measure(right_hand=0.))
        planted = controller(self.base, self.measure(right_hand=0.))
        self.assertEqual(int(controller.stage[0]), 3)
        self.assertAlmostEqual(float(planted[0, LIMB_IDS[5]]), -0.1)

    def test_requires_shin_and_both_hands_to_start(self):
        waiting = self.stepper(self.base, {"palm_shin_force": torch.tensor([[5., 5., 0., 0.]])})
        self.assertEqual(int(self.stepper.attempts[0]), 0)
        torch.testing.assert_close(waiting, self.base)
        self.stepper(self.base, self.measure(left_hand=0.))
        self.assertEqual(int(self.stepper.attempts[0]), 0)


if __name__ == "__main__":
    unittest.main()
