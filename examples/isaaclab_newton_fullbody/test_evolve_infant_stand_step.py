"""Check tactile step-control phases and muscle targeting."""

import unittest
from types import SimpleNamespace

import torch

from evolve_infant_stand_step import (JOINTS, PARAMETERS, phase_window,
                                      landing_reflex_drives, selective_swing_action,
                                      step_drives, step_events,
                                      sustained_gait_success)
from infant_crawl_env import CRAWL_JOINTS


class InfantStandingStepTests(unittest.TestCase):
    def test_phase_window(self):
        elapsed = torch.tensor([float("-inf"), 0., 0.2, 0.4, 1.])
        values = phase_window(elapsed, 0., 0.4)
        self.assertTrue(torch.allclose(values, torch.tensor([0., 0., 1., 0., 0.]),
                                       atol=1e-6))

    def test_eight_muscle_offsets(self):
        base = torch.zeros((2, len(CRAWL_JOINTS)))
        parameters = torch.ones((2, len(PARAMETERS)))
        shift = torch.tensor([0.5, 0.])
        swing = torch.tensor([1., 0.5])
        result = step_drives(base, parameters, shift, swing)
        self.assertTrue(torch.equal(result[0, list(JOINTS[:3])],
                                    torch.tensor([0.5, 0.5, 0.5])))
        self.assertTrue(torch.equal(result[0, list(JOINTS[3:])],
                                    torch.tensor([-1., -1., 1., 1., 1.])))
        self.assertTrue(torch.equal(result[1, list(JOINTS[:3])], torch.zeros(3)))
        self.assertTrue(torch.equal(result[1, list(JOINTS[3:])],
                                    torch.tensor([-0.5, -0.5, 0.5, 0.5, 0.5])))
        untouched = [index for index in range(len(CRAWL_JOINTS)) if index not in JOINTS]
        self.assertTrue(torch.equal(result[:, untouched], base[:, untouched]))
        with self.assertRaises(ValueError):
            step_drives(base, torch.ones((1, len(PARAMETERS))), shift, swing)

    def test_stance_preload_moves_pitch_before_swing(self):
        base = torch.zeros((1, len(CRAWL_JOINTS)))
        parameters = torch.zeros((1, len(PARAMETERS)))
        parameters[:, 6:] = 1
        shift = torch.tensor([0.5])
        swing = torch.tensor([0.])
        self.assertTrue(torch.equal(step_drives(base, parameters, shift, swing)
                                    [0, list(JOINTS[6:])], torch.zeros(2)))
        self.assertTrue(torch.equal(step_drives(base, parameters, shift, swing, True)
                                    [0, list(JOINTS[6:])], torch.full((2,), 0.5)))

    def test_selective_excitation_only_changes_swing_actuators(self):
        actuator_count = len(CRAWL_JOINTS)
        environment = SimpleNamespace(
            source=SimpleNamespace(nu=actuator_count),
            crawl_actuators=torch.arange(actuator_count),
            action_from_drives=lambda drives, baseline, amplitude:
            torch.full((1, 2 * actuator_count), baseline))
        drives = torch.zeros((1, actuator_count))
        drives[:, JOINTS[3]] = -1
        action = selective_swing_action(environment, drives, torch.ones(1), 1)
        self.assertAlmostEqual(float(action[0, JOINTS[3]]), 1.)
        self.assertAlmostEqual(float(action[0, JOINTS[3] + actuator_count]), 0.)
        self.assertAlmostEqual(float(action[0, JOINTS[4]]), 0.5)
        self.assertAlmostEqual(float(action[0, JOINTS[0]]), 0.4)
        with self.assertRaises(ValueError):
            selective_swing_action(environment, drives, torch.ones(1), 1.1)

    def test_landing_reflex_targets_bilateral_pitch_after_contact(self):
        base = torch.zeros((3, len(CRAWL_JOINTS)))
        parameters = torch.tensor([[0.5, -0.25]]).expand(3, -1)
        elapsed = torch.tensor([-0.01, 0.1, 0.5])
        drives = landing_reflex_drives(base, parameters, elapsed, 0.5)
        self.assertTrue(torch.equal(drives[0], base[0]))
        self.assertTrue(torch.equal(drives[2], base[2]))
        self.assertTrue(torch.equal(drives[1, [JOINTS[3], JOINTS[6]]],
                                    torch.full((2,), 0.5)))
        self.assertTrue(torch.equal(drives[1, [JOINTS[5], JOINTS[7]]],
                                    torch.full((2,), -0.25)))
        with self.assertRaises(ValueError):
            landing_reflex_drives(base, parameters[:1], elapsed, 0.5)

    def test_step_requires_elevated_forward_swing_and_stable_landing(self):
        qualified = torch.zeros(1, dtype=torch.bool)
        completed = qualified.clone()
        streak = torch.zeros(1, dtype=torch.int32)
        active = torch.ones(1, dtype=torch.bool)
        support = active.clone()
        airborne = active.clone()
        qualified, streak, completed = step_events(
            qualified, streak, completed, active, support, airborne,
            torch.tensor([0.04]), torch.tensor([0.01]))
        self.assertFalse(bool(qualified[0]))
        qualified, streak, completed = step_events(
            qualified, streak, completed, active, support, airborne,
            torch.tensor([0.04]), torch.tensor([0.03]))
        self.assertTrue(bool(qualified[0]))
        for _ in range(9):
            qualified, streak, completed = step_events(
                qualified, streak, completed, active, support, ~airborne,
                torch.tensor([0.04]), torch.tensor([0.]))
        self.assertFalse(bool(completed[0]))
        qualified, streak, completed = step_events(
            qualified, streak, completed, active, support, ~airborne,
            torch.tensor([0.04]), torch.tensor([0.]))
        self.assertTrue(bool(completed[0]))

    def test_sustained_gait_requires_same_world_step_and_support(self):
        gait = torch.tensor([True, False, True])
        support_steps = torch.tensor([99, 101, 101])
        self.assertTrue(torch.equal(sustained_gait_success(gait, support_steps, 0.005),
                                    torch.tensor([False, False, True])))
        with self.assertRaises(ValueError):
            sustained_gait_success(gait, support_steps[:2], 0.005)


if __name__ == "__main__":
    unittest.main()
