"""Check diagonal oscillator routing in palm-control action space."""

import unittest
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from evaluate_infant_crawl_cpg import load_parameters
from evolve_infant_crawl_cpg import (LIMB_IDS, WRIST_IDS, apply_gait_gain, cpg_drives,
                                    coupled_sustained_gait_bonus, crawl_success_gate,
                                    gait_objective_score,
                                    guard_arm_oscillation, height_gait_gain, shin_gait_gain,
                                    sustained_gait_bonus)
from infant_crawl_env import PALM_JOINTS
from infant_crawl_oscillator import OSCILLATOR_IDS, SHOULDER_LIFT_IDS, fixed_limb_drives


class InfantCrawlCpgTests(unittest.TestCase):
    def test_legacy_success_without_late_support_is_not_sustained_crawl(self):
        legacy, sustained = crawl_success_gate(
            torch.tensor([0.06]), torch.tensor([0.78]), torch.tensor([0.77]),
            torch.tensor([0.14]), torch.tensor([0.]), torch.tensor([2.]),
            torch.tensor([6.]))
        self.assertTrue(bool(legacy[0]))
        self.assertFalse(bool(sustained[0]))

    def test_diagonal_limbs_receive_opposite_drives(self):
        base = torch.zeros((1, len(PALM_JOINTS)))
        parameters = torch.tensor([[1., 0.5, 0., 0.25, 0., 0.]])
        drives = cpg_drives(base, parameters, 0.25)
        self.assertAlmostEqual(float(drives[0, LIMB_IDS[0]]), 0.5)
        self.assertAlmostEqual(float(drives[0, LIMB_IDS[1]]), -0.5)
        self.assertAlmostEqual(float(drives[0, LIMB_IDS[4]]), -0.25)
        self.assertAlmostEqual(float(drives[0, LIMB_IDS[5]]), 0.25)

    def test_shape_validation(self):
        with self.assertRaises(ValueError):
            cpg_drives(torch.zeros((1, len(PALM_JOINTS))), torch.zeros((1, 5)), 0.)

    def test_wrist_oscillation_changes_opposite_hands(self):
        base = torch.zeros((1, len(PALM_JOINTS)))
        parameters = torch.tensor([[1., 0., 0., 0., 0., 0., 0.4]])
        drives = cpg_drives(base, parameters, 0.25)
        self.assertAlmostEqual(float(drives[0, WRIST_IDS[0]]), 0.4)
        self.assertAlmostEqual(float(drives[0, WRIST_IDS[1]]), -0.4)

    def test_shoulder_lift_alternates_without_changing_legacy_drives(self):
        base = torch.zeros((1, len(PALM_JOINTS)))
        legacy = torch.tensor([[1., 0.2, 0., 0., 0., 0., 0.1]])
        torch.testing.assert_close(cpg_drives(base, torch.cat((legacy, torch.zeros((1, 1))), 1), 0.25),
                                   cpg_drives(base, legacy, 0.25))
        with_lift = torch.cat((legacy, torch.tensor([[0.4]])), 1)
        first = cpg_drives(base, with_lift, 0.25)
        second = cpg_drives(base, with_lift, 0.75)
        self.assertAlmostEqual(float(first[0, SHOULDER_LIFT_IDS[0]]), -0.4)
        self.assertAlmostEqual(float(first[0, SHOULDER_LIFT_IDS[1]]), 0.)
        self.assertAlmostEqual(float(second[0, SHOULDER_LIFT_IDS[0]]), 0.)
        self.assertAlmostEqual(float(second[0, SHOULDER_LIFT_IDS[1]]), -0.4)

    def test_phase_offset_reverses_shoulder_lift_timing(self):
        base = torch.zeros((1, len(PALM_JOINTS)))
        parameters = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0.4]])
        old = cpg_drives(base, parameters, 0.25)
        zero_phase = cpg_drives(base, torch.cat((parameters, torch.zeros((1, 1))), 1), 0.25)
        torch.testing.assert_close(zero_phase, old)
        reversed_phase = cpg_drives(base, torch.cat((parameters, torch.tensor([[math.pi]])), 1), 0.25)
        self.assertAlmostEqual(float(reversed_phase[0, SHOULDER_LIFT_IDS[0]]), 0.)
        self.assertAlmostEqual(float(reversed_phase[0, SHOULDER_LIFT_IDS[1]]), -0.4)

    def test_legacy_oscillator_is_padded_with_zero_wrist_drive(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "oscillator.npz"
            np.savez_compressed(path, parameters=np.arange(6, dtype=np.float32))
            loaded = load_parameters(path, 7, "cpu")
            torch.testing.assert_close(loaded, torch.tensor([0., 1., 2., 3., 4., 5., 0.]))
            with self.assertRaises(ValueError):
                load_parameters(path, 5, "cpu")

    def test_wrist_oscillator_is_padded_with_zero_shoulder_lift(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "oscillator.npz"
            np.savez_compressed(path, parameters=np.arange(7, dtype=np.float32))
            torch.testing.assert_close(load_parameters(path, 8, "cpu"),
                                       torch.tensor([0., 1., 2., 3., 4., 5., 6., 0.]))
            torch.testing.assert_close(load_parameters(path, 9, "cpu"),
                                       torch.tensor([0., 1., 2., 3., 4., 5., 6., 0., 0.]))

    def test_fixed_limb_reference_includes_shoulder_lift_for_eight_parameters(self):
        learned = torch.full((1, len(PALM_JOINTS)), 0.7)
        reference = torch.zeros_like(learned)
        parameters = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0.4]])
        combined = fixed_limb_drives(learned, reference, parameters, 0.25)
        self.assertAlmostEqual(float(combined[0, SHOULDER_LIFT_IDS[0]]), -0.4)
        self.assertAlmostEqual(float(combined[0, SHOULDER_LIFT_IDS[1]]), 0.)

    def test_gait_objective_prefers_supported_alternating_motion(self):
        score = gait_objective_score(
            forward=torch.tensor([0.1, 0.1]),
            support_fraction=torch.tensor([0.8, 0.3]),
            elevated_fraction=torch.tensor([0.9, 0.7]),
            four_fraction=torch.tensor([0.1, 0.0]),
            pelvis_fraction=torch.zeros(2),
            hand_switches=torch.tensor([2., 0.]),
            shin_switches=torch.tensor([2., 2.]),
            crawl_success=torch.tensor([True, False]),
            amplitudes=torch.zeros((2, 5)))
        self.assertGreater(float(score[0]), float(score[1]))

    def test_sustained_bonus_prefers_late_progress_and_support(self):
        bonus = sustained_gait_bonus(torch.tensor([0.05, -0.05]),
                                     torch.tensor([0.8, 0.2]),
                                     torch.tensor([0.9, 0.4]))
        self.assertGreater(float(bonus[0]), float(bonus[1]))
        self.assertLess(float(sustained_gait_bonus(torch.tensor([-0.1]),
                                                    torch.tensor([0.]),
                                                    torch.tensor([0.]))), 0.)

    def test_coupled_bonus_requires_supported_late_progress(self):
        late_forward = torch.tensor([0.08, 0.08, 0., -0.08])
        final_support = torch.tensor([0.8, 0., 0.8, 0.8])
        final_elevated = torch.tensor([0.9, 0., 0.9, 0.9])
        hand_switches = torch.tensor([2., 2., 0., 0.])
        bonus = coupled_sustained_gait_bonus(late_forward, final_support,
                                             final_elevated, hand_switches)
        self.assertGreater(float(bonus[0]), float(bonus[1]))
        self.assertEqual(float(bonus[2]), 0.)
        self.assertLess(float(bonus[3]), 0.)

    def test_contact_guard_keeps_stance_arm_when_other_palm_is_unloaded(self):
        base = torch.zeros((2, len(PALM_JOINTS)))
        oscillated = torch.ones_like(base)
        force = torch.tensor([[5., 0., 5., 5.], [0., 5., 5., 5.]])
        guarded, allowed = guard_arm_oscillation(base, oscillated, force,
                                                  shoulder_lift=True)
        torch.testing.assert_close(allowed, torch.tensor([[False, True], [True, False]]))
        for index in (LIMB_IDS[0], LIMB_IDS[2], WRIST_IDS[0], SHOULDER_LIFT_IDS[0]):
            torch.testing.assert_close(guarded[:, index], torch.tensor([0., 1.]))
        for index in (LIMB_IDS[1], LIMB_IDS[3], WRIST_IDS[1], SHOULDER_LIFT_IDS[1]):
            torch.testing.assert_close(guarded[:, index], torch.tensor([1., 0.]))

    def test_height_gate_disables_oscillation_below_support_height(self):
        gain = height_gait_gain(torch.tensor([0.15, 0.18, 0.21]))
        torch.testing.assert_close(gain, torch.tensor([0., 0.5, 1.]))

    def test_shin_gate_uses_only_shin_contact_force(self):
        force = torch.tensor([[100., 100., 5., 5.], [0., 0., 10., 10.],
                              [0., 0., 15., 15.]])
        torch.testing.assert_close(shin_gait_gain(force), torch.tensor([0., 0.5, 1.]))

    def test_leg_only_gate_preserves_arm_oscillation(self):
        base = torch.zeros((1, len(PALM_JOINTS)))
        drives = torch.ones_like(base)
        adjusted = apply_gait_gain(base, drives, torch.tensor([0.25]), leg_only=True)
        self.assertAlmostEqual(float(adjusted[0, LIMB_IDS[0]]), 1.)
        self.assertAlmostEqual(float(adjusted[0, LIMB_IDS[4]]), 0.25)

    def test_fixed_limbs_keep_oscillator_and_train_other_joints(self):
        learned = torch.full((1, len(PALM_JOINTS)), 0.7)
        reference = torch.zeros_like(learned)
        parameters = torch.tensor([[1., 0.4, 0., 0., 0., 0., 0.]])
        combined = fixed_limb_drives(learned, reference, parameters, 0.25)
        self.assertAlmostEqual(float(combined[0, LIMB_IDS[0]]), 0.4)
        self.assertAlmostEqual(float(combined[0, LIMB_IDS[1]]), -0.4)
        other_joint = next(index for index in range(len(PALM_JOINTS))
                           if index not in OSCILLATOR_IDS)
        self.assertAlmostEqual(float(combined[0, other_joint]), 0.7)


if __name__ == "__main__":
    unittest.main()
