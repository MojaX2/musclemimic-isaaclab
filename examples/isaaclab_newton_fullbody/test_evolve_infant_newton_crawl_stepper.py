"""Check that Newton stepper selection rewards supported progress."""

import unittest

import torch

from evolve_infant_newton_crawl_stepper import sustained_crawl_score


class NewtonStepperEvolutionTests(unittest.TestCase):
    def test_forward_progress_requires_sustained_support(self):
        result = {"forward_displacement_m": torch.tensor([0.1, 0.1, -0.1, 0.]),
                  "minimum_support_fraction": torch.ones(4),
                  "final_quarter_support_fraction": torch.tensor([1., 0., 1., 1.]),
                  "hand_switches": torch.full((4,), 2.)}
        parameters = torch.zeros((4, 6))
        scores = sustained_crawl_score(result, parameters)
        self.assertGreater(float(scores[0]), float(scores[1]))
        self.assertGreater(float(scores[0]), float(scores[2]))
        self.assertGreater(float(scores[0]), float(scores[3]))

    def test_completed_recontact_beats_abort_only_with_late_support(self):
        result = {"forward_displacement_m": torch.zeros(4),
                  "minimum_support_fraction": torch.ones(4),
                  "final_quarter_support_fraction": torch.tensor([1., 1., 0., 0.]),
                  "hand_switches": torch.zeros(4),
                  "step_completions": torch.tensor([2., 0., 2., 0.]),
                  "step_aborts": torch.tensor([0., 2., 0., 2.])}
        scores = sustained_crawl_score(result, torch.zeros((4, 6)))
        self.assertGreater(float(scores[0]), float(scores[1]))
        self.assertEqual(float(scores[2]), float(scores[3]) + 0.5)

    def test_forward_priority_rewards_only_supported_progress(self):
        result = {"forward_displacement_m": torch.tensor([0.1, 0.1, -0.1]),
                  "minimum_support_fraction": torch.ones(3),
                  "final_quarter_support_fraction": torch.tensor([1., 0., 1.]),
                  "hand_switches": torch.zeros(3)}
        parameters = torch.zeros((3, 6))
        base = sustained_crawl_score(result, parameters)
        prioritized = sustained_crawl_score(result, parameters, forward_priority=4)
        self.assertGreater(float(prioritized[0] - base[0]), 0)
        self.assertEqual(float(prioritized[1] - base[1]), 0)
        self.assertLess(float(prioritized[2] - base[2]), 0)


if __name__ == "__main__":
    unittest.main()
