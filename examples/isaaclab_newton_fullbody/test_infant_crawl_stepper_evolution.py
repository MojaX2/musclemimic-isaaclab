"""Check the stepper GA objective balances progress and stance quality."""

import unittest

import torch

from evolve_infant_crawl_stepper import stepper_score


class StepperEvolutionTests(unittest.TestCase):
    def test_score_prefers_supported_forward_stepping(self):
        result = {"forward_displacement_m": torch.tensor([0.1, -0.05]),
                  "minimum_support_fraction": torch.tensor([0.7, 0.9]),
                  "final_quarter_support_fraction": torch.tensor([0.6, 0.9]),
                  "hand_switches": torch.tensor([2., 0.]),
                  "crawl_success": torch.tensor([True, False])}
        score = stepper_score(result, torch.zeros((2, 6)))
        self.assertGreater(float(score[0]), float(score[1]))


if __name__ == "__main__":
    unittest.main()
