"""Check that forward sliding is not classified as a crawl gait."""

import unittest

import numpy as np

from infant_crawl_metrics import crawl_locomotion_metrics


class InfantCrawlMetricsTests(unittest.TestCase):
    def test_late_collapse_rejects_legacy_success(self):
        forces = np.zeros((12, 1, 4))
        for step in range(9):
            forces[step, 0, [step % 2, 2 + step % 2]] = 5
        height = np.full((12, 1), 0.2)
        result = crawl_locomotion_metrics(forces, height, height, np.zeros((12, 1)),
                                          np.array([0.1]))
        self.assertEqual(result["legacy_forward_crawl_success_per_world"], [True])
        self.assertEqual(result["forward_crawl_success_per_world"], [False])
        self.assertEqual(result["late_minimum_support_fraction_per_world"], [0.0])

    def test_alternating_support_with_forward_progress_qualifies(self):
        forces = np.zeros((10, 2, 4))
        for step in range(10):
            if step % 2 == 0:
                forces[step, :, [0, 2]] = 5
            else:
                forces[step, :, [1, 3]] = 5
        height = np.full((10, 2), 0.2)
        pelvis = np.zeros((10, 2))
        result = crawl_locomotion_metrics(forces, height, height, pelvis,
                                          np.array([0.1, 0.0]))
        self.assertEqual(result["forward_crawl_success_per_world"], [True, False])
        self.assertEqual(result["forward_crawl_success_fraction"], 0.5)
        self.assertEqual(result["hand_support_switches_per_world"], [9, 9])

    def test_no_alternation_does_not_qualify(self):
        forces = np.full((10, 1, 4), 5.)
        height = np.full((10, 1), 0.2)
        result = crawl_locomotion_metrics(forces, height, height, np.zeros((10, 1)),
                                          np.array([0.2]))
        self.assertEqual(result["forward_crawl_success_fraction"], 0.)

    def test_single_contact_switch_is_not_sustained_gait(self):
        forces = np.zeros((10, 1, 4))
        forces[:5, 0, [0, 2]] = 5
        forces[5:, 0, [1, 3]] = 5
        height = np.full((10, 1), 0.2)
        result = crawl_locomotion_metrics(forces, height, height, np.zeros((10, 1)),
                                          np.array([0.1]))
        self.assertEqual(result["hand_support_switches_per_world"], [1])
        self.assertEqual(result["forward_crawl_success_fraction"], 0.)

    def test_switches_while_crawling_low_do_not_count(self):
        forces = np.zeros((10, 1, 4))
        forces[:4, 0, [0, 2]] = 5
        forces[4:6, 0, [1, 3]] = 5
        forces[6:8, 0, [1, 3]] = 5
        forces[8:, 0, [0, 2]] = 5
        height = np.full((10, 1), 0.2)
        height[4:6] = 0.1
        result = crawl_locomotion_metrics(forces, height, height, np.zeros((10, 1)),
                                          np.array([0.1]))
        self.assertEqual(result["hand_support_switches_per_world"], [1])
        self.assertEqual(result["forward_crawl_success_fraction"], 0.)


if __name__ == "__main__":
    unittest.main()
