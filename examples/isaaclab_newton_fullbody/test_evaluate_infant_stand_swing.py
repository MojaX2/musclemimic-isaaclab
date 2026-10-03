"""Check the unilateral standing swing command."""

import math
import unittest

import torch

from evaluate_infant_stand_swing import ANKLE, HIP, KNEE, swing_drives, swing_envelope
from infant_crawl_env import CRAWL_JOINTS


class StandingSwingTests(unittest.TestCase):
    def test_envelope_bounds_and_peak(self):
        self.assertEqual(swing_envelope(0.15), 0)
        self.assertAlmostEqual(swing_envelope(0.525), 1)
        self.assertEqual(swing_envelope(0.9), 0)
        with self.assertRaises(ValueError):
            swing_envelope(0.5, 0.9, 0.15)

    def test_swing_targets_only_right_leg(self):
        base = torch.full((2, len(CRAWL_JOINTS)), 0.2)
        control = swing_drives(base, "control", 1)
        self.assertIs(control, base)
        hip = swing_drives(base, "hip", 1)
        self.assertTrue(torch.equal(hip[:, HIP], torch.full((2,), -1.)))
        self.assertTrue(torch.equal(hip[:, KNEE], base[:, KNEE]))
        combined = swing_drives(base, "hip_knee", 1)
        self.assertTrue(torch.equal(combined[:, HIP], torch.full((2,), -1.)))
        self.assertTrue(torch.equal(combined[:, KNEE], torch.full((2,), -1.)))
        self.assertTrue(torch.equal(combined[:, ANKLE], torch.full((2,), 0.5)))
        unchanged = [index for index in range(len(CRAWL_JOINTS))
                     if index not in (HIP, KNEE, ANKLE)]
        self.assertTrue(torch.equal(combined[:, unchanged], base[:, unchanged]))
        self.assertTrue(torch.equal(base, torch.full_like(base, 0.2)))
        with self.assertRaises(ValueError):
            swing_drives(base, "unsupported", math.pi)


if __name__ == "__main__":
    unittest.main()
