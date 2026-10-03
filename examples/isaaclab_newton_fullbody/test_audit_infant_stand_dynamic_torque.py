"""Check torque-envelope violation accounting."""

import unittest

import numpy as np

from audit_infant_stand_dynamic_torque import envelope_violations


class StandDynamicTorqueTests(unittest.TestCase):
    def test_violation_is_zero_inside_and_distance_outside(self):
        required = np.array([[-2., 0., 3.]])
        lower = np.array([[-1., -1., -1.]])
        upper = np.array([[1., 1., 1.]])
        np.testing.assert_array_equal(envelope_violations(required, lower, upper),
                                      [[1., 0., 2.]])

    def test_invalid_envelope_rejected(self):
        with self.assertRaises(ValueError):
            envelope_violations(np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 1)))
        with self.assertRaises(ValueError):
            envelope_violations(np.zeros((1, 1)), np.ones((1, 1)), np.zeros((1, 1)))


if __name__ == "__main__":
    unittest.main()
