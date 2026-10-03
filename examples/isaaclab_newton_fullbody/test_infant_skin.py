"""Check infant contact-region coverage and prop exclusion."""

from pathlib import Path
import unittest

import mujoco
import numpy as np

from developmental_skin import REGIONS
from infant_skin import infant_geom_region_ids


MIMO_SCENE = Path(__file__).resolve().parents[3] / "MIMo/mimoEnv/assets/benchmarkv2_scene.xml"


@unittest.skipUnless(MIMO_SCENE.exists(), "MIMo reference checkout unavailable")
class InfantSkinTests(unittest.TestCase):
    def test_all_regions_covered_and_props_excluded(self):
        model = mujoco.MjModel.from_xml_path(str(MIMO_SCENE))
        mapping = infant_geom_region_ids(model)
        self.assertEqual(set(mapping[mapping >= 0]), set(range(len(REGIONS))))
        self.assertEqual(mapping[0], -1)
        self.assertEqual(mapping[1], -1)
        self.assertEqual(mapping[2], -1)
        self.assertGreater(int(np.sum(mapping >= 0)), 50)


if __name__ == "__main__":
    unittest.main()
