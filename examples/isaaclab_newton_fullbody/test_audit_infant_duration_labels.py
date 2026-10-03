"""Check experiment duration extraction without running a simulator."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from audit_infant_duration_labels import actual_duration


class DurationLabelAuditTests(unittest.TestCase):
    def test_reported_duration_takes_precedence(self):
        self.assertEqual(actual_duration({"duration_s": 8}, {}), (8., "reported"))

    def test_computed_duration_uses_scene_timestep(self):
        with TemporaryDirectory() as directory:
            scene = Path(directory) / "scene.xml"
            scene.touch()
            self.assertEqual(actual_duration({"scene": str(scene), "control_steps": 160,
                                             "physics_per_control": 10}, {scene: 0.005}),
                             (8., "computed"))

    def test_unresolved_duration_is_not_assumed(self):
        self.assertEqual(actual_duration({"control_steps": 160}, {}),
                         (None, "metadata_missing"))


if __name__ == "__main__":
    unittest.main()
