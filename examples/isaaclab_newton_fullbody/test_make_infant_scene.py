"""Check exported infant MJCF retains implicit MuJoCo contact defaults."""

import unittest

from make_infant_scene import explicit_geom_gap_defaults, explicit_solref_defaults


class InfantSceneTests(unittest.TestCase):
    def test_single_contact_solref_gets_explicit_damping(self):
        xml = '<default solref="0.005"/><geom solref="-20000 -20"/>'
        self.assertEqual(explicit_solref_defaults(xml),
                         '<default solref="0.005 1"/><geom solref="-20000 -20"/>')

    def test_implicit_geom_gap_gets_explicit_zero(self):
        xml = '<default><geom/></default><worldbody><geom name="floor"/><geom gap="0.03"/></worldbody>'
        self.assertEqual(explicit_geom_gap_defaults(xml),
                         '<default><geom gap="0"/></default><worldbody>'
                         '<geom gap="0" name="floor"/><geom gap="0.03"/></worldbody>')


if __name__ == "__main__":
    unittest.main()
