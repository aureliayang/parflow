"""Python configuration contract for the experimental spherical input keys."""

from pathlib import Path
import tempfile
import unittest

from parflow import Run
from parflow.tools.database.domains import BoolDomain
from parflow.tools.io import read_pfidb


class SphericalGeometryKeysTest(unittest.TestCase):
    def test_default_is_boolean_false(self):
        run = Run("spherical_keys_default")
        self.assertIs(run.Solver.Experimental.SphericalAccumulation, False)
        details = run.Solver.Experimental._details_["SphericalAccumulation"]
        self.assertIn("BoolDomain", details["domains"])

    def test_boolean_and_radius_filename_round_trip(self):
        run = Run("spherical_keys_round_trip")
        filename = "radius_center.pfb"
        run.Solver.SphericalGeometry.RadiusCenter.FileName = filename
        self.assertIn("FileName", run.Solver.SphericalGeometry.RadiusCenter._details_)
        with tempfile.TemporaryDirectory() as directory:
            for enabled in (True, False):
                run.Solver.Experimental.SphericalAccumulation = enabled
                run.write(file_name="spherical", working_directory=directory)
                data = read_pfidb(Path(directory) / "spherical.pfidb")
                self.assertIs(data["Solver.Experimental.SphericalAccumulation"], enabled)
                self.assertEqual(
                    data["Solver.SphericalGeometry.RadiusCenter.FileName"], filename
                )

    def test_boolean_domain_rejects_integer_switch(self):
        domain = BoolDomain()
        self.assertFalse(domain.validate(True))
        self.assertFalse(domain.validate(False))
        self.assertTrue(domain.validate(1))
        self.assertTrue(domain.validate(0))


if __name__ == "__main__":
    unittest.main()
