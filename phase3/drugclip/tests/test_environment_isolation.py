from __future__ import annotations

import sys
import unittest

import phase3
from phase3.drugclip.config import DrugCLIPPhase3Config, REPO_ROOT
from phase3.drugclip.doctor import run_doctor


class EnvironmentIsolationTests(unittest.TestCase):
    def test_root_exports_only_new_namespace(self) -> None:
        self.assertEqual(phase3.__all__, ["drugclip"])
        self.assertNotIn("phase3.active_algorithm", sys.modules)

    def test_output_root_is_isolated(self) -> None:
        config = DrugCLIPPhase3Config()
        config.validate()
        self.assertEqual(
            config.output_root.resolve(),
            (REPO_ROOT / "phase3" / "runs" / "drugclip").resolve(),
        )

    def test_doctor_finds_no_legacy_imports(self) -> None:
        ok, checks = run_doctor()
        self.assertTrue(ok, checks)
        self.assertEqual(checks["legacy_import_offenders"], [])


if __name__ == "__main__":
    unittest.main()
