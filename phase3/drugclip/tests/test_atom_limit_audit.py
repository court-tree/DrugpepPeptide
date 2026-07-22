from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from phase3.drugclip.io_utils import write_jsonl
from phase3.drugclip.validate_data_layer import audit_phase2_atom_limit


class AtomLimitAuditTests(unittest.TestCase):
    def test_phase2_prefix_slice_is_deterministic_and_detects_broken_residue(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "interfaces.jsonl"
            atoms = []
            for residue_index in range(86):
                for atom_name in ("N", "CA", "C"):
                    atoms.append(
                        {
                            "residue_id": f"A:{residue_index + 1}",
                            "atom_name": atom_name,
                            "residue_name": "ALA",
                            "element": "C",
                            "x": float(residue_index),
                            "y": 0.0,
                            "z": 0.0,
                        }
                    )
            write_jsonl(path, [{"pair_id": "iface:test", "receptor_atoms": atoms}])
            first = audit_phase2_atom_limit(path, limit=256)
            second = audit_phase2_atom_limit(path, limit=256)
            self.assertEqual(first, second)
            self.assertEqual(first["phase2_rule"], "receptor_raw_atoms[:max_receptor_atoms]")
            self.assertEqual(first["interfaces_over_limit"], 1)
            self.assertEqual(first["interfaces_with_incomplete_N_CA_C_after_exact_phase2_slice"], 1)
            self.assertFalse(first["geometry_complete_after_exact_phase2_slice"])
            self.assertEqual(first["status"], "phase2_atom_prefix_rule_accepted")


if __name__ == "__main__":
    unittest.main()
