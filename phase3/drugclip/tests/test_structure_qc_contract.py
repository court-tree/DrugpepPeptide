from __future__ import annotations

import unittest

from phase3.drugclip.structure_qc import ParsedResidue, has_complete_backbone


def atom(name: str) -> dict[str, object]:
    return {"atom_name": name, "x": 0.0, "y": 0.0, "z": 0.0}


class StructureQcContractTests(unittest.TestCase):
    def test_receptor_residue_requires_n_ca_c(self) -> None:
        complete = ParsedResidue("A", "1", "ALA", (atom("N"), atom("CA"), atom("C")))
        ca_only = ParsedResidue("A", "2", "ALA", (atom("CA"),))
        self.assertTrue(has_complete_backbone(complete))
        self.assertFalse(has_complete_backbone(ca_only))


if __name__ == "__main__":
    unittest.main()
