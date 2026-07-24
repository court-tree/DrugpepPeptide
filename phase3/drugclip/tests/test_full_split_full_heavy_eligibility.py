from __future__ import annotations

import unittest
from unittest.mock import patch

from phase3.drugclip.audit_full_split_full_heavy_eligibility import (
    _classification_report,
    _coverage_scenarios,
    _split_eligible_report,
    build_sequence_eligibility_registry,
)
from phase3.drugclip.full_heavy_adaptation_contract import ATOM_CAP_EXCLUSIVE
from phase3.drugclip.train_only_torsion_prior_prototype import context_key


def _pair(
    pair_id: str,
    sequence: str,
    *,
    split: str = "train",
    relation: str | None = None,
    evidence_ids: tuple[str, ...] = ("e1",),
    source: str = "BioLiP2",
) -> dict:
    return {
        "interface_pair_id": pair_id,
        "split": split,
        "biological_pair_id": relation or f"relation:{pair_id}",
        "biological_receptor_id": f"receptor:{pair_id}",
        "receptor_interface_id": f"interface:{pair_id}",
        "peptide_sequence": sequence,
        "evidence_ids": list(evidence_ids),
        "source_database": source,
        "source_databases": [source],
        "receptor_family": "rfam:test",
        "structure_pdb_ids": ["1abc"],
    }


def _audit(classification: str) -> dict:
    return {
        "evidence_id": "e1",
        "chemistry_classification": classification,
        "exclusion_reason": (
            None if classification == "ordinary_linear_standard"
            else f"reason:{classification}"
        ),
        "structure_path": "peptide.pdb",
        "receptor_structure_path": "receptor.pdb",
        "source_database": "BioLiP2",
        "resolved_peptide_chain": "P",
        "terminal_state_determined": (
            classification == "ordinary_linear_standard"
        ),
        "peptide_receptor_covalent_connection_detected": (
            classification == "receptor_covalent"
        ),
        "peptide_receptor_explicit_connections": [],
        "peptide_other_covalent_geometry": [],
        "minimum_peptide_other_covalent_distance_angstrom": None,
        "detectable_ss_bond": classification == "known_disulfide",
        "ss_bond_evidence": [],
        "head_to_tail_closure_detected": (
            classification == "cyclic_or_crosslinked"
        ),
        "noncanonical_internal_connections": [],
        "modified_residue_detected": (
            classification == "modified_or_nonstandard"
        ),
        "modified_residue_positions": [],
        "residue_names": [],
    }


def _torsion_groups(sequences: list[str]) -> dict[str, list[dict]]:
    return {
        context_key(sequence, index): [{}]
        for sequence in sequences
        for index in range(len(sequence))
    }


class FullSplitEligibilityAuditTests(unittest.TestCase):
    def test_any_unsafe_instance_excludes_whole_sequence(self) -> None:
        pairs = [
            _pair("p1", "AAAA", evidence_ids=("safe",)),
            _pair("p2", "AAAA", evidence_ids=("unsafe",)),
        ]

        def result(_pair_row, evidence_id, *_args):
            return _audit(
                "ordinary_linear_standard"
                if evidence_id == "safe"
                else "known_disulfide"
            )

        with patch(
            "phase3.drugclip.audit_full_split_full_heavy_eligibility."
            "_structure_audit",
            side_effect=result,
        ):
            registry = build_sequence_eligibility_registry(
                pairs,
                evidence={},
                mmcif_root=None,
                qbiolip_root=None,
                biolip_root=None,
                torsion_groups=_torsion_groups(["AAAA"]),
                progress_every=0,
            )
        self.assertEqual(len(registry), 1)
        self.assertEqual(
            registry[0]["chemistry_classification"], "known_disulfide"
        )
        self.assertFalse(registry[0]["eligible"])
        self.assertEqual(registry[0]["pair_count"], 2)

    def test_atom_count_at_cap_is_rejected(self) -> None:
        sequence = "WASLWNWFDITNWLWYIRKK"
        pairs = [_pair("p1", sequence)]
        with patch(
            "phase3.drugclip.audit_full_split_full_heavy_eligibility."
            "_structure_audit",
            return_value=_audit("ordinary_linear_standard"),
        ):
            registry = build_sequence_eligibility_registry(
                pairs,
                evidence={},
                mmcif_root=None,
                qbiolip_root=None,
                biolip_root=None,
                torsion_groups=_torsion_groups([sequence]),
                progress_every=0,
            )
        self.assertGreaterEqual(
            registry[0]["theoretical_heavy_atom_count"],
            ATOM_CAP_EXCLUSIVE,
        )
        self.assertFalse(registry[0]["eligible"])

    def test_classification_counts_pairs_sequences_relations_and_sources(
        self,
    ) -> None:
        pairs = [
            _pair("p1", "AAAA", relation="r1", source="BioLiP2"),
            _pair("p2", "AAAA", relation="r2", source="Q-BioLiP"),
            _pair("p3", "GGGG", relation="r3", source="BioLiP2"),
        ]
        registry = {
            "AAAA": {
                "chemistry_classification": "ordinary_linear_standard",
                "theoretical_heavy_atom_count": 20,
                "eligible": True,
            },
            "GGGG": {
                "chemistry_classification": "chemistry_insufficient",
                "theoretical_heavy_atom_count": 16,
                "eligible": False,
            },
        }
        report = _classification_report(pairs, registry)
        ordinary = report["ordinary_linear_standard"]
        self.assertEqual(ordinary["pair_count"], 2)
        self.assertEqual(ordinary["unique_sequence_count"], 1)
        self.assertEqual(ordinary["biological_relation_count"], 2)
        self.assertEqual(
            ordinary["primary_source_database_pair_counts"],
            {"BioLiP2": 1, "Q-BioLiP": 1},
        )

    def test_coverage_is_pair_weighted_and_split_specific(self) -> None:
        pairs = [
            _pair("t1", "AAAA"),
            _pair("t2", "AAAA"),
            _pair("t3", "GGGG"),
        ]
        registry = {
            "AAAA": {
                "chemistry_classification": "ordinary_linear_standard",
                "theoretical_heavy_atom_count": 20,
                "torsion_prior_covered": True,
                "eligible": True,
            },
            "GGGG": {
                "chemistry_classification": "chemistry_insufficient",
                "theoretical_heavy_atom_count": 16,
                "torsion_prior_covered": True,
                "eligible": False,
            },
        }
        split = _split_eligible_report(pairs, registry, "train")
        coverage = _coverage_scenarios(pairs, registry)
        self.assertEqual(split["pair_count"], 2)
        self.assertAlmostEqual(split["pair_coverage_fraction"], 2 / 3)
        self.assertAlmostEqual(
            coverage["immediately_supported"]["fraction"], 2 / 3
        )
        self.assertAlmostEqual(
            coverage["still_blocked_by_insufficient_metadata"]["fraction"],
            1 / 3,
        )


if __name__ == "__main__":
    unittest.main()
