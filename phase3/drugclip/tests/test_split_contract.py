from __future__ import annotations

import unittest

from phase3.drugclip.split_and_audit import build_argparser, build_pair_components


class SplitContractTests(unittest.TestCase):
    def test_cli_requires_real_pairs(self) -> None:
        with self.assertRaises(SystemExit):
            build_argparser().parse_args(
                ["--mmseqs", "mmseqs", "--output_dir", "output"]
            )

    def test_cli_requires_mmseqs(self) -> None:
        with self.assertRaises(SystemExit):
            build_argparser().parse_args(
                ["--real_pairs", "pairs.jsonl", "--output_dir", "output"]
            )

    def test_similar_nonidentical_peptides_do_not_create_edges(self) -> None:
        rows = [
            {"pair_id": "p1", "biological_receptor_id": "r1", "peptide_sequence": "AAAAAAAAAA", "structure_pdb_ids": ["1abc"]},
            {"pair_id": "p2", "biological_receptor_id": "r2", "peptide_sequence": "AAAAAAAACC", "structure_pdb_ids": ["2xyz"]},
        ]
        components, summary = build_pair_components(rows, {"r1": "f1", "r2": "f2"})
        self.assertEqual(
            {frozenset(group) for group in components},
            {frozenset({"p1"}), frozenset({"p2"})},
        )
        self.assertNotIn("similar_peptide_edges", summary)

    def test_exact_peptide_still_creates_split_edge(self) -> None:
        rows = [
            {"pair_id": "p1", "biological_receptor_id": "r1", "peptide_sequence": "ACDE", "structure_pdb_ids": ["1abc"]},
            {"pair_id": "p2", "biological_receptor_id": "r2", "peptide_sequence": "ACDE", "structure_pdb_ids": ["2xyz"]},
        ]
        components, _ = build_pair_components(rows, {"r1": "f1", "r2": "f2"})
        self.assertEqual(components, [{"p1", "p2"}])


if __name__ == "__main__":
    unittest.main()
