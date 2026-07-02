from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gemmi

from phase2.pepclip.data import PepCLIP3DDataset, PepCLIPDataset, collate_pepclip_3d
from phase3.v1.builder import (
    BuildConfig,
    ResidueItem,
    assign_receptor_family_keys,
    assign_splits,
    build_dataset,
    load_complex_model_with_assembly,
    validate_peptide,
    validate_peptide_continuity,
)


AA_NAMES = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE"]


def add_atom(residue: gemmi.Residue, name: str, element: str, x: float, y: float, z: float) -> None:
    atom = gemmi.Atom()
    atom.name = name
    atom.element = gemmi.Element(element)
    atom.pos = gemmi.Position(x, y, z)
    residue.add_atom(atom)


def make_residue(name: str, seqid: int, x_offset: float) -> gemmi.Residue:
    residue = gemmi.Residue()
    residue.name = name
    residue.seqid = gemmi.SeqId(str(seqid))
    add_atom(residue, "N", "N", x_offset, 0.0, 0.0)
    add_atom(residue, "CA", "C", x_offset + 0.4, 0.0, 0.0)
    add_atom(residue, "C", "C", x_offset + 0.8, 0.0, 0.0)
    add_atom(residue, "O", "O", x_offset + 1.0, 0.2, 0.0)
    add_atom(residue, "CB", "C", x_offset + 0.4, 0.8, 0.0)
    return residue


def write_test_complex(path: Path, peptide_offset: float = 0.2) -> None:
    structure = gemmi.Structure()
    structure.name = "phase3_v1_test"
    model = gemmi.Model("1")
    receptor = gemmi.Chain("A")
    peptide = gemmi.Chain("B")
    for i, name in enumerate(["LYS", "ARG", "TYR", "SER", "ASP"], start=1):
        receptor.add_residue(make_residue(name, i, float(i - 1)))
    for i, name in enumerate(AA_NAMES, start=1):
        peptide.add_residue(make_residue(name, i, float(i - 1) + peptide_offset))
    model.add_chain(receptor)
    model.add_chain(peptide)
    structure.add_model(model)
    structure.write_pdb(str(path))


class Phase3V1BuilderTest(unittest.TestCase):
    def test_peptide_validation_rejects_nonstandard_and_gaps(self) -> None:
        residues = [
            ResidueItem("B", make_residue(name, i, float(i)), i - 1)
            for i, name in enumerate(["ALA", "CYS", "ASP", "MSE", "PHE", "GLY", "HIS", "ILE"], start=1)
        ]
        self.assertEqual(validate_peptide(residues, 8, 20), (False, "noncanonical_sequence"))

        gapped = [
            ResidueItem("B", make_residue("ALA", number, float(index)), index)
            for index, number in enumerate([1, 2, 3, 5, 6, 7, 8, 9])
        ]
        self.assertEqual(
            validate_peptide_continuity(gapped, 1, 9),
            (False, "non_contiguous_peptide_segment"),
        )

    def test_build_dataset_exports_v1_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structure_path = root / "complex.pdb"
            write_test_complex(structure_path)
            input_path = root / "records.jsonl"
            records = [
                {
                    "source_database": "BioLiP_peptide",
                    "source_entry_id": "entry1",
                    "pdb_id": "test1",
                    "biological_assembly_id": "biolip_source_pdb",
                    "assembly_confidence": "biolip_binding_site_record",
                    "complex_structure_file": "complex.pdb",
                    "receptor_chain_id": "A",
                    "peptide_chain_id": "B",
                    "peptide_residue_start": 1,
                    "peptide_residue_end": 8,
                },
                {
                    "source_database": "BioLiP_peptide",
                    "source_entry_id": "entry1_duplicate",
                    "pdb_id": "test1",
                    "biological_assembly_id": "biolip_source_pdb",
                    "assembly_confidence": "biolip_binding_site_record",
                    "complex_structure_file": "complex.pdb",
                    "receptor_chain_id": "A",
                    "peptide_chain_id": "B",
                    "peptide_residue_start": 1,
                    "peptide_residue_end": 8,
                },
            ]
            input_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
            output_dir = root / "out"
            summary = build_dataset(
                BuildConfig(
                    input_jsonl=input_path,
                    structure_root=root,
                    output_dir=output_dir,
                    split_mode="pair_level",
                    progress_every=0,
                )
            )
            self.assertEqual(summary["final_anchor_count"], 1)
            self.assertEqual(summary["reject_counts"], {})
            self.assertTrue((output_dir / "receptor_peptide_anchor.csv").exists())
            self.assertTrue((output_dir / "positive_strong_bound_edges.jsonl").exists())
            self.assertTrue((output_dir / "track_a_train.jsonl").exists())
            self.assertTrue((output_dir / "track_b_train.jsonl").exists())
            self.assertTrue((output_dir / "manual_structure_audit" / "manual_structure_audit_samples.jsonl").exists())
            audit = json.loads((output_dir / "dataset_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["dropped_duplicates"], 1)
            self.assertEqual(audit["manual_structure_audit"]["sample_count_written"], 1)
            manual_sample = json.loads(
                (output_dir / "manual_structure_audit" / "manual_structure_audit_samples.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertTrue(Path(manual_sample["pymol_script"]).exists())
            self.assertIn("human_decision", manual_sample)
            self.assertIn("distance contacts5A", Path(manual_sample["pymol_script"]).read_text(encoding="utf-8"))
            edge = json.loads((output_dir / "positive_strong_bound_edges.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual(edge["edge_type"], "positive_strong_bound")
            conformer = json.loads((output_dir / "peptide_true_bound_conformer.jsonl").read_text(encoding="utf-8").strip())
            self.assertTrue(conformer["is_true_bound"])
            track_b = json.loads((output_dir / "track_b_train.jsonl").read_text(encoding="utf-8").strip())
            patch = json.loads(Path(track_b["receptor_patch_coords_path"]).read_text(encoding="utf-8"))
            self.assertGreater(track_b["receptor_context_atom_count"], 0)
            self.assertGreater(track_b["receptor_interface_atom_count"], 0)
            self.assertGreater(track_b["receptor_peptide_contact_pair_count_5A"], 0)
            self.assertIn("receptor_context_atoms", patch)
            self.assertIn("receptor_context_backbone_atoms", patch)
            self.assertIn("receptor_context_residues", patch)
            self.assertIn("receptor_peptide_contact_pairs_5A", patch)
            self.assertEqual(patch["receptor_context_atoms"][0]["atom_index"], 0)
            track_a_dataset = PepCLIPDataset(output_dir / "track_a_train.jsonl")
            self.assertEqual(len(track_a_dataset), 1)
            self.assertGreater(len(track_a_dataset[0]["receptor_sequence"]), 0)
            track_b_dataset = PepCLIP3DDataset(output_dir / "track_b_train.jsonl")
            item = track_b_dataset[0]
            self.assertGreater(item["num_receptor_atoms"], 0)
            self.assertGreater(item["num_peptide_atoms"], 0)
            batch = collate_pepclip_3d([item])
            self.assertEqual(batch["receptor_coords"].shape[-1], 3)
            self.assertEqual(batch["peptide_coords"].shape[-1], 3)

    def test_rejects_plain_complex_without_biological_assembly_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_test_complex(root / "complex.pdb")
            input_path = root / "records.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "source_database": "BioLiP_peptide",
                        "source_entry_id": "entry1",
                        "pdb_id": "test1",
                        "biological_assembly_id": "1",
                        "complex_structure_file": "complex.pdb",
                        "receptor_chain_id": "A",
                        "peptide_chain_id": "B",
                        "peptide_residue_start": 1,
                        "peptide_residue_end": 8,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            summary = build_dataset(
                BuildConfig(
                    input_jsonl=input_path,
                    structure_root=root,
                    output_dir=root / "out",
                    split_mode="pair_level",
                    progress_every=0,
                )
            )
            self.assertEqual(summary["final_anchor_count"], 0)
            self.assertEqual(summary["reject_counts"], {"no_biological_assembly_metadata": 1})

    def test_transform_to_assembly_path_is_used_for_generic_assembly_record(self) -> None:
        class FakeAssembly:
            name = "1"

        class FakeStructure:
            def __init__(self) -> None:
                self.assemblies = [FakeAssembly()]
                self.transformed = False

            def __len__(self) -> int:
                return 1

            def __getitem__(self, index: int) -> str:
                self.assert_index = index
                return "MODEL_AFTER_ASSEMBLY"

            def transform_to_assembly(self, assembly_name, how) -> None:
                self.transformed = (assembly_name == "1")

        fake_structure = FakeStructure()
        record = {
            "source_database": "BioLiP_peptide",
            "biological_assembly_id": "1",
            "assembly_confidence": "generic_mmcif_assembly",
        }
        with mock.patch("phase3.v1.builder.load_structure", return_value=fake_structure):
            model, info = load_complex_model_with_assembly(
                Path("dummy.cif"),
                record,
                BuildConfig(
                    input_jsonl=Path("in.jsonl"),
                    structure_root=Path("."),
                    output_dir=Path("out"),
                ),
            )
        self.assertEqual(model, "MODEL_AFTER_ASSEMBLY")
        self.assertTrue(fake_structure.transformed)
        self.assertEqual(info["assembly_policy"], "gemmi_transform_to_assembly")
        self.assertEqual(info["assembly_status"], "reconstructed")

    def test_receptor_family_similarity_clustering_is_not_exact_sequence_only(self) -> None:
        anchors = [
            {
                "receptor_sequence_key": "rec_a",
                "receptor_sequence": "ACDEFGHIKLMNPQRSTVWY" * 3,
            },
            {
                "receptor_sequence_key": "rec_b",
                "receptor_sequence": "ACDEFGHIKLMNPQRSTVWY" * 2 + "ACDEYGHIKLMNPQRSTVWY",
            },
            {
                "receptor_sequence_key": "rec_c",
                "receptor_sequence": "YYYYYYYYYYYYYYYYYYYY",
            },
        ]
        summary = assign_receptor_family_keys(
            anchors,
            BuildConfig(
                input_jsonl=Path("in.jsonl"),
                structure_root=Path("."),
                output_dir=Path("out"),
                receptor_family_identity_threshold=0.80,
                receptor_family_min_coverage=0.60,
            ),
        )
        self.assertEqual(summary["receptor_family_method"], "sequence_similarity_greedy")
        self.assertEqual(anchors[0]["receptor_family_key"], anchors[1]["receptor_family_key"])
        self.assertNotEqual(anchors[0]["receptor_family_key"], anchors[2]["receptor_family_key"])
        self.assertEqual(anchors[0]["receptor_family_method"], "sequence_similarity_greedy")

    def test_strict_split_uses_union_of_peptide_receptor_family_and_pdb(self) -> None:
        anchors = [
            {
                "anchor_id": "a1",
                "peptide_sequence_key": "pep1",
                "receptor_family_key": "rfam1",
                "pdb_key": "pdb1",
            },
            {
                "anchor_id": "a2",
                "peptide_sequence_key": "pep1",
                "receptor_family_key": "rfam2",
                "pdb_key": "pdb2",
            },
            {
                "anchor_id": "a3",
                "peptide_sequence_key": "pep3",
                "receptor_family_key": "rfam2",
                "pdb_key": "pdb3",
            },
        ]
        assign_splits(
            anchors,
            BuildConfig(
                input_jsonl=Path("in.jsonl"),
                structure_root=Path("."),
                output_dir=Path("out"),
                split_mode="strict",
                train_fraction=0.34,
                val_fraction=0.33,
            ),
        )
        self.assertEqual({anchor["split"] for anchor in anchors}, {anchors[0]["split"]})
        self.assertEqual(len({anchor["split_group"] for anchor in anchors}), 1)



    def test_strict_split_balances_large_groups_without_empty_validation_when_possible(self) -> None:
        anchors = []
        for idx in range(30):
            anchors.append(
                {
                    "anchor_id": f"big_{idx}",
                    "peptide_sequence_key": f"pep_big_{idx}",
                    "receptor_family_key": "rfam_big",
                    "pdb_key": f"pdb_big_{idx}",
                }
            )
        for idx in range(20):
            anchors.append(
                {
                    "anchor_id": f"small_{idx}",
                    "peptide_sequence_key": f"pep_small_{idx}",
                    "receptor_family_key": f"rfam_small_{idx}",
                    "pdb_key": f"pdb_small_{idx}",
                }
            )
        assign_splits(
            anchors,
            BuildConfig(
                input_jsonl=Path("in.jsonl"),
                structure_root=Path("."),
                output_dir=Path("out"),
                split_mode="strict",
                train_fraction=0.8,
                val_fraction=0.1,
            ),
        )
        split_counts = {split: sum(1 for anchor in anchors if anchor["split"] == split) for split in ("train", "val", "test")}
        self.assertGreater(split_counts["train"], 0)
        self.assertGreater(split_counts["val"], 0)
        self.assertGreater(split_counts["test"], 0)
        self.assertEqual(
            len({anchor["split"] for anchor in anchors if anchor["receptor_family_key"] == "rfam_big"}),
            1,
        )
if __name__ == "__main__":
    unittest.main()


