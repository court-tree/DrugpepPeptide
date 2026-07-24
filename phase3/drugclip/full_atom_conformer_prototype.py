"""Candidate-independent full-heavy peptide conformer prototype.

This module is deliberately separate from the formal random_conformer_v3
contract.  It accepts only a standard, unmodified, linear peptide sequence and
fixed seed.  It never accepts receptor, interface, contact, evidence, or bound
pose inputs.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import time
from typing import Any

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from phase3.drugclip.standard_residue_topology import (
    canonical_geometry_audit,
)


SCHEMA_VERSION = "pepclip-full-heavy-conformer-prototype-v1"
GENERATOR_ID = "rdkit-molfromsequence-etkdgv3-mmff94s-1000iter-prototype-v2"
CHEMISTRY_CLASS = "ordinary_linear_unmodified_standard_peptide"
DEFAULT_BASE_SEED = 20260723
MAX_EMBED_ATTEMPTS = 25
MMFF_MAX_ITERATIONS = 1000

AA1_TO_3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

REQUIRED_HEAVY_ATOMS = {
    "ALA": {"N", "CA", "C", "O", "CB"},
    "ARG": {"N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"},
    "ASN": {"N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"},
    "ASP": {"N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"},
    "CYS": {"N", "CA", "C", "O", "CB", "SG"},
    "GLN": {"N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"},
    "GLU": {"N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"},
    "GLY": {"N", "CA", "C", "O"},
    "HIS": {"N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"},
    "ILE": {"N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"},
    "LEU": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"},
    "LYS": {"N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"},
    "MET": {"N", "CA", "C", "O", "CB", "CG", "SD", "CE"},
    "PHE": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "PRO": {"N", "CA", "C", "O", "CB", "CG", "CD"},
    "SER": {"N", "CA", "C", "O", "CB", "OG"},
    "THR": {"N", "CA", "C", "O", "CB", "OG1", "CG2"},
    "TRP": {
        "N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2",
        "CE3", "CZ2", "CZ3", "CH2",
    },
    "TYR": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"},
    "VAL": {"N", "CA", "C", "O", "CB", "CG1", "CG2"},
}


class UnsupportedPeptideChemistry(ValueError):
    """The sequence/chemistry is outside the safe prototype boundary."""


class ConformerGenerationError(RuntimeError):
    """No conformer satisfying the prototype geometry contract was generated."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


def classify_sequence(
    sequence: str,
    *,
    chemistry_class: str = CHEMISTRY_CLASS,
) -> dict[str, Any]:
    normalized = str(sequence).strip().upper()
    if chemistry_class != CHEMISTRY_CLASS:
        raise UnsupportedPeptideChemistry(
            f"unsupported_chemistry_class:{chemistry_class}; "
            "prototype_accepts_only_ordinary_linear_unmodified_standard_peptide"
        )
    if not normalized:
        raise UnsupportedPeptideChemistry("empty_peptide_sequence")
    invalid = sorted(set(normalized) - set(AA1_TO_3))
    if invalid:
        raise UnsupportedPeptideChemistry(f"nonstandard_or_ambiguous_residues:{invalid}")
    if normalized.count("C") > 1:
        raise UnsupportedPeptideChemistry(
            "multiple_cysteines_require_explicit_disulfide_state"
        )
    return {
        "sequence": normalized,
        "chemistry_class": CHEMISTRY_CLASS,
        "topology": "linear",
        "modifications": [],
        "disulfide_bonds": [],
        "n_terminus": "rdkit_standard_free_amine",
        "c_terminus": "rdkit_standard_free_carboxylic_acid",
    }


def _base_molecule(sequence: str) -> Chem.Mol:
    molecule = Chem.MolFromSequence(sequence)
    if molecule is None:
        raise UnsupportedPeptideChemistry("rdkit_mol_from_sequence_failed")
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return molecule


def _atom_identity(atom: Chem.Atom) -> dict[str, Any]:
    info = atom.GetPDBResidueInfo()
    if info is None:
        raise ConformerGenerationError(f"atom_missing_pdb_residue_info:{atom.GetIdx()}")
    return {
        "atom_index": int(atom.GetIdx()),
        "atom_name": info.GetName().strip().upper(),
        "element": atom.GetSymbol().upper(),
        "residue_index": int(info.GetResidueNumber()),
        "residue_name": info.GetResidueName().strip().upper(),
    }


def _validate_topology(molecule: Chem.Mol, sequence: str) -> list[dict[str, Any]]:
    identities = [_atom_identity(atom) for atom in molecule.GetAtoms()]
    by_residue: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for identity in identities:
        by_residue[identity["residue_index"]].append(identity)
    if sorted(by_residue) != list(range(1, len(sequence) + 1)):
        raise ConformerGenerationError("residue_index_contract_mismatch")

    for residue_index, one_letter in enumerate(sequence, start=1):
        rows = by_residue[residue_index]
        expected_name = AA1_TO_3[one_letter]
        names = {row["atom_name"] for row in rows}
        residue_names = {row["residue_name"] for row in rows}
        if residue_names != {expected_name}:
            raise ConformerGenerationError(
                f"residue_name_contract_mismatch:{residue_index}:{sorted(residue_names)}"
            )
        required = set(REQUIRED_HEAVY_ATOMS[expected_name])
        if residue_index == len(sequence):
            required.add("OXT")
        missing = sorted(required - names)
        if missing:
            raise ConformerGenerationError(
                f"required_heavy_atoms_missing:{residue_index}:{expected_name}:{missing}"
            )
        duplicates = sorted(name for name in names if sum(row["atom_name"] == name for row in rows) != 1)
        if duplicates:
            raise ConformerGenerationError(
                f"duplicate_atom_names:{residue_index}:{duplicates}"
            )
    return identities


def _attempt_seed(
    sequence: str,
    base_seed: int,
    conformer_index: int,
    attempt_index: int,
) -> int:
    material = "|".join([
        GENERATOR_ID,
        CHEMISTRY_CLASS,
        sequence,
        str(int(base_seed)),
        str(int(conformer_index)),
        str(int(attempt_index)),
    ]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big") % 2147483647 or 1


def _coordinates(molecule: Chem.Mol, heavy_atom_count: int) -> list[list[float]]:
    conformer = molecule.GetConformer()
    return [
        [float(value) for value in conformer.GetAtomPosition(index)]
        for index in range(heavy_atom_count)
    ]


def _coordinate_sha256(coordinates: list[list[float]]) -> str:
    canonical = ";".join(
        ",".join(format(value, ".12f") for value in xyz)
        for xyz in coordinates
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest().upper()


def atom_identity_sha256(identities: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        identities,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest().upper()


def _geometry_audit(
    molecule: Chem.Mol,
    coordinates: list[list[float]],
    identities: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate standard-PDB coordinates against the canonical peptide graph."""
    canonical_identities = identities or [
        _atom_identity(molecule.GetAtomWithIdx(index))
        for index in range(len(coordinates))
    ]
    try:
        return canonical_geometry_audit(canonical_identities, coordinates)
    except ValueError as error:
        raise ConformerGenerationError(str(error)) from error


def _generate_one(
    base: Chem.Mol,
    sequence: str,
    base_seed: int,
    conformer_index: int,
) -> dict[str, Any]:
    attempt_records: list[dict[str, Any]] = []
    heavy_atom_count = base.GetNumAtoms()
    for attempt_index in range(MAX_EMBED_ATTEMPTS):
        seed = _attempt_seed(sequence, base_seed, conformer_index, attempt_index)
        molecule = Chem.AddHs(Chem.Mol(base))
        parameters = AllChem.ETKDGv3()
        parameters.randomSeed = seed
        parameters.numThreads = 1
        parameters.useRandomCoords = True
        parameters.clearConfs = True
        embed_started = time.perf_counter()
        embed_status = int(AllChem.EmbedMolecule(molecule, parameters))
        embed_seconds = time.perf_counter() - embed_started
        attempt_record: dict[str, Any] = {
            "attempt_index": attempt_index,
            "random_seed": seed,
            "embedding_seconds": embed_seconds,
            "embedding_status": embed_status,
            "mmff_seconds": None,
            "mmff_return_status": None,
            "rejection_reason": None,
        }
        if embed_status != 0:
            attempt_record["rejection_reason"] = "embedding_failed"
            attempt_records.append(attempt_record)
            continue
        if not AllChem.MMFFHasAllMoleculeParams(molecule):
            raise ConformerGenerationError("mmff_parameters_incomplete")
        mmff_started = time.perf_counter()
        status = int(
            AllChem.MMFFOptimizeMolecule(
                molecule,
                mmffVariant="MMFF94s",
                maxIters=MMFF_MAX_ITERATIONS,
                confId=0,
            )
        )
        mmff_seconds = time.perf_counter() - mmff_started
        attempt_record["mmff_seconds"] = mmff_seconds
        attempt_record["mmff_return_status"] = status
        if status != 0:
            attempt_record["rejection_reason"] = f"mmff_not_converged:{status}"
            attempt_records.append(attempt_record)
            continue
        coordinates = _coordinates(molecule, heavy_atom_count)
        try:
            geometry = _geometry_audit(molecule, coordinates)
        except ConformerGenerationError as error:
            attempt_record["rejection_reason"] = f"geometry_rejected:{error}"
            attempt_records.append(attempt_record)
            continue
        attempt_records.append(attempt_record)
        properties = AllChem.MMFFGetMoleculeProperties(molecule, mmffVariant="MMFF94s")
        force_field = AllChem.MMFFGetMoleculeForceField(molecule, properties, confId=0)
        return {
            "conformer_index": conformer_index,
            "attempt_index": attempt_index,
            "random_seed": seed,
            "embedding_seconds": embed_seconds,
            "mmff_seconds": mmff_seconds,
            "mmff_variant": "MMFF94s",
            "mmff_status": status,
            "mmff_energy": float(force_field.CalcEnergy()),
            "coordinate_sha256": _coordinate_sha256(coordinates),
            "geometry_audit": geometry,
            "coordinates": coordinates,
            "attempt_records": attempt_records,
        }
    raise ConformerGenerationError(
        f"conformer_attempts_exhausted:index={conformer_index}:"
        + "|".join(
            f"attempt={row['attempt_index']}:{row['rejection_reason']}"
            for row in attempt_records
        ),
        details={
            "failed_conformer_index": conformer_index,
            "attempt_records": attempt_records,
            "maximum_attempts": MAX_EMBED_ATTEMPTS,
            "mmff_max_iterations": MMFF_MAX_ITERATIONS,
        },
    )


def generate_full_heavy_conformers(
    sequence: str,
    *,
    num_conformers: int = 10,
    base_seed: int = DEFAULT_BASE_SEED,
    chemistry_class: str = CHEMISTRY_CLASS,
) -> dict[str, Any]:
    """Generate deterministic, candidate-independent full-heavy conformers."""
    if not 1 <= int(num_conformers) <= 10:
        raise ValueError("num_conformers_must_be_between_1_and_10")
    chemistry = classify_sequence(sequence, chemistry_class=chemistry_class)
    normalized = chemistry["sequence"]
    base = _base_molecule(normalized)
    identities = _validate_topology(base, normalized)

    conformers: list[dict[str, Any]] = []
    coordinate_hashes: set[str] = set()
    for conformer_index in range(int(num_conformers)):
        try:
            conformer = _generate_one(
                base, normalized, int(base_seed), conformer_index
            )
        except ConformerGenerationError as error:
            error.details.update({
                "accepted_conformer_count": len(conformers),
                "accepted_conformer_coordinate_sha256": [
                    row["coordinate_sha256"] for row in conformers
                ],
                "atom_identity_sha256": atom_identity_sha256(identities),
            })
            raise
        if conformer["coordinate_sha256"] in coordinate_hashes:
            raise ConformerGenerationError(
                f"duplicate_conformer_coordinates:index={conformer_index}"
            )
        coordinate_hashes.add(conformer["coordinate_sha256"])
        conformers.append(conformer)

    return {
        "schema_version": SCHEMA_VERSION,
        "generator_id": GENERATOR_ID,
        "generator_version": {
            "rdkit": rdBase.rdkitVersion,
            "etkdg": "v3",
            "force_field": "MMFF94s",
            "mmff_max_iterations": MMFF_MAX_ITERATIONS,
            "maximum_embed_attempts_per_conformer": MAX_EMBED_ATTEMPTS,
        },
        "peptide_sequence": normalized,
        "chemistry": chemistry,
        "base_seed": int(base_seed),
        "atom_count": len(identities),
        "atom_identity": identities,
        "atom_identity_sha256": atom_identity_sha256(identities),
        "conformer_count": len(conformers),
        "conformers": conformers,
        "dependency_contract": {
            "allowed_inputs": ["peptide_sequence", "peptide_chemistry", "fixed_seed", "generator_version"],
            "target_bound_inputs_used": False,
        },
    }


def conformer_atoms(payload: dict[str, Any], conformer_index: int) -> list[dict[str, Any]]:
    """Return one conformer in the atom dictionary format consumed by PepCLIP."""
    identities = payload["atom_identity"]
    conformer = payload["conformers"][int(conformer_index)]
    coordinates = conformer["coordinates"]
    if len(identities) != len(coordinates):
        raise ValueError("atom_identity_coordinate_count_mismatch")
    return [
        {
            **identity,
            "x": float(xyz[0]),
            "y": float(xyz[1]),
            "z": float(xyz[2]),
        }
        for identity, xyz in zip(identities, coordinates)
    ]
