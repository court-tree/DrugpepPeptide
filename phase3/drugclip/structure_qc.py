"""Experimental-complex QC and receptor-interface extraction.

This module never returns or stores bound-peptide training coordinates. The
bound peptide is read transiently only to verify the relation and define the
receptor interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import gemmi
import numpy as np
from scipy.spatial import cKDTree


AA1_TO_3 = {"A":"ALA", "R":"ARG", "N":"ASN", "D":"ASP", "C":"CYS", "Q":"GLN", "E":"GLU", "G":"GLY", "H":"HIS", "I":"ILE", "L":"LEU", "K":"LYS", "M":"MET", "F":"PHE", "P":"PRO", "S":"SER", "T":"THR", "W":"TRP", "Y":"TYR", "V":"VAL"}
AA3_TO_1 = {value: key for key, value in AA1_TO_3.items()}
BACKBONE_NAMES = {"N", "CA", "C"}


@dataclass(frozen=True)
class ParsedResidue:
    chain_id: str
    residue_id: str
    residue_name: str
    atoms: tuple[dict[str, Any], ...]


def _existing_path(row: dict[str, Any], qbiolip_root: Path, biolip_root: Path, field: str) -> Path:
    raw = str(row.get(field) or "")
    if raw and Path(raw).is_file():
        return Path(raw)
    basename = Path(raw.replace("\\", "/")).name
    if str(row.get("source_database") or "") == "Q-BioLiP_PIII":
        folder = "nonredund_rec" if field == "receptor_structure_file" else "nonredund_lig"
        return qbiolip_root / "extracted" / folder / basename
    return biolip_root / (basename or f"{row.get('pdb_id', '')}.pdb")


@lru_cache(maxsize=8)
def _parse_all(path_text: str) -> tuple[ParsedResidue, ...]:
    structure = gemmi.read_structure(path_text)
    if len(structure) == 0:
        return ()
    output: list[ParsedResidue] = []
    for chain in structure[0]:
        chain_id = str(chain.name)
        for residue in chain:
            letter = str(gemmi.find_tabulated_residue(residue.name.strip().upper()).one_letter_code).upper()
            residue_name = AA1_TO_3.get(letter)
            if residue_name is None:
                continue
            residue_id = str(residue.seqid.num)
            insertion = str(residue.seqid.icode).strip()
            if insertion and insertion != "?":
                residue_id += insertion
            atoms = []
            seen = set()
            for atom in residue:
                atom_name = str(atom.name).strip().upper()
                altloc = str(atom.altloc).strip().upper()
                if atom_name in seen or altloc not in {"", "A", "\x00"}:
                    continue
                element = str(atom.element.name).strip().upper()
                if element == "H":
                    continue
                seen.add(atom_name)
                atoms.append({"atom_name": atom_name, "residue_name": residue_name, "element": element or atom_name[:1], "x": float(atom.pos.x), "y": float(atom.pos.y), "z": float(atom.pos.z), "residue_id": f"{chain_id}:{residue_id}"})
            if atoms:
                output.append(ParsedResidue(chain_id, residue_id, residue_name, tuple(atoms)))
    return tuple(output)


def _read_chain(path: Path, chain_id: str | None) -> list[ParsedResidue]:
    residues = _parse_all(str(path.resolve()))
    return [row for row in residues if not chain_id or row.chain_id == chain_id]


def has_complete_backbone(residue: ParsedResidue) -> bool:
    """Return whether a receptor residue supplies the model's N/CA/C backbone."""

    return BACKBONE_NAMES <= {str(atom["atom_name"]) for atom in residue.atoms}


def coordinate_qc(
    row: dict[str, Any],
    relation_id: str,
    qbiolip_root: Path,
    biolip_root: Path,
    contact_cutoff: float = 6.0,
    min_interface_residues: int = 2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    separate = str(row.get("source_database") or "") == "Q-BioLiP_PIII"
    receptor_field = "receptor_structure_file" if separate else "complex_structure_file"
    peptide_field = "peptide_structure_file" if separate else "complex_structure_file"
    receptor_path = _existing_path(row, qbiolip_root, biolip_root, receptor_field)
    peptide_path = _existing_path(row, qbiolip_root, biolip_root, peptide_field)
    if not receptor_path.is_file() or not peptide_path.is_file():
        raise ValueError("missing_coordinate_file")

    receptor_chain = str(row.get("receptor_chain_id") or "")
    peptide_chain = None if separate else str(row.get("peptide_chain_id") or "")
    receptor_residues = _read_chain(receptor_path, receptor_chain or None)
    peptide_residues = _read_chain(peptide_path, peptide_chain)
    if not receptor_residues:
        raise ValueError("missing_receptor_chain")
    if not peptide_residues:
        raise ValueError("missing_peptide_chain")

    observed_receptor = "".join(AA3_TO_1[item.residue_name] for item in receptor_residues)
    if observed_receptor != str(row["receptor_sequence"]).upper():
        raise ValueError("receptor_coordinate_sequence_mismatch")
    complete_peptide = [residue for residue in peptide_residues if BACKBONE_NAMES <= {atom["atom_name"] for atom in residue.atoms}]
    observed_peptide = "".join(AA3_TO_1[item.residue_name] for item in complete_peptide)
    if observed_peptide != str(row["peptide_sequence"]).upper():
        raise ValueError(f"peptide_coordinate_sequence_mismatch:{observed_peptide}")

    peptide_atoms = [atom for residue in peptide_residues for atom in residue.atoms]
    tree = cKDTree(np.asarray([[atom["x"], atom["y"], atom["z"]] for atom in peptide_atoms], dtype=np.float64))
    selected = []
    contacted_peptide_residues: set[str] = set()
    atom_contacts = 0
    for residue in receptor_residues:
        # Incomplete receptor residues are not valid 3D nodes. Remove them
        # before contact selection instead of silently exporting partial input.
        if not has_complete_backbone(residue):
            continue
        coords = np.asarray([[atom["x"], atom["y"], atom["z"]] for atom in residue.atoms], dtype=np.float64)
        neighbor_lists = tree.query_ball_point(coords, contact_cutoff)
        indices = {index for neighbors in neighbor_lists for index in neighbors}
        if indices:
            selected.append(residue)
            atom_contacts += sum(len(neighbors) for neighbors in neighbor_lists)
            contacted_peptide_residues.update(str(peptide_atoms[index]["residue_id"]) for index in indices)
    if len(selected) < min_interface_residues:
        raise ValueError("insufficient_interface_residues")
    if len(contacted_peptide_residues) < 2:
        raise ValueError("insufficient_contacted_peptide_residues")

    interface = {
        "pair_id": relation_id,
        "receptor_interface_key": f"{row.get('pdb_id')}:{receptor_chain}:contact_{contact_cutoff:g}A",
        "receptor_patch_sequence": "".join(AA3_TO_1[item.residue_name] for item in selected),
        "receptor_atoms": [atom for residue in selected for atom in residue.atoms],
        "source_pdb_id": str(row.get("pdb_id") or ""),
        "source_chain_id": receptor_chain,
        "evidence": str(row.get("evidence_id") or ""),
        "interface_residue_count": len(selected),
        "contacted_peptide_residue_count": len(contacted_peptide_residues),
        "atom_contact_count": atom_contacts,
    }
    metrics = {"evidence_id": str(row.get("evidence_id") or ""), "pdb_id": str(row.get("pdb_id") or ""), "interface_residues": len(selected), "contacted_peptide_residues": len(contacted_peptide_residues), "atom_contacts": atom_contacts}
    return interface, metrics
