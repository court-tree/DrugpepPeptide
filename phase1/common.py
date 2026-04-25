from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import gemmi
import numpy as np
from scipy.spatial import cKDTree


STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "MSE", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
}

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "MSE": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y",
    "VAL": "V",
}


@dataclass
class ChainResidue:
    residue: gemmi.Residue
    coords: np.ndarray
    seq_index: int


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield json.loads(s)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def is_protein_residue(residue_or_name: Any) -> bool:
    if isinstance(residue_or_name, gemmi.Residue):
        name = residue_or_name.name
    else:
        name = str(residue_or_name)
    return name.strip().upper() in STANDARD_AA


def load_structure(cif_path: Path) -> gemmi.Structure:
    st = gemmi.read_structure(str(cif_path))
    st.setup_entities()
    return st


def get_model(structure: gemmi.Structure) -> gemmi.Model:
    if len(structure) == 0:
        raise ValueError(f"Empty structure: {structure.name}")
    return structure[0]


def find_chain_by_name(model: gemmi.Model, chain_name: str) -> gemmi.Chain:
    target = str(chain_name).strip()
    for chain in model:
        if str(chain.name).strip() == target:
            return chain
    available = [str(chain.name) for chain in model]
    raise KeyError(f"Chain not found: {chain_name}; available_chains={available}")


def heavy_atoms(residue: gemmi.Residue) -> List[gemmi.Atom]:
    return [atom for atom in residue if atom.element.name != "H"]


def residue_heavy_atom_coords(residue: gemmi.Residue) -> np.ndarray:
    coords = [[atom.pos.x, atom.pos.y, atom.pos.z] for atom in heavy_atoms(residue)]
    if not coords:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(coords, dtype=np.float32)


def residue_seqid_num(residue: gemmi.Residue) -> int:
    if residue.seqid.num is None:
        raise ValueError(f"Residue missing seqid.num: {residue.name}")
    return int(residue.seqid.num)


def residue_seqid_icode(residue: gemmi.Residue) -> str:
    return str(residue.seqid.icode).strip()


def residue_id_string(residue: gemmi.Residue) -> str:
    num = residue_seqid_num(residue)
    icode = residue_seqid_icode(residue)
    if icode:
        return f"{num}{icode}:{residue.name}"
    return f"{num}:{residue.name}"


def chain_residues(chain: gemmi.Chain) -> List[ChainResidue]:
    out: List[ChainResidue] = []
    for residue in chain:
        if not is_protein_residue(residue):
            continue
        out.append(
            ChainResidue(
                residue=residue,
                coords=residue_heavy_atom_coords(residue),
                seq_index=len(out),
            )
        )
    return out


def chain_sequence(residues: Sequence[ChainResidue]) -> str:
    return "".join(AA3_TO_1.get(item.residue.name.strip().upper(), "X") for item in residues)


def build_receptor_tree(
    residues: Sequence[ChainResidue],
) -> Tuple[Optional[cKDTree], np.ndarray]:
    coords_blocks: List[np.ndarray] = []
    atom_to_res_idx: List[int] = []
    for res_idx, item in enumerate(residues):
        if len(item.coords) == 0:
            continue
        coords_blocks.append(item.coords)
        atom_to_res_idx.extend([res_idx] * len(item.coords))
    if not coords_blocks:
        return None, np.zeros((0,), dtype=np.int32)
    all_coords = np.concatenate(coords_blocks, axis=0)
    return cKDTree(all_coords), np.asarray(atom_to_res_idx, dtype=np.int32)


def get_atom_by_name(residue: gemmi.Residue, atom_name: str) -> Optional[gemmi.Atom]:
    target = atom_name.strip()
    for atom in residue:
        if atom.name.strip() == target:
            return atom
    return None


def peptide_bond_continuous(
    left_res: gemmi.Residue,
    right_res: gemmi.Residue,
    min_dist: float = 1.2,
    max_dist: float = 1.6,
) -> bool:
    c_atom = get_atom_by_name(left_res, "C")
    n_atom = get_atom_by_name(right_res, "N")
    if c_atom is None or n_atom is None:
        return False
    dx = c_atom.pos.x - n_atom.pos.x
    dy = c_atom.pos.y - n_atom.pos.y
    dz = c_atom.pos.z - n_atom.pos.z
    dist = float((dx * dx + dy * dy + dz * dz) ** 0.5)
    return min_dist <= dist <= max_dist


def window_is_continuous(residues: Sequence[ChainResidue]) -> bool:
    if len(residues) <= 1:
        return True
    for i in range(len(residues) - 1):
        if not peptide_bond_continuous(residues[i].residue, residues[i + 1].residue):
            return False
    return True


def window_bounds_match_candidate(
    candidate: Dict[str, Any],
    peptide_window: Sequence[ChainResidue],
) -> bool:
    if not peptide_window:
        return False
    start_res = peptide_window[0].residue
    end_res = peptide_window[-1].residue
    return (
        residue_seqid_num(start_res) == int(candidate["peptide_start_resseq"])
        and residue_seqid_icode(start_res) == str(candidate["peptide_start_icode"])
        and residue_seqid_num(end_res) == int(candidate["peptide_end_resseq"])
        and residue_seqid_icode(end_res) == str(candidate["peptide_end_icode"])
    )


def compute_window_contact_stats(
    peptide_window: Sequence[ChainResidue],
    receptor_tree: cKDTree,
    atom_to_res_idx: np.ndarray,
    contact_cutoff: float,
) -> Dict[str, Any]:
    per_residue_contacts: List[int] = []
    contact_mask: List[bool] = []
    total_contacts = 0
    for item in peptide_window:
        if len(item.coords) == 0:
            per_residue_contacts.append(0)
            contact_mask.append(False)
            continue
        neighbors = receptor_tree.query_ball_point(item.coords, r=contact_cutoff)
        touched_residues = {
            int(atom_to_res_idx[atom_idx])
            for atom_hits in neighbors
            for atom_idx in atom_hits
        }
        n_contacts = int(len(touched_residues))
        per_residue_contacts.append(n_contacts)
        contact_mask.append(n_contacts > 0)
        total_contacts += n_contacts

    window_len = len(peptide_window)
    avg_contact_count = (total_contacts / window_len) if window_len > 0 else 0.0
    contact_coverage = (sum(1 for x in contact_mask if x) / window_len) if window_len > 0 else 0.0

    longest_contact_run = 0
    current_run = 0
    for flag in contact_mask:
        if flag:
            current_run += 1
            longest_contact_run = max(longest_contact_run, current_run)
        else:
            current_run = 0

    return {
        "total_contact_count": int(total_contacts),
        "avg_contact_count": float(avg_contact_count),
        "contact_coverage": float(contact_coverage),
        "per_residue_contacts": per_residue_contacts,
        "longest_contact_run": int(longest_contact_run),
    }


def interval_iou(a_l: int, a_r: int, b_l: int, b_r: int) -> float:
    inter = max(0, min(a_r, b_r) - max(a_l, b_l) + 1)
    if inter <= 0:
        return 0.0
    union = (a_r - a_l + 1) + (b_r - b_l + 1) - inter
    if union <= 0:
        return 0.0
    return inter / union


def interval_contains(a_l: int, a_r: int, b_l: int, b_r: int) -> bool:
    return a_l <= b_l and a_r >= b_r


def sequence_identity_same_length(seq_a: str, seq_b: str) -> float:
    if not seq_a or not seq_b:
        return 0.0
    n = min(len(seq_a), len(seq_b))
    if n == 0:
        return 0.0
    matches = sum(1 for i in range(n) if seq_a[i] == seq_b[i])
    return matches / max(len(seq_a), len(seq_b))
