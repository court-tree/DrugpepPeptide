from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import freesasa
import gemmi
import numpy as np
from scipy.spatial import cKDTree


STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "MSE", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL"
}

VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "P": 1.80,
    "SE": 1.90,
}


class Step4Config:
    def __init__(
        self,
        contact_cutoff_6a: float = 6.0,
        pocket_cutoff_6a: float = 6.0,
        min_contact_residues_6a: int = 2,
        min_contact_coverage_for_rbsa: float = 0.25,
        min_pocket_size_for_rbsa: int = 2,
        min_rbsa_raw: float = 0.05,
        debug: bool = False,
    ) -> None:
        self.contact_cutoff_6a = contact_cutoff_6a
        self.pocket_cutoff_6a = pocket_cutoff_6a
        self.min_contact_residues_6a = min_contact_residues_6a
        self.min_contact_coverage_for_rbsa = min_contact_coverage_for_rbsa
        self.min_pocket_size_for_rbsa = min_pocket_size_for_rbsa
        self.min_rbsa_raw = min_rbsa_raw
        self.debug = debug


_WORKER_CFG: Optional[Step4Config] = None
_WORKER_CHAIN_CACHE: Dict[Tuple[str, str, str], Dict[str, Any]] = {}


def is_protein_residue(resname: str) -> bool:
    return resname in STANDARD_AA


def load_structure(cif_path: Path) -> gemmi.Structure:
    st = gemmi.read_structure(str(cif_path))
    st.setup_entities()
    return st


def get_model(structure: gemmi.Structure) -> gemmi.Model:
    if len(structure) == 0:
        raise ValueError("Empty structure: no models found")
    return structure[0]


def find_chain_by_name(model: gemmi.Model, chain_name: str) -> gemmi.Chain:
    available = [chain.name for chain in model]
    for chain in model:
        if chain.name == chain_name:
            return chain
    target = str(chain_name).strip()
    for chain in model:
        if str(chain.name).strip() == target:
            return chain
    raise KeyError(f"Chain not found: {chain_name}; available_chains={available}")


def residue_heavy_atoms(residue: gemmi.Residue) -> List[gemmi.Atom]:
    return [atom for atom in residue if atom.element.name != "H"]


def residue_heavy_atom_coords(residue: gemmi.Residue) -> np.ndarray:
    coords = []
    for atom in residue_heavy_atoms(residue):
        coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
    if not coords:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(coords, dtype=float)


def chain_residues_with_coords_keep_alignment(chain: gemmi.Chain) -> List[Tuple[gemmi.Residue, np.ndarray]]:
    out: List[Tuple[gemmi.Residue, np.ndarray]] = []
    for residue in chain:
        if not is_protein_residue(residue.name):
            continue
        coords = residue_heavy_atom_coords(residue)
        out.append((residue, coords))
    return out


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


def atom_element(atom: gemmi.Atom) -> str:
    e = atom.element.name.strip().upper()
    if not e:
        e = atom.name.strip()[:1].upper()
    return e


def atom_radius(atom: gemmi.Atom) -> float:
    return VDW_RADII.get(atom_element(atom), 1.70)


def heavy_atom_coords_from_residues(residues: List[gemmi.Residue]) -> np.ndarray:
    coords = []
    for residue in residues:
        for atom in residue_heavy_atoms(residue):
            coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
    if not coords:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(coords, dtype=float)


def build_receptor_kdtree_from_residues(
    rec_residues: List[Tuple[gemmi.Residue, np.ndarray]]
) -> Tuple[Optional[cKDTree], np.ndarray, np.ndarray]:
    coords_list: List[np.ndarray] = []
    atom_to_res_idx: List[int] = []
    for idx, (_, coords) in enumerate(rec_residues):
        if len(coords) > 0:
            coords_list.append(coords)
            atom_to_res_idx.extend([idx] * len(coords))
    if not coords_list:
        return None, np.zeros((0,), dtype=int), np.zeros((0, 3), dtype=float)
    all_coords = np.concatenate(coords_list, axis=0)
    return cKDTree(all_coords), np.asarray(atom_to_res_idx, dtype=int), all_coords


def validate_candidate_window(cand: Dict[str, Any], peptide_residues: List[gemmi.Residue]) -> None:
    if not peptide_residues:
        raise ValueError("Empty peptide residue window after slicing")
    start_res = peptide_residues[0]
    end_res = peptide_residues[-1]
    start_num = residue_seqid_num(start_res)
    start_icode = residue_seqid_icode(start_res)
    end_num = residue_seqid_num(end_res)
    end_icode = residue_seqid_icode(end_res)

    if start_num != int(cand["peptide_start_resseq"]) or start_icode != str(cand["peptide_start_icode"]):
        raise ValueError(
            f"Peptide start mismatch: Step3=({cand['peptide_start_resseq']},{cand['peptide_start_icode']}) "
            f"Step4=({start_num},{start_icode})"
        )
    if end_num != int(cand["peptide_end_resseq"]) or end_icode != str(cand["peptide_end_icode"]):
        raise ValueError(
            f"Peptide end mismatch: Step3=({cand['peptide_end_resseq']},{cand['peptide_end_icode']}) "
            f"Step4=({end_num},{end_icode})"
        )


def residue_contact_flags(
    peptide_coords_per_residue: List[np.ndarray], receptor_tree: cKDTree, cutoff: float
) -> List[bool]:
    flags: List[bool] = []
    for coords in peptide_coords_per_residue:
        if len(coords) == 0:
            flags.append(False)
            continue
        dists, _ = receptor_tree.query(coords, k=1)
        flags.append(float(np.min(dists)) <= cutoff)
    return flags


def count_heavy_contacts_with_tree(peptide_coords: np.ndarray, receptor_tree: cKDTree, cutoff: float) -> int:
    if len(peptide_coords) == 0:
        return 0
    neighbors = receptor_tree.query_ball_point(peptide_coords, r=cutoff)
    return int(sum(len(x) for x in neighbors))


def compute_pocket_size_with_tree(
    peptide_coords: np.ndarray,
    receptor_tree: cKDTree,
    atom_to_res_idx: np.ndarray,
    cutoff: float,
) -> int:
    if len(peptide_coords) == 0 or len(atom_to_res_idx) == 0:
        return 0
    neighbors = receptor_tree.query_ball_point(peptide_coords, r=cutoff)
    receptor_residue_ids = {
        int(atom_to_res_idx[atom_idx])
        for atom_neighbors in neighbors
        for atom_idx in atom_neighbors
    }
    return len(receptor_residue_ids)


def passes_rbsa_prescreen(
    n_contact_residues_6a: int,
    contact_coverage_6a: float,
    pocket_size_6a: int,
    cfg: Step4Config,
) -> Tuple[bool, Optional[str]]:
    if n_contact_residues_6a < cfg.min_contact_residues_6a:
        return False, "too_few_contact_residues_6A"
    if contact_coverage_6a < cfg.min_contact_coverage_for_rbsa:
        return False, "low_contact_coverage_6A"
    if pocket_size_6a < cfg.min_pocket_size_for_rbsa:
        return False, "small_pocket_6A"
    return True, None


def get_atom_by_name(residue: gemmi.Residue, atom_name: str) -> Optional[gemmi.Atom]:
    for atom in residue:
        if atom.name.strip() == atom_name:
            return atom
    return None


def peptide_bond_continuous(left_res: gemmi.Residue, right_res: gemmi.Residue, min_dist: float = 1.2, max_dist: float = 1.6) -> bool:
    c_atom = get_atom_by_name(left_res, "C")
    n_atom = get_atom_by_name(right_res, "N")
    if c_atom is None or n_atom is None:
        return False
    dx = c_atom.pos.x - n_atom.pos.x
    dy = c_atom.pos.y - n_atom.pos.y
    dz = c_atom.pos.z - n_atom.pos.z
    dist = (dx * dx + dy * dy + dz * dz) ** 0.5
    return min_dist <= dist <= max_dist


def compute_covalent_bias_risk(all_pep_res: List[Tuple[gemmi.Residue, np.ndarray]], left_idx: int, right_idx: int) -> float:
    risk = 0.0
    if left_idx > 0:
        prev_res, _ = all_pep_res[left_idx - 1]
        cur_res, _ = all_pep_res[left_idx]
        if peptide_bond_continuous(prev_res, cur_res):
            risk += 0.5
    if right_idx + 1 < len(all_pep_res):
        cur_res, _ = all_pep_res[right_idx]
        next_res, _ = all_pep_res[right_idx + 1]
        if peptide_bond_continuous(cur_res, next_res):
            risk += 0.5
    return float(risk)


def atoms_from_residues(
    residues: List[gemmi.Residue], chain_id_override: Optional[str] = None
) -> List[Tuple[str, str, str, str, float, float, float, float]]:
    atoms = []
    chain_id = (chain_id_override or "A")[:1]
    for residue in residues:
        resname = residue.name
        resnum = str(residue_seqid_num(residue))
        icode = residue_seqid_icode(residue)
        if icode:
            resnum = f"{resnum}{icode}"
        for atom in residue_heavy_atoms(residue):
            atoms.append((
                atom.name.strip() or atom.element.name,
                resname,
                resnum,
                chain_id,
                float(atom.pos.x),
                float(atom.pos.y),
                float(atom.pos.z),
                float(atom_radius(atom)),
            ))
    return atoms


def atoms_from_cached_slices(
    per_residue_atoms: List[List[Tuple[str, str, str, str, float, float, float, float]]],
    left_idx: int,
    right_idx: int,
    chain_id_override: Optional[str] = None,
) -> List[Tuple[str, str, str, str, float, float, float, float]]:
    chain_id = ((chain_id_override or "A")[:1]) if chain_id_override else None
    atoms: List[Tuple[str, str, str, str, float, float, float, float]] = []
    for residue_atoms in per_residue_atoms[left_idx:right_idx + 1]:
        if chain_id is None:
            atoms.extend(residue_atoms)
        else:
            atoms.extend([
                (atom_name, resname, resnum, chain_id, x, y, z, radius)
                for atom_name, resname, resnum, _old_chain_id, x, y, z, radius in residue_atoms
            ])
    return atoms


def concat_cached_coords(
    per_residue_coords: List[np.ndarray],
    left_idx: int,
    right_idx: int,
) -> np.ndarray:
    coords = [arr for arr in per_residue_coords[left_idx:right_idx + 1] if len(arr) > 0]
    if not coords:
        return np.zeros((0, 3), dtype=float)
    return np.concatenate(coords, axis=0)


def build_freesasa_structure(
    atoms: List[Tuple[str, str, str, str, float, float, float, float]]
) -> freesasa.Structure:
    fs = freesasa.Structure()
    for atom_name, resname, resnum, chain_id, x, y, z, _radius in atoms:
        fs.addAtom(atom_name, resname, resnum, chain_id, x, y, z)
    radii = [a[7] for a in atoms]
    fs.setRadii(radii)
    return fs


def compute_rbsa_raw_from_atoms(
    peptide_atoms: List[Tuple[str, str, str, str, float, float, float, float]],
    receptor_atoms: List[Tuple[str, str, str, str, float, float, float, float]],
) -> Tuple[float, float, float, float]:
    complex_atoms = peptide_atoms + receptor_atoms

    fs_free = build_freesasa_structure(peptide_atoms)
    res_free = freesasa.calc(fs_free)
    peptide_sasa_free = float(res_free.totalArea())

    fs_complex = build_freesasa_structure(complex_atoms)
    res_complex = freesasa.calc(fs_complex)
    selections = freesasa.selectArea(("pep, chain P",), fs_complex, res_complex)
    peptide_sasa_bound = float(selections["pep"])

    peptide_buried_sasa = max(0.0, peptide_sasa_free - peptide_sasa_bound)
    rbsa_raw = (peptide_buried_sasa / peptide_sasa_free) if peptide_sasa_free > 0 else 0.0
    return rbsa_raw, peptide_sasa_free, peptide_sasa_bound, peptide_buried_sasa


def get_chain_context(
    cand: Dict[str, Any],
    pdb_dir: Path,
) -> Dict[str, Any]:
    global _WORKER_CHAIN_CACHE

    cache_key = (
        cand["source_file"],
        cand["receptor_chain_id"],
        cand["peptide_source_chain_id"],
    )
    cached = _WORKER_CHAIN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    cif_path = pdb_dir / cand["source_file"]
    if not cif_path.exists():
        raise FileNotFoundError(f"Missing CIF: {cif_path}")

    structure = load_structure(cif_path)
    model = get_model(structure)
    receptor_chain = find_chain_by_name(model, cand["receptor_chain_id"])
    peptide_chain = find_chain_by_name(model, cand["peptide_source_chain_id"])

    all_peptide_residues = chain_residues_with_coords_keep_alignment(peptide_chain)
    all_receptor_residues = chain_residues_with_coords_keep_alignment(receptor_chain)
    receptor_tree, atom_to_res_idx, receptor_coords = build_receptor_kdtree_from_residues(all_receptor_residues)
    if receptor_tree is None or len(receptor_coords) == 0:
        raise ValueError("Empty receptor heavy-atom coordinates")

    receptor_atoms = atoms_from_residues([res for res, _ in all_receptor_residues], chain_id_override="R")
    peptide_coords_per_residue = [coords for _res, coords in all_peptide_residues]
    peptide_atoms_per_residue = [
        atoms_from_residues([res], chain_id_override="P")
        for res, _coords in all_peptide_residues
    ]

    cached = {
        "all_peptide_residues": all_peptide_residues,
        "all_receptor_residues": all_receptor_residues,
        "receptor_tree": receptor_tree,
        "atom_to_res_idx": atom_to_res_idx,
        "receptor_coords": receptor_coords,
        "receptor_atoms": receptor_atoms,
        "peptide_coords_per_residue": peptide_coords_per_residue,
        "peptide_atoms_per_residue": peptide_atoms_per_residue,
    }
    _WORKER_CHAIN_CACHE[cache_key] = cached
    return cached


def score_candidate(cand: Dict[str, Any], pdb_dir: Path, cfg: Step4Config) -> Dict[str, Any]:
    if cfg.debug:
        print(
            f"[DEBUG] load_structure candidate_id={cand.get('candidate_id','')} "
            f"pdb_id={cand.get('pdb_id','')} source_file={cand.get('source_file','')}",
            flush=True,
        )

    context = get_chain_context(cand, pdb_dir)
    all_peptide_residues = context["all_peptide_residues"]
    all_receptor_residues = context["all_receptor_residues"]
    peptide_coords_per_residue = context["peptide_coords_per_residue"]

    left_idx = int(cand["final_left_index"])
    right_idx = int(cand["final_right_index"])
    if left_idx < 0 or right_idx >= len(all_peptide_residues) or left_idx > right_idx:
        raise ValueError(f"Invalid candidate indices: {left_idx} to {right_idx}")

    peptide_residues = [res for res, _ in all_peptide_residues[left_idx:right_idx + 1]]
    validate_candidate_window(cand, peptide_residues)

    receptor_tree = context["receptor_tree"]
    atom_to_res_idx = context["atom_to_res_idx"]
    receptor_atoms = context["receptor_atoms"]
    peptide_window_coords = peptide_coords_per_residue[left_idx:right_idx + 1]
    peptide_coords = concat_cached_coords(peptide_window_coords, 0, len(peptide_window_coords) - 1)
    if len(peptide_coords) == 0:
        raise ValueError("Peptide has no heavy atoms")
    peptide_atoms = atoms_from_cached_slices(context["peptide_atoms_per_residue"], left_idx, right_idx)

    flags_6a = residue_contact_flags(peptide_window_coords, receptor_tree, cfg.contact_cutoff_6a)

    n_contact_residues_6a = int(sum(1 for x in flags_6a if x))
    contact_coverage_6a = n_contact_residues_6a / len(peptide_residues) if peptide_residues else 0.0

    n_contact_atoms_6a = count_heavy_contacts_with_tree(peptide_coords, receptor_tree, cfg.contact_cutoff_6a)
    pocket_size = compute_pocket_size_with_tree(peptide_coords, receptor_tree, atom_to_res_idx, cfg.pocket_cutoff_6a)
    covalent_bias_risk = compute_covalent_bias_risk(all_peptide_residues, left_idx, right_idx)

    rbsa_prescreen_passed, rbsa_prescreen_fail_reason = passes_rbsa_prescreen(
        n_contact_residues_6a=n_contact_residues_6a,
        contact_coverage_6a=contact_coverage_6a,
        pocket_size_6a=pocket_size,
        cfg=cfg,
    )

    rbsa_skipped_for_prescreen = not rbsa_prescreen_passed
    if rbsa_skipped_for_prescreen:
        rbsa_raw = 0.0
        peptide_sasa_free = 0.0
        peptide_sasa_bound = 0.0
        peptide_buried_sasa = 0.0
    else:
        if cfg.debug:
            print(
                f"[DEBUG] entering_freesasa candidate_id={cand.get('candidate_id','')} "
                f"pdb_id={cand.get('pdb_id','')} source_file={cand.get('source_file','')}",
                flush=True,
            )

        rbsa_raw, peptide_sasa_free, peptide_sasa_bound, peptide_buried_sasa = compute_rbsa_raw_from_atoms(
            peptide_atoms, receptor_atoms
        )

    passes_sanity = rbsa_prescreen_passed and (rbsa_raw >= cfg.min_rbsa_raw)
    drop_reason = None
    if not passes_sanity:
        if n_contact_residues_6a < cfg.min_contact_residues_6a:
            drop_reason = "too_few_contact_residues_6A"
        elif not rbsa_prescreen_passed and rbsa_prescreen_fail_reason is not None:
            drop_reason = f"failed_rbsa_prescreen:{rbsa_prescreen_fail_reason}"
        elif rbsa_raw < cfg.min_rbsa_raw:
            drop_reason = "rbsa_too_low"

    return {
        "candidate_id": cand["candidate_id"],
        "parent_task_id": cand["parent_task_id"],
        "pdb_id": cand["pdb_id"],
        "source_file": cand["source_file"],
        "assembly_id": cand.get("assembly_id", "unknown"),
        "chain_pair_id": cand["chain_pair_id"],
        "direction": cand["direction"],
        "receptor_chain_id": cand["receptor_chain_id"],
        "peptide_source_chain_id": cand["peptide_source_chain_id"],
        "method": cand.get("method"),

        "anchor_min_distance": cand["anchor_min_distance"],
        "anchor_peptide_res_index": cand["anchor_peptide_res_index"],
        "anchor_receptor_res_index": cand["anchor_receptor_res_index"],
        "final_left_index": cand["final_left_index"],
        "final_right_index": cand["final_right_index"],
        "peptide_length": cand["peptide_length"],
        "both_cap": bool(cand.get("hit_left_growth_cap", False) and cand.get("hit_right_growth_cap", False)),

        "peptide_start_resseq": cand["peptide_start_resseq"],
        "peptide_start_icode": cand["peptide_start_icode"],
        "peptide_end_resseq": cand["peptide_end_resseq"],
        "peptide_end_icode": cand["peptide_end_icode"],
        "peptide_residue_ids": [residue_id_string(r) for r in peptide_residues],

        "rBSA_raw": rbsa_raw,
        "contact_coverage_6A": contact_coverage_6a,
        "n_contact_atoms_6A": n_contact_atoms_6a,
        "n_contact_residues_6A": n_contact_residues_6a,
        "pocket_size_6A": pocket_size,
        "covalent_bias_risk": covalent_bias_risk,
        "peptide_sasa_free": peptide_sasa_free,
        "peptide_sasa_bound": peptide_sasa_bound,
        "peptide_buried_sasa": peptide_buried_sasa,
        "step4_rbsa_prescreen_passed": rbsa_prescreen_passed,
        "step4_rbsa_prescreen_fail_reason": rbsa_prescreen_fail_reason,
        "step4_rbsa_skipped_for_prescreen": rbsa_skipped_for_prescreen,

        "step4_passes_sanity": passes_sanity,
        "step4_drop_reason": drop_reason,
    }


def init_worker(cfg: Step4Config) -> None:
    global _WORKER_CFG
    global _WORKER_CHAIN_CACHE
    _WORKER_CFG = cfg
    _WORKER_CHAIN_CACHE = {}


def worker(payload: Tuple[Dict[str, Any], str]) -> Dict[str, Any]:
    cand, pdb_dir_str = payload
    pdb_dir = Path(pdb_dir_str)
    try:
        if _WORKER_CFG is None:
            raise RuntimeError("Worker config is not initialized.")

        if _WORKER_CFG.debug:
            print(
                f"[DEBUG] start candidate_id={cand.get('candidate_id','')} "
                f"pdb_id={cand.get('pdb_id','')} "
                f"source_file={cand.get('source_file','')}",
                flush=True,
            )

        feat = score_candidate(cand, pdb_dir, _WORKER_CFG)
        return {"ok": True, "candidate_id": cand.get("candidate_id", ""), "feature": feat, "error": None}
    except Exception as e:
        return {
            "ok": False,
            "candidate_id": cand.get("candidate_id", ""),
            "pdb_id": cand.get("pdb_id", ""),
            "source_file": cand.get("source_file", ""),
            "direction": cand.get("direction", ""),
            "receptor_chain_id": cand.get("receptor_chain_id", ""),
            "peptide_source_chain_id": cand.get("peptide_source_chain_id", ""),
            "feature": None,
            "error": {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(limit=3),
            },
        }


def iter_candidates(path: Path, max_candidates: int = 0) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_candidates > 0 and i >= max_candidates:
                break
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)


def count_candidates(path: Path, max_candidates: int = 0) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_candidates > 0 and i >= max_candidates:
                break
            if line.strip():
                n += 1
    return n


def handle_result(
    result: Dict[str, Any],
    f_out,
    f_err,
    keep_failed_sanity: bool,
) -> Tuple[int, int, int]:
    kept = 0
    dropped_sanity = 0
    errors = 0

    if result["ok"]:
        feat = result["feature"]
        if feat is None:
            errors += 1
        else:
            if feat.get("step4_passes_sanity", False) or keep_failed_sanity:
                kept += 1
                f_out.write(json.dumps(feat, ensure_ascii=False) + "\n")
            else:
                dropped_sanity += 1
    else:
        errors += 1
        if f_err is not None:
            f_err.write(json.dumps(result, ensure_ascii=False) + "\n")

    return kept, dropped_sanity, errors


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="PeptideCLIP Step-4 Method A postscore candidates")
    parser.add_argument("--candidate_jsonl", type=str, required=True)
    parser.add_argument("--pdb_dir", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--error_jsonl", type=str, default="")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    parser.add_argument("--chunksize", type=int, default=20)
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument("--max_candidates", type=int, default=0)
    parser.add_argument("--contact_cutoff_6a", type=float, default=6.0)
    parser.add_argument("--pocket_cutoff_6a", type=float, default=6.0)
    parser.add_argument("--min_contact_residues_6a", type=int, default=2)
    parser.add_argument("--min_contact_coverage_for_rbsa", type=float, default=0.25)
    parser.add_argument("--min_pocket_size_for_rbsa", type=int, default=2)
    parser.add_argument("--min_rbsa_raw", type=float, default=0.05)
    parser.add_argument(
        "--keep_failed_sanity",
        action="store_true",
        help="Keep rows that fail sanity checks in output_jsonl.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-candidate debug logs.",
    )
    args = parser.parse_args()

    cfg = Step4Config(
        contact_cutoff_6a=args.contact_cutoff_6a,
        pocket_cutoff_6a=args.pocket_cutoff_6a,
        min_contact_residues_6a=args.min_contact_residues_6a,
        min_contact_coverage_for_rbsa=args.min_contact_coverage_for_rbsa,
        min_pocket_size_for_rbsa=args.min_pocket_size_for_rbsa,
        min_rbsa_raw=args.min_rbsa_raw,
        debug=args.debug,
    )

    candidate_jsonl = Path(args.candidate_jsonl)
    pdb_dir = Path(args.pdb_dir)
    output_jsonl = Path(args.output_jsonl)
    error_jsonl = Path(args.error_jsonl) if args.error_jsonl else None

    start_time = time.time()
    pid = os.getpid()

    print("=" * 80, flush=True)
    print(f"[START] PID={pid}", flush=True)
    print(f"[START] candidate_jsonl         = {candidate_jsonl}", flush=True)
    print(f"[START] pdb_dir                = {pdb_dir}", flush=True)
    print(f"[START] output_jsonl           = {output_jsonl}", flush=True)
    print(f"[START] error_jsonl            = {error_jsonl}", flush=True)
    print(f"[START] workers                = {args.workers}", flush=True)
    print(f"[START] chunksize              = {args.chunksize}", flush=True)
    print(f"[START] progress_every         = {args.progress_every}", flush=True)
    print(f"[START] max_candidates         = {args.max_candidates}", flush=True)
    print(f"[START] contact_cutoff_6a      = {args.contact_cutoff_6a}", flush=True)
    print(f"[START] pocket_cutoff_6a       = {args.pocket_cutoff_6a}", flush=True)
    print(f"[START] min_contact_residues_6a= {args.min_contact_residues_6a}", flush=True)
    print(f"[START] min_contact_coverage_for_rbsa = {args.min_contact_coverage_for_rbsa}", flush=True)
    print(f"[START] min_pocket_size_for_rbsa      = {args.min_pocket_size_for_rbsa}", flush=True)
    print(f"[START] min_rbsa_raw           = {args.min_rbsa_raw}", flush=True)
    print(f"[START] debug                  = {args.debug}", flush=True)
    print("=" * 80, flush=True)

    total_candidates = count_candidates(candidate_jsonl, args.max_candidates)
    print(f"[INFO] Counted {total_candidates} Step-3 candidates", flush=True)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if error_jsonl is not None:
        error_jsonl.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    kept = 0
    dropped_sanity = 0
    errors = 0

    f_err = None
    try:
        with output_jsonl.open("w", encoding="utf-8") as f_out:
            if error_jsonl is not None:
                f_err = error_jsonl.open("w", encoding="utf-8")

            if args.workers <= 1:
                print("[INFO] Running in true serial + streaming mode.", flush=True)
                init_worker(cfg)

                for cand in iter_candidates(candidate_jsonl, args.max_candidates):
                    result = worker((cand, str(pdb_dir)))
                    processed += 1

                    d_kept, d_drop, d_err = handle_result(
                        result=result,
                        f_out=f_out,
                        f_err=f_err,
                        keep_failed_sanity=args.keep_failed_sanity,
                    )
                    kept += d_kept
                    dropped_sanity += d_drop
                    errors += d_err

                    if processed % args.progress_every == 0:
                        elapsed = time.time() - start_time
                        speed = processed / elapsed if elapsed > 0 else 0.0
                        print(
                            f"[PROGRESS] {processed}/{total_candidates} | kept={kept} | dropped_sanity={dropped_sanity} | "
                            f"errors={errors} | elapsed={elapsed/60:.1f} min | speed={speed:.2f} cand/s",
                            flush=True,
                        )
            else:
                print("[INFO] Running in multiprocessing + streaming mode.", flush=True)
                payloads = ((cand, str(pdb_dir)) for cand in iter_candidates(candidate_jsonl, args.max_candidates))
                with mp.Pool(processes=args.workers, initializer=init_worker, initargs=(cfg,)) as pool:
                    for result in pool.imap_unordered(worker, payloads, chunksize=args.chunksize):
                        processed += 1

                        d_kept, d_drop, d_err = handle_result(
                            result=result,
                            f_out=f_out,
                            f_err=f_err,
                            keep_failed_sanity=args.keep_failed_sanity,
                        )
                        kept += d_kept
                        dropped_sanity += d_drop
                        errors += d_err

                        if processed % args.progress_every == 0:
                            elapsed = time.time() - start_time
                            speed = processed / elapsed if elapsed > 0 else 0.0
                            print(
                                f"[PROGRESS] {processed}/{total_candidates} | kept={kept} | dropped_sanity={dropped_sanity} | "
                                f"errors={errors} | elapsed={elapsed/60:.1f} min | speed={speed:.2f} cand/s",
                                flush=True,
                            )

        elapsed = time.time() - start_time
        print("=" * 80, flush=True)
        print("[DONE] Step-4 postscore finished.", flush=True)
        print(f"[DONE] Processed candidates : {processed}", flush=True)
        print(f"[DONE] Kept candidates      : {kept}", flush=True)
        print(f"[DONE] Dropped by sanity    : {dropped_sanity}", flush=True)
        print(f"[DONE] Worker errors       : {errors}", flush=True)
        print(f"[DONE] Elapsed time        : {elapsed/60:.2f} min", flush=True)
        print(f"[DONE] Output JSONL        : {output_jsonl}", flush=True)
        if error_jsonl is not None:
            print(f"[DONE] Error JSONL         : {error_jsonl}", flush=True)
        print("=" * 80, flush=True)
    finally:
        if f_err is not None:
            f_err.close()


if __name__ == "__main__":
    main()
