from __future__ import annotations

"""
Phase-1 Step 8 v4
Finalize dataset into Track A / Track B LMDB with:
- stable schema for main/monitor split
- Step7-aware metadata preservation
- Step5 sampling provenance preservation
- natural-terminus and proxy-cap metadata
- alignment-safe residue indexing (consistent with Step3/4/6 v2)
"""

import argparse
import json
import multiprocessing as mp
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import gemmi
import lmdb
import msgpack
import msgpack_numpy
import numpy as np
from scipy.spatial import cKDTree

msgpack_numpy.patch()

STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "MSE", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL"
}


@dataclass
class AtomRecord:
    coord: np.ndarray
    element: str
    atom_name: str
    residue_tag: str
    is_cap_atom: bool


# =========================================================
# Basic helpers
# =========================================================
def is_standard_protein_residue(res: gemmi.Residue) -> bool:
    return res.name in STANDARD_AA


def residue_heavy_atom_coords(residue: gemmi.Residue) -> np.ndarray:
    coords = []
    for atom in residue:
        if atom.element.name == "H":
            continue
        coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
    if not coords:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(coords, dtype=np.float32)


def chain_residues_with_coords_keep_alignment(chain: gemmi.Chain) -> List[Tuple[gemmi.Residue, np.ndarray]]:
    out: List[Tuple[gemmi.Residue, np.ndarray]] = []
    for residue in chain:
        if not is_standard_protein_residue(residue):
            continue
        out.append((residue, residue_heavy_atom_coords(residue)))
    return out


def get_heavy_atoms(residue: gemmi.Residue) -> List[gemmi.Atom]:
    return [a for a in residue if a.element.name != "H"]


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


def residue_seqid_num(residue: gemmi.Residue) -> int:
    return int(residue.seqid.num)


def residue_seqid_icode(residue: gemmi.Residue) -> str:
    return str(residue.seqid.icode).strip()


def residue_id_string(res: gemmi.Residue) -> str:
    num = residue_seqid_num(res)
    icode = residue_seqid_icode(res)
    return f"{num}{icode}:{res.name}" if icode else f"{num}:{res.name}"


def residue_one_letter(res: gemmi.Residue) -> str:
    try:
        info = gemmi.find_tabulated_residue(res.name)
        code = info.one_letter_code
        if code and code != "?":
            return code
    except Exception:
        pass
    return "X"


def get_atom_coord(res: gemmi.Residue, atom_name: str) -> Optional[np.ndarray]:
    for atom in res:
        if atom.name.strip() == atom_name:
            return np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float32)
    return None


def check_peptide_bond(
    left_res: gemmi.Residue,
    right_res: gemmi.Residue,
    min_dist: float = 1.2,
    max_dist: float = 1.8,
) -> bool:
    c_pos = get_atom_coord(left_res, "C")
    n_pos = get_atom_coord(right_res, "N")
    if c_pos is None or n_pos is None:
        return False
    dist = float(np.linalg.norm(c_pos - n_pos))
    return min_dist <= dist <= max_dist


def validate_candidate_window(
    row: Dict[str, Any],
    peptide_residues: List[gemmi.Residue],
) -> None:
    if not peptide_residues:
        raise ValueError("Empty peptide residue window after slicing")

    start_res = peptide_residues[0]
    end_res = peptide_residues[-1]

    start_num = residue_seqid_num(start_res)
    start_icode = residue_seqid_icode(start_res)
    end_num = residue_seqid_num(end_res)
    end_icode = residue_seqid_icode(end_res)

    if start_num != int(row["peptide_start_resseq"]) or start_icode != str(row["peptide_start_icode"]):
        raise ValueError(
            f"Peptide start mismatch: row=({row['peptide_start_resseq']},{row['peptide_start_icode']}) "
            f"actual=({start_num},{start_icode})"
        )

    if end_num != int(row["peptide_end_resseq"]) or end_icode != str(row["peptide_end_icode"]):
        raise ValueError(
            f"Peptide end mismatch: row=({row['peptide_end_resseq']},{row['peptide_end_icode']}) "
            f"actual=({end_num},{end_icode})"
        )


def pack_msgpack(obj: Dict[str, Any]) -> bytes:
    return msgpack.packb(obj, use_bin_type=True)


# =========================================================
# Original peptide atoms
# =========================================================
def build_original_peptide_atom_records(
    peptide_residues_original: List[gemmi.Residue],
) -> List[AtomRecord]:
    atoms: List[AtomRecord] = []
    for res in peptide_residues_original:
        res_tag = residue_id_string(res)
        for atom in get_heavy_atoms(res):
            atoms.append(
                AtomRecord(
                    coord=np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float32),
                    element=atom.element.name,
                    atom_name=atom.name.strip() or atom.element.name,
                    residue_tag=res_tag,
                    is_cap_atom=False,
                )
            )
    return atoms


# =========================================================
# Proxy-capping via neighbor atom borrowing
# =========================================================
def build_n_cap_from_left_neighbor(left_neighbor: gemmi.Residue) -> List[AtomRecord]:
    """
    Proxy N-cap:
    borrow C / O / CA coordinates from the left neighboring residue.
    """
    out: List[AtomRecord] = []
    target_atoms = {
        "C": ("ACE_C", "C"),
        "O": ("ACE_O", "O"),
        "CA": ("ACE_CH3", "C"),
    }

    for src_atom_name, (cap_atom_name, element) in target_atoms.items():
        coord = get_atom_coord(left_neighbor, src_atom_name)
        if coord is not None:
            out.append(
                AtomRecord(
                    coord=coord,
                    element=element,
                    atom_name=cap_atom_name,
                    residue_tag="N_CAP_PROXY",
                    is_cap_atom=True,
                )
            )
    return out


def build_c_cap_from_right_neighbor(right_neighbor: gemmi.Residue) -> List[AtomRecord]:
    """
    Proxy C-cap:
    borrow only the N coordinate from the right neighboring residue.
    """
    out: List[AtomRecord] = []
    coord = get_atom_coord(right_neighbor, "N")
    if coord is not None:
        out.append(
            AtomRecord(
                coord=coord,
                element="N",
                atom_name="NME_N",
                residue_tag="C_CAP_PROXY",
                is_cap_atom=True,
            )
        )
    return out


def build_proxy_capped_peptide_atoms(
    all_pep_res_with_coords: List[Tuple[gemmi.Residue, np.ndarray]],
    left_idx: int,
    right_idx: int,
) -> Tuple[List[AtomRecord], Dict[str, Any]]:
    peptide_residues_original = [res for res, _coords in all_pep_res_with_coords[left_idx:right_idx + 1]]
    if not peptide_residues_original:
        raise ValueError("Original peptide window is empty")

    atom_records = build_original_peptide_atom_records(peptide_residues_original)

    n_cap_atoms: List[AtomRecord] = []
    c_cap_atoms: List[AtomRecord] = []

    has_natural_left_terminus = (left_idx == 0)
    has_natural_right_terminus = (right_idx + 1 >= len(all_pep_res_with_coords))

    left_neighbor_continuous = False
    right_neighbor_continuous = False

    first_res = peptide_residues_original[0]
    last_res = peptide_residues_original[-1]

    if left_idx > 0:
        left_neighbor = all_pep_res_with_coords[left_idx - 1][0]
        if check_peptide_bond(left_neighbor, first_res):
            left_neighbor_continuous = True
            n_cap_atoms = build_n_cap_from_left_neighbor(left_neighbor)

    if right_idx + 1 < len(all_pep_res_with_coords):
        right_neighbor = all_pep_res_with_coords[right_idx + 1][0]
        if check_peptide_bond(last_res, right_neighbor):
            right_neighbor_continuous = True
            c_cap_atoms = build_c_cap_from_right_neighbor(right_neighbor)

    final_atom_records = atom_records + n_cap_atoms + c_cap_atoms

    meta = {
        "has_n_cap": len(n_cap_atoms) > 0,
        "has_c_cap": len(c_cap_atoms) > 0,
        "n_n_cap_atoms": len(n_cap_atoms),
        "n_c_cap_atoms": len(c_cap_atoms),
        "n_original_heavy_atoms": len(atom_records),
        "n_cap_atoms_total": len(n_cap_atoms) + len(c_cap_atoms),
        "peptide_length_original": len(peptide_residues_original),
        "peptide_residue_ids_original": [residue_id_string(r) for r in peptide_residues_original],
        "has_natural_left_terminus": has_natural_left_terminus,
        "has_natural_right_terminus": has_natural_right_terminus,
        "left_neighbor_continuous": left_neighbor_continuous,
        "right_neighbor_continuous": right_neighbor_continuous,
        "is_terminal_native": bool(has_natural_left_terminus or has_natural_right_terminus),
    }
    return final_atom_records, meta


def atom_records_to_arrays(atom_records: List[AtomRecord]) -> Dict[str, Any]:
    if not atom_records:
        raise ValueError("No atom records")

    return {
        "coords": np.asarray([a.coord for a in atom_records], dtype=np.float32),
        "elements": np.asarray([a.element for a in atom_records], dtype=str),
        "atom_names": np.asarray([a.atom_name for a in atom_records], dtype=str),
        "residue_tags": np.asarray([a.residue_tag for a in atom_records], dtype=str),
        "is_cap_atom": np.asarray([a.is_cap_atom for a in atom_records], dtype=bool),
    }


# =========================================================
# Pocket extraction
# =========================================================
def extract_pocket_from_peptide_query(
    receptor_residues: List[gemmi.Residue],
    peptide_query_coords: np.ndarray,
    cutoff: float,
) -> Tuple[List[gemmi.Residue], np.ndarray, List[str], List[str]]:
    rec_coords_list: List[List[float]] = []
    rec_res_map: List[int] = []

    for i, res in enumerate(receptor_residues):
        for atom in get_heavy_atoms(res):
            rec_coords_list.append([atom.pos.x, atom.pos.y, atom.pos.z])
            rec_res_map.append(i)

    rec_coords_np = np.asarray(rec_coords_list, dtype=np.float32)
    if len(rec_coords_np) == 0:
        raise ValueError("Receptor has no heavy atoms")

    rec_tree = cKDTree(rec_coords_np)
    hits = rec_tree.query_ball_point(peptide_query_coords, r=cutoff)

    pocket_atom_indices = set()
    for h in hits:
        pocket_atom_indices.update(h)

    if not pocket_atom_indices:
        raise ValueError("No pocket atoms found within cutoff")

    pocket_res_indices = sorted(set(rec_res_map[i] for i in pocket_atom_indices))
    pocket_residues = [receptor_residues[i] for i in pocket_res_indices]

    pocket_coords: List[List[float]] = []
    pocket_elements: List[str] = []
    pocket_atom_names: List[str] = []

    for res in pocket_residues:
        for atom in get_heavy_atoms(res):
            pocket_coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
            pocket_elements.append(atom.element.name)
            pocket_atom_names.append(atom.name.strip() or atom.element.name)

    if not pocket_coords:
        raise ValueError("Pocket has no heavy atoms after extraction")

    return (
        pocket_residues,
        np.asarray(pocket_coords, dtype=np.float32),
        pocket_elements,
        pocket_atom_names,
    )


# =========================================================
# Sequence / metadata helpers
# =========================================================
def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    if x is None:
        return default
    try:
        return float(x)
    except Exception:
        return default


def preserve_optional_fields(row: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in keys:
        if k in row:
            out[k] = row[k]
    return out


# =========================================================
# Worker
# =========================================================
def process_single_candidate(payload: Tuple[Dict[str, Any], str, float]) -> Dict[str, Any]:
    row, pdb_dir_str, cutoff = payload
    cid_str = str(row["candidate_id"])

    try:
        cif_path = Path(pdb_dir_str) / row["source_file"]
        if not cif_path.exists():
            raise FileNotFoundError(f"Missing CIF: {cif_path}")

        st = gemmi.read_structure(str(cif_path))
        st.setup_entities()
        if len(st) == 0:
            raise ValueError("Empty structure")
        model = st[0]

        rec_chain = find_chain_by_name(model, row["receptor_chain_id"])
        pep_chain = find_chain_by_name(model, row["peptide_source_chain_id"])

        all_pep_res_with_coords = chain_residues_with_coords_keep_alignment(pep_chain)
        all_rec_res_with_coords = chain_residues_with_coords_keep_alignment(rec_chain)

        if not all_pep_res_with_coords:
            raise ValueError("No valid peptide residues in peptide source chain")
        if not all_rec_res_with_coords:
            raise ValueError("No valid receptor residues in receptor chain")

        left_idx = int(row["final_left_index"])
        right_idx = int(row["final_right_index"])

        if left_idx < 0 or right_idx >= len(all_pep_res_with_coords) or left_idx > right_idx:
            raise ValueError("Invalid peptide indices out of bounds")

        peptide_residues_original = [res for res, _coords in all_pep_res_with_coords[left_idx:right_idx + 1]]
        validate_candidate_window(row, peptide_residues_original)

        rec_residues_all = [res for res, _coords in all_rec_res_with_coords]

        pep_seq_original = "".join(residue_one_letter(r) for r in peptide_residues_original)
        rec_full_seq = "".join(residue_one_letter(r) for r in rec_residues_all)

        capped_atom_records, cap_meta = build_proxy_capped_peptide_atoms(
            all_pep_res_with_coords, left_idx, right_idx
        )
        pep_arrays = atom_records_to_arrays(capped_atom_records)

        pocket_residues, pocket_coords, pocket_elements, pocket_atom_names = extract_pocket_from_peptide_query(
            rec_residues_all, pep_arrays["coords"], cutoff
        )

        # Track A: compact 1D sample
        track_a_1d = {
            "candidate_id": cid_str,
            "split": (row.get("split") or "main_train"),
            "group_key": row.get("group_key"),
            "bin_key": row.get("bin_key"),
            "selection_mode": row.get("step5_selection_mode"),
            "selected_by_sampling": bool(row.get("step5_selected_by_sampling", False)),
            "receptor_chain_id": row["receptor_chain_id"],
            "peptide_source_chain_id": row["peptide_source_chain_id"],
            "rec_full_seq": rec_full_seq,
            "pep_seq_original": pep_seq_original,
            "final_left_index": left_idx,
            "final_right_index": right_idx,
            "has_n_cap": cap_meta["has_n_cap"],
            "has_c_cap": cap_meta["has_c_cap"],
            "is_terminal_native": cap_meta["is_terminal_native"],
            "has_natural_left_terminus": cap_meta["has_natural_left_terminus"],
            "has_natural_right_terminus": cap_meta["has_natural_right_terminus"],
        }

        # Track B: 3D sample
        track_b_3d = {
            "candidate_id": cid_str,
            "pep_coords": pep_arrays["coords"],
            "pep_elements": pep_arrays["elements"],
            "pep_atom_names": pep_arrays["atom_names"],
            "pep_residue_tags": pep_arrays["residue_tags"],
            "pep_is_cap_atom": pep_arrays["is_cap_atom"],
            "pocket_coords": pocket_coords,
            "pocket_elements": np.asarray(pocket_elements, dtype=str),
            "pocket_atom_names": np.asarray(pocket_atom_names, dtype=str),
        }

        meta = dict(row)
        meta.update({
            "step8_version": "v4_track_schema_stable",
            "capping_mode": "proxy_neighbor_atom_borrowing",
            "pocket_cutoff": float(cutoff),
            "has_n_cap": cap_meta["has_n_cap"],
            "has_c_cap": cap_meta["has_c_cap"],
            "n_n_cap_atoms": cap_meta["n_n_cap_atoms"],
            "n_c_cap_atoms": cap_meta["n_c_cap_atoms"],
            "n_original_heavy_atoms": cap_meta["n_original_heavy_atoms"],
            "n_cap_atoms_total": cap_meta["n_cap_atoms_total"],
            "peptide_length_original": cap_meta["peptide_length_original"],
            "final_peptide_residue_count_3d": cap_meta["peptide_length_original"],
            "final_pocket_num_residues": len(pocket_residues),
            "final_pocket_num_heavy_atoms": int(len(pocket_coords)),
            "peptide_residue_ids_original": cap_meta["peptide_residue_ids_original"],
            "pocket_residue_ids": [residue_id_string(r) for r in pocket_residues],
            "peptide_sequence": pep_seq_original,
            "receptor_full_sequence": rec_full_seq,
            "has_natural_left_terminus": cap_meta["has_natural_left_terminus"],
            "has_natural_right_terminus": cap_meta["has_natural_right_terminus"],
            "left_neighbor_continuous": cap_meta["left_neighbor_continuous"],
            "right_neighbor_continuous": cap_meta["right_neighbor_continuous"],
            "is_terminal_native": cap_meta["is_terminal_native"],
        })

        # preserve optional provenance/family fields if present upstream
        meta.update(preserve_optional_fields(row, [
            "receptor_family_id",
            "receptor_pfam",
            "receptor_pfam_id",
            "receptor_seq_cluster",
            "receptor_sequence_identity_cluster",
            "receptor_cluster_id",
            "peptide_sequence_cluster",
            "peptide_motif",
            "step5_sampling_weight",
            "step5_selected_by_sampling",
            "step5_selection_mode",
            "score_final",
            "rBSA_raw",
            "contact_coverage_6A",
            "n_contact_atoms_6A",
            "covalent_bias_risk",
            "monitor_group_sample_count",
            "monitor_group_representative",
        ]))

        return {
            "ok": True,
            "cid_str": cid_str,
            "cid_bytes": cid_str.encode("utf-8"),
            "track_a": track_a_1d,
            "track_b": track_b_3d,
            "meta": meta,
        }

    except Exception as e:
        return {
            "ok": False,
            "cid_str": cid_str,
            "error": {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(limit=2),
            },
        }


# =========================================================
# LMDB batch write
# =========================================================
def flush_batch(
    env_1d: lmdb.Environment,
    env_3d: lmdb.Environment,
    batch_records: List[Dict[str, Any]],
) -> None:
    if not batch_records:
        return

    with env_1d.begin(write=True) as txn_1d, env_3d.begin(write=True) as txn_3d:
        for rec in batch_records:
            txn_1d.put(rec["cid_bytes"], pack_msgpack(rec["track_a"]))
            txn_3d.put(rec["cid_bytes"], pack_msgpack(rec["track_b"]))


# =========================================================
# Main
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Step 8 v4: finalize dataset into LMDB")
    parser.add_argument("--input_jsonls", type=str, nargs="+", required=True, help="Step7 main/monitor JSONL(s)")
    parser.add_argument("--pdb_dir", type=str, required=True)
    parser.add_argument("--lmdb_dir", type=str, required=True)
    parser.add_argument("--pocket_cutoff", type=float, default=6.0)
    parser.add_argument("--map_size_gb", type=int, default=20)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    parser.add_argument("--chunksize", type=int, default=20)
    parser.add_argument("--commit_every", type=int, default=1000)
    args = parser.parse_args()

    lmdb_path = Path(args.lmdb_dir)
    lmdb_path.mkdir(parents=True, exist_ok=True)
    map_size = args.map_size_gb * 1024 * 1024 * 1024

    env_1d = lmdb.open(
        str(lmdb_path / "track_a_1d"),
        map_size=map_size,
        subdir=True,
        lock=True,
        readonly=False,
        meminit=False,
    )
    env_3d = lmdb.open(
        str(lmdb_path / "track_b_3d"),
        map_size=map_size,
        subdir=True,
        lock=True,
        readonly=False,
        meminit=False,
    )

    meta_out = lmdb_path / "final_metadata.jsonl"
    err_out = lmdb_path / "step8_errors.jsonl"
    summary_out = lmdb_path / "step8_summary.json"

    rows: List[Dict[str, Any]] = []
    for p in args.input_jsonls:
        with Path(p).open("r", encoding="utf-8") as f:
            rows.extend([json.loads(line) for line in f if line.strip()])

    payloads = [(row, args.pdb_dir, args.pocket_cutoff) for row in rows]

    print("=" * 80, flush=True)
    print("[START] Step 8 v4: Finalize dataset into LMDB", flush=True)
    print(f"[START] Total targets = {len(payloads)}", flush=True)
    print(f"[START] pocket_cutoff = {args.pocket_cutoff}", flush=True)
    print("=" * 80, flush=True)

    success_count = 0
    error_count = 0
    start_time = time.time()
    batch_records: List[Dict[str, Any]] = []

    split_counts: Dict[str, int] = {}
    selection_mode_counts: Dict[str, int] = {}
    terminal_native_count = 0
    n_cap_count = 0
    c_cap_count = 0

    try:
        with mp.Pool(args.workers) as pool, \
             meta_out.open("w", encoding="utf-8") as f_meta, \
             err_out.open("w", encoding="utf-8") as f_err:

            for result in pool.imap_unordered(process_single_candidate, payloads, chunksize=args.chunksize):
                if result["ok"]:
                    batch_records.append(result)
                    meta = result["meta"]
                    f_meta.write(json.dumps(meta, ensure_ascii=False) + "\n")
                    success_count += 1

                    split = str(meta.get("split", "unknown"))
                    split_counts[split] = split_counts.get(split, 0) + 1

                    selection_mode = str(meta.get("step5_selection_mode", "unknown"))
                    selection_mode_counts[selection_mode] = selection_mode_counts.get(selection_mode, 0) + 1

                    if bool(meta.get("is_terminal_native", False)):
                        terminal_native_count += 1
                    if bool(meta.get("has_n_cap", False)):
                        n_cap_count += 1
                    if bool(meta.get("has_c_cap", False)):
                        c_cap_count += 1
                else:
                    error_count += 1
                    f_err.write(json.dumps({
                        "candidate_id": result["cid_str"],
                        "error": result["error"],
                    }, ensure_ascii=False) + "\n")

                if len(batch_records) >= args.commit_every:
                    flush_batch(env_1d, env_3d, batch_records)
                    batch_records = []

                total = success_count + error_count
                if total % 500 == 0:
                    print(
                        f"   [PROGRESS] {total}/{len(payloads)} | "
                        f"OK: {success_count} | ERR: {error_count} | "
                        f"Time: {(time.time() - start_time)/60:.1f}m",
                        flush=True
                    )

            if batch_records:
                flush_batch(env_1d, env_3d, batch_records)

    finally:
        env_1d.close()
        env_3d.close()

    elapsed_seconds = time.time() - start_time

    summary = {
        "step8_version": "v4_track_schema_stable",
        "input_count": len(rows),
        "success_count": success_count,
        "error_count": error_count,
        "success_ratio": (success_count / len(rows)) if rows else 0.0,
        "elapsed_seconds": elapsed_seconds,
        "pocket_cutoff": float(args.pocket_cutoff),
        "split_counts": split_counts,
        "selection_mode_counts": selection_mode_counts,
        "terminal_native_count": terminal_native_count,
        "terminal_native_ratio": (terminal_native_count / success_count) if success_count else 0.0,
        "n_cap_count": n_cap_count,
        "n_cap_ratio": (n_cap_count / success_count) if success_count else 0.0,
        "c_cap_count": c_cap_count,
        "c_cap_ratio": (c_cap_count / success_count) if success_count else 0.0,
        "track_a_path": str(lmdb_path / "track_a_1d"),
        "track_b_path": str(lmdb_path / "track_b_3d"),
        "metadata_jsonl": str(meta_out),
        "errors_jsonl": str(err_out),
    }

    with summary_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 80, flush=True)
    print("[DONE] Step 8 v4 finished.", flush=True)
    print(f"[DONE] Success: {success_count} | Errors: {error_count}", flush=True)
    print(f"[DONE] Metadata: {meta_out}", flush=True)
    print(f"[DONE] Errors:   {err_out}", flush=True)
    print(f"[DONE] Summary:  {summary_out}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
