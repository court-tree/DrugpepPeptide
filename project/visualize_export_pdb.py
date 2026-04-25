from __future__ import annotations

import argparse
import json
import string
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import gemmi
import numpy as np


STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "MSE", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
}

CHAIN_POOL = [c for c in string.ascii_uppercase if c not in {"R", "S", "X"}]


def load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def find_rows(
    rows: Sequence[Dict[str, Any]],
    candidate_id: str = "",
    parent_task_id: str = "",
) -> List[Dict[str, Any]]:
    if bool(candidate_id) == bool(parent_task_id):
        raise ValueError("Provide exactly one of --candidate_id or --parent_task_id")

    if candidate_id:
        matched = [row for row in rows if str(row.get("candidate_id", "")) == candidate_id]
    else:
        matched = [row for row in rows if str(row.get("parent_task_id", "")) == parent_task_id]

    if not matched:
        key = f"candidate_id={candidate_id}" if candidate_id else f"parent_task_id={parent_task_id}"
        raise ValueError(f"No rows matched {key}")
    return matched


def resolve_structure_path(pdb_dir: Path, row: Dict[str, Any]) -> Path:
    candidates = []
    source_file = str(row.get("source_file", "")).strip()
    pdb_id = str(row.get("pdb_id", "")).strip()

    if source_file:
        candidates.append(pdb_dir / source_file)
    if pdb_id:
        candidates.append(pdb_dir / f"{pdb_id}.cif")
        candidates.append(pdb_dir / f"{pdb_id.lower()}.cif")
        candidates.append(pdb_dir / f"{pdb_id.upper()}.cif")

    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Could not resolve structure file for pdb_id={pdb_id}, source_file={source_file} under {pdb_dir}"
    )


def load_structure(path: Path) -> gemmi.Structure:
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    if len(st) == 0:
        raise ValueError(f"Empty structure: {path}")
    return st


def find_chain_by_name(model: gemmi.Model, chain_name: str) -> gemmi.Chain:
    target = str(chain_name).strip()
    for chain in model:
        if str(chain.name).strip() == target:
            return chain
    available = [str(c.name) for c in model]
    raise KeyError(f"Chain not found: {chain_name}; available_chains={available}")


def is_standard_protein_residue(residue: gemmi.Residue) -> bool:
    return residue.name in STANDARD_AA


def heavy_atoms(residue: gemmi.Residue) -> List[gemmi.Atom]:
    return [atom for atom in residue if atom.element.name != "H"]


def residue_heavy_atom_coords(residue: gemmi.Residue) -> np.ndarray:
    coords = [[atom.pos.x, atom.pos.y, atom.pos.z] for atom in heavy_atoms(residue)]
    if not coords:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(coords, dtype=np.float32)


def chain_residues_with_alignment(chain: gemmi.Chain) -> List[Tuple[gemmi.Residue, np.ndarray]]:
    out: List[Tuple[gemmi.Residue, np.ndarray]] = []
    for residue in chain:
        if not is_standard_protein_residue(residue):
            continue
        out.append((residue, residue_heavy_atom_coords(residue)))
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


def collect_patch_residues(
    receptor_residues: Sequence[Tuple[gemmi.Residue, np.ndarray]],
    peptide_residues: Sequence[gemmi.Residue],
    cutoff: float,
) -> List[gemmi.Residue]:
    peptide_coords_list = [residue_heavy_atom_coords(res) for res in peptide_residues]
    peptide_coords_list = [arr for arr in peptide_coords_list if len(arr) > 0]
    if not peptide_coords_list:
        return []
    peptide_coords = np.concatenate(peptide_coords_list, axis=0)

    patch: List[gemmi.Residue] = []
    cutoff_sq = float(cutoff) ** 2
    for res, coords in receptor_residues:
        if len(coords) == 0:
            continue
        deltas = coords[:, None, :] - peptide_coords[None, :, :]
        d2 = np.sum(deltas * deltas, axis=2)
        if float(np.min(d2)) <= cutoff_sq:
            patch.append(res)
    return patch


def format_atom_line(
    serial: int,
    atom_name: str,
    resname: str,
    chain_id: str,
    resseq: int,
    icode: str,
    x: float,
    y: float,
    z: float,
    element: str,
) -> str:
    atom_name = atom_name[:4]
    if len(atom_name) < 4:
        atom_field = f"{atom_name:>4}"
    else:
        atom_field = atom_name
    icode_char = (icode or " ")[:1]
    element_field = (element or "").strip()[:2].rjust(2)
    return (
        f"ATOM  {serial:5d} {atom_field}{' ':1}{resname:>3} {chain_id:1}"
        f"{resseq:4d}{icode_char:1}   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"{1.00:6.2f}{20.00:6.2f}          {element_field:>2}"
    )


def write_residue_block(
    lines: List[str],
    residues: Sequence[gemmi.Residue],
    chain_id: str,
    serial_start: int,
) -> int:
    serial = serial_start
    last_resname = "UNK"
    last_resseq = 1
    last_icode = " "
    for residue in residues:
        resseq = residue_seqid_num(residue)
        icode = residue_seqid_icode(residue)
        last_resname = residue.name
        last_resseq = resseq
        last_icode = (icode or " ")[:1]
        for atom in heavy_atoms(residue):
            lines.append(
                format_atom_line(
                    serial=serial,
                    atom_name=atom.name.strip(),
                    resname=residue.name,
                    chain_id=chain_id,
                    resseq=resseq,
                    icode=icode,
                    x=float(atom.pos.x),
                    y=float(atom.pos.y),
                    z=float(atom.pos.z),
                    element=atom.element.name,
                )
            )
            serial += 1
    if residues:
        lines.append(f"TER   {serial:5d}      {last_resname:>3} {chain_id:1}{last_resseq:4d}{last_icode}")
        serial += 1
    return serial


def sort_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            int(row.get("final_left_index", -1)),
            int(row.get("final_right_index", -1)),
            str(row.get("candidate_id", "")),
        ),
    )


def validate_same_context(rows: Sequence[Dict[str, Any]]) -> None:
    first = rows[0]
    keys = ["pdb_id", "source_file", "receptor_chain_id", "peptide_source_chain_id"]
    for row in rows[1:]:
        for key in keys:
            if str(row.get(key, "")) != str(first.get(key, "")):
                raise ValueError(f"Rows do not share common {key}; this exporter expects one structure/context per export")


def build_remarks(
    rows: Sequence[Dict[str, Any]],
    mode: str,
    chain_map: Sequence[Tuple[str, str]],
    receptor_scope: str,
    patch_cutoff: float,
) -> List[str]:
    first = rows[0]
    remarks = [
        "REMARK 900 PEPTIDECLIP VISUALIZATION EXPORT",
        f"REMARK 900 MODE {mode}",
        f"REMARK 900 PDB_ID {first.get('pdb_id', '')}",
        f"REMARK 900 SOURCE_FILE {first.get('source_file', '')}",
        f"REMARK 900 RECEPTOR_CHAIN {first.get('receptor_chain_id', '')}",
        f"REMARK 900 PEPTIDE_SOURCE_CHAIN {first.get('peptide_source_chain_id', '')}",
        f"REMARK 900 RECEPTOR_SCOPE {receptor_scope}",
        f"REMARK 900 PATCH_CUTOFF {patch_cutoff:.2f}",
    ]
    for chain_id, desc in chain_map:
        remarks.append(f"REMARK 900 CHAIN {chain_id} {desc}")
    for row in rows:
        remarks.append(
            "REMARK 901 "
            f"CANDIDATE {row.get('candidate_id', '')} "
            f"TASK {row.get('parent_task_id', '')} "
            f"LEFT {row.get('final_left_index', '')} "
            f"RIGHT {row.get('final_right_index', '')} "
            f"LEN {row.get('peptide_length', '')}"
        )
    return remarks


def export_pdb(
    input_jsonl: Path,
    pdb_dir: Path,
    output_pdb: Path,
    candidate_id: str,
    parent_task_id: str,
    receptor_scope: str,
    patch_cutoff: float,
    include_source_full: bool,
    max_task_candidates: int,
) -> None:
    rows_all = load_jsonl_rows(input_jsonl)
    rows = sort_rows(find_rows(rows_all, candidate_id=candidate_id, parent_task_id=parent_task_id))
    validate_same_context(rows)
    if max_task_candidates > 0 and len(rows) > max_task_candidates:
        rows = rows[:max_task_candidates]

    structure_path = resolve_structure_path(pdb_dir, rows[0])
    st = load_structure(structure_path)
    model = st[0]

    rec_chain = find_chain_by_name(model, rows[0]["receptor_chain_id"])
    pep_chain = find_chain_by_name(model, rows[0]["peptide_source_chain_id"])
    rec_res_all = chain_residues_with_alignment(rec_chain)
    pep_res_all = chain_residues_with_alignment(pep_chain)
    if not rec_res_all or not pep_res_all:
        raise ValueError("Selected chains do not contain standard protein residues")

    candidate_windows: List[Tuple[Dict[str, Any], List[gemmi.Residue]]] = []
    for row in rows:
        left_idx = int(row["final_left_index"])
        right_idx = int(row["final_right_index"])
        if left_idx < 0 or right_idx >= len(pep_res_all) or left_idx > right_idx:
            raise ValueError(f"Invalid peptide window for candidate {row.get('candidate_id', '')}: {left_idx}-{right_idx}")
        candidate_windows.append((row, [res for res, _coords in pep_res_all[left_idx:right_idx + 1]]))

    if receptor_scope == "patch":
        patch_union: List[gemmi.Residue] = []
        seen = set()
        for _row, peptide_window in candidate_windows:
            for res in collect_patch_residues(rec_res_all, peptide_window, patch_cutoff):
                key = residue_id_string(res)
                if key not in seen:
                    seen.add(key)
                    patch_union.append(res)
        receptor_block = patch_union
    else:
        receptor_block = [res for res, _coords in rec_res_all]

    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    chain_map: List[Tuple[str, str]] = [("R", "receptor" if receptor_scope == "full" else "receptor_patch_union")]
    if include_source_full:
        chain_map.append(("S", "source_peptide_full_chain"))

    lines: List[str] = []
    mode = "candidate" if candidate_id else "task"

    if candidate_id:
        chain_map.append(("P", f"candidate_window {rows[0].get('candidate_id', '')}"))
        patch_single = collect_patch_residues(rec_res_all, candidate_windows[0][1], patch_cutoff)
        if patch_single:
            chain_map.append(("X", f"candidate_patch cutoff={patch_cutoff:.1f}A"))
    else:
        if len(candidate_windows) > len(CHAIN_POOL):
            raise ValueError(f"Task has {len(candidate_windows)} candidates, but only {len(CHAIN_POOL)} chain slots are available")
        for idx, (row, _peptide_window) in enumerate(candidate_windows):
            chain_map.append((CHAIN_POOL[idx], f"candidate_window {row.get('candidate_id', '')}"))

    lines.extend(build_remarks(rows, mode=mode, chain_map=chain_map, receptor_scope=receptor_scope, patch_cutoff=patch_cutoff))

    serial = 1
    serial = write_residue_block(lines, receptor_block, chain_id="R", serial_start=serial)

    if include_source_full:
        serial = write_residue_block(lines, [res for res, _coords in pep_res_all], chain_id="S", serial_start=serial)

    if candidate_id:
        serial = write_residue_block(lines, candidate_windows[0][1], chain_id="P", serial_start=serial)
        patch_single = collect_patch_residues(rec_res_all, candidate_windows[0][1], patch_cutoff)
        if patch_single:
            serial = write_residue_block(lines, patch_single, chain_id="X", serial_start=serial)
    else:
        for idx, (_row, peptide_window) in enumerate(candidate_windows):
            serial = write_residue_block(lines, peptide_window, chain_id=CHAIN_POOL[idx], serial_start=serial)

    lines.append("END")
    output_pdb.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export candidate/task visualization into a single PDB file for PyMOL/Chimera inspection."
    )
    parser.add_argument("--input_jsonl", type=str, required=True, help="Input JSONL from Step3/4/5/6/7 containing candidate rows")
    parser.add_argument("--pdb_dir", type=str, required=True, help="Directory containing source CIF files")
    parser.add_argument("--output_pdb", type=str, required=True, help="Output visualization PDB path")
    parser.add_argument("--candidate_id", type=str, default="", help="Export a single candidate by candidate_id")
    parser.add_argument("--parent_task_id", type=str, default="", help="Export all candidates from one parent_task_id")
    parser.add_argument("--receptor_scope", type=str, choices=["full", "patch"], default="full")
    parser.add_argument("--patch_cutoff", type=float, default=6.0, help="Patch cutoff in Angstrom for candidate patch extraction")
    parser.add_argument("--no_source_full", action="store_true", help="Do not include the full peptide source chain as chain S")
    parser.add_argument("--max_task_candidates", type=int, default=12, help="Cap task exports to avoid oversized PDBs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_pdb(
        input_jsonl=Path(args.input_jsonl),
        pdb_dir=Path(args.pdb_dir),
        output_pdb=Path(args.output_pdb),
        candidate_id=str(args.candidate_id).strip(),
        parent_task_id=str(args.parent_task_id).strip(),
        receptor_scope=args.receptor_scope,
        patch_cutoff=float(args.patch_cutoff),
        include_source_full=not args.no_source_full,
        max_task_candidates=int(args.max_task_candidates),
    )
    print(f"[DONE] Wrote visualization PDB: {args.output_pdb}", flush=True)


if __name__ == "__main__":
    main()
