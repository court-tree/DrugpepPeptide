from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from visualize_export_pdb import (
    chain_residues_with_alignment,
    collect_patch_residues,
    find_chain_by_name,
    heavy_atoms,
    load_jsonl_rows,
    load_structure,
    resolve_structure_path,
    residue_id_string,
    residue_seqid_icode,
    residue_seqid_num,
    format_atom_line,
)


def sanitize_token(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("._-") or "unknown"


def write_residue_block(
    lines: List[str],
    residues,
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


def filter_rows(
    rows: Sequence[Dict[str, Any]],
    pdb_id: str,
    receptor_chain_id: str,
    peptide_source_chain_id: str,
) -> List[Dict[str, Any]]:
    matched = [row for row in rows if str(row.get("pdb_id", "")).lower() == pdb_id.lower()]
    if receptor_chain_id:
        matched = [row for row in matched if str(row.get("receptor_chain_id", "")).strip() == receptor_chain_id.strip()]
    if peptide_source_chain_id:
        matched = [row for row in matched if str(row.get("peptide_source_chain_id", "")).strip() == peptide_source_chain_id.strip()]
    return matched


def sort_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("receptor_chain_id", "")),
            str(row.get("peptide_source_chain_id", "")),
            int(row.get("final_left_index", -1)),
            int(row.get("final_right_index", -1)),
            str(row.get("candidate_id", "")),
        ),
    )


def build_single_interface_pdb(
    row: Dict[str, Any],
    pdb_dir: Path,
    output_pdb: Path,
    patch_cutoff: float,
    include_full_receptor: bool,
    include_source_full_chain: bool,
) -> Dict[str, Any]:
    structure_path = resolve_structure_path(pdb_dir, row)
    st = load_structure(structure_path)
    model = st[0]

    rec_chain = find_chain_by_name(model, row["receptor_chain_id"])
    pep_chain = find_chain_by_name(model, row["peptide_source_chain_id"])
    rec_res_all = chain_residues_with_alignment(rec_chain)
    pep_res_all = chain_residues_with_alignment(pep_chain)

    left_idx = int(row["final_left_index"])
    right_idx = int(row["final_right_index"])
    if left_idx < 0 or right_idx >= len(pep_res_all) or left_idx > right_idx:
        raise ValueError(
            f"Invalid peptide window for candidate {row.get('candidate_id', '')}: {left_idx}-{right_idx}"
        )

    peptide_window = [res for res, _coords in pep_res_all[left_idx:right_idx + 1]]
    patch_residues = collect_patch_residues(rec_res_all, peptide_window, patch_cutoff)
    receptor_full = [res for res, _coords in rec_res_all]
    source_full = [res for res, _coords in pep_res_all]

    lines: List[str] = [
        "REMARK 900 PEPTIDECLIP PROTEIN-INTERFACE EXPORT",
        f"REMARK 900 PDB_ID {row.get('pdb_id', '')}",
        f"REMARK 900 CANDIDATE_ID {row.get('candidate_id', '')}",
        f"REMARK 900 PARENT_TASK_ID {row.get('parent_task_id', '')}",
        f"REMARK 900 RECEPTOR_CHAIN {row.get('receptor_chain_id', '')}",
        f"REMARK 900 PEPTIDE_SOURCE_CHAIN {row.get('peptide_source_chain_id', '')}",
        f"REMARK 900 FINAL_LEFT_INDEX {row.get('final_left_index', '')}",
        f"REMARK 900 FINAL_RIGHT_INDEX {row.get('final_right_index', '')}",
        f"REMARK 900 PEPTIDE_LENGTH {row.get('peptide_length', '')}",
        f"REMARK 900 PATCH_CUTOFF {patch_cutoff:.2f}",
        f"REMARK 900 CONTACT_COVERAGE_6A {row.get('contact_coverage_6A', '')}",
        f"REMARK 900 RBSA_PROXY {row.get('rBSA_proxy', row.get('rBSA_raw', ''))}",
        "REMARK 900 CHAIN F receptor_full_chain" if include_full_receptor else "REMARK 900 CHAIN F omitted",
        "REMARK 900 CHAIN L peptide_source_full_chain" if include_source_full_chain else "REMARK 900 CHAIN L omitted",
        "REMARK 900 CHAIN X receptor_local_patch",
        "REMARK 900 CHAIN P candidate_peptide_window",
    ]

    serial = 1
    if include_full_receptor:
        serial = write_residue_block(lines, receptor_full, chain_id="F", serial_start=serial)
    if include_source_full_chain:
        serial = write_residue_block(lines, source_full, chain_id="L", serial_start=serial)
    serial = write_residue_block(lines, patch_residues, chain_id="X", serial_start=serial)
    serial = write_residue_block(lines, peptide_window, chain_id="P", serial_start=serial)
    lines.append("END")

    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    output_pdb.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "candidate_id": row.get("candidate_id", ""),
        "parent_task_id": row.get("parent_task_id", ""),
        "pdb_id": row.get("pdb_id", ""),
        "receptor_chain_id": row.get("receptor_chain_id", ""),
        "peptide_source_chain_id": row.get("peptide_source_chain_id", ""),
        "peptide_length": int(row.get("peptide_length", 0)),
        "contact_coverage_6A": float(row.get("contact_coverage_6A", 0.0)),
        "rBSA_proxy": float(row.get("rBSA_proxy", row.get("rBSA_raw", 0.0))),
        "patch_num_residues": len(patch_residues),
        "peptide_residue_ids": [residue_id_string(res) for res in peptide_window],
        "patch_residue_ids": [residue_id_string(res) for res in patch_residues],
        "output_pdb": str(output_pdb),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one protein's peptide interfaces into separate PDB files, one interface per file."
    )
    parser.add_argument("--input_jsonl", type=str, required=True, help="Input JSONL from Step5/Step6C/etc.")
    parser.add_argument("--pdb_dir", type=str, required=True, help="Directory containing source CIF files")
    parser.add_argument("--pdb_id", type=str, required=True, help="Target PDB ID")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save per-interface PDB files")
    parser.add_argument("--receptor_chain_id", type=str, default="", help="Optional receptor chain filter")
    parser.add_argument("--peptide_source_chain_id", type=str, default="", help="Optional peptide source chain filter")
    parser.add_argument("--patch_cutoff", type=float, default=6.0)
    parser.add_argument("--max_candidates", type=int, default=20, help="Cap the number of exported interfaces")
    parser.add_argument("--sort_by", type=str, choices=["rbsa", "coverage", "length"], default="rbsa")
    parser.add_argument("--include_full_receptor", action="store_true", help="Include the receptor full chain as chain F")
    parser.add_argument("--include_source_full_chain", action="store_true", help="Include the original peptide-source full chain as chain L")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_jsonl = Path(args.input_jsonl)
    pdb_dir = Path(args.pdb_dir)
    output_dir = Path(args.output_dir)

    rows = load_jsonl_rows(input_jsonl)
    rows = filter_rows(
        rows,
        pdb_id=args.pdb_id,
        receptor_chain_id=args.receptor_chain_id,
        peptide_source_chain_id=args.peptide_source_chain_id,
    )
    if not rows:
        raise ValueError("No matching rows found for the requested protein filters")

    rows = sort_rows(rows)
    if args.sort_by == "rbsa":
        rows = sorted(rows, key=lambda r: (-float(r.get("rBSA_proxy", r.get("rBSA_raw", 0.0))), -float(r.get("contact_coverage_6A", 0.0))))
    elif args.sort_by == "coverage":
        rows = sorted(rows, key=lambda r: (-float(r.get("contact_coverage_6A", 0.0)), -float(r.get("rBSA_proxy", r.get("rBSA_raw", 0.0)))))
    elif args.sort_by == "length":
        rows = sorted(rows, key=lambda r: (-int(r.get("peptide_length", 0)), -float(r.get("rBSA_proxy", r.get("rBSA_raw", 0.0)))))

    if args.max_candidates > 0:
        rows = rows[: args.max_candidates]

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []

    protein_token = sanitize_token(args.pdb_id)
    for idx, row in enumerate(rows, start=1):
        rec = sanitize_token(row.get("receptor_chain_id", "R"))
        pep = sanitize_token(row.get("peptide_source_chain_id", "P"))
        cand = sanitize_token(row.get("candidate_id", ""))[:12]
        out_name = f"{protein_token}_{idx:02d}_{rec}_{pep}_{cand}.pdb"
        out_path = output_dir / out_name
        info = build_single_interface_pdb(
            row=row,
            pdb_dir=pdb_dir,
            output_pdb=out_path,
            patch_cutoff=float(args.patch_cutoff),
            include_full_receptor=bool(args.include_full_receptor),
            include_source_full_chain=bool(args.include_source_full_chain),
        )
        manifest.append(info)
        print(f"[DONE] wrote {out_path}", flush=True)

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    readme_path = output_dir / "README.txt"
    readme_path.write_text(
        (
            "One PDB = one peptide interface.\n\n"
            "Chain semantics:\n"
            "  F = receptor full chain (only if --include_full_receptor)\n"
            "  L = original peptide-source full chain (only if --include_source_full_chain)\n"
            "  X = receptor local patch around the exported peptide window\n"
            "  P = exported peptide window\n\n"
            "Recommended first look in PyMOL:\n"
            "  hide everything, all\n"
            "  show cartoon, chain F\n"
            "  color gray70, chain F\n"
            "  show cartoon, chain L\n"
            "  color yellow, chain L\n"
            "  show sticks, chain X\n"
            "  color cyan, chain X\n"
            "  show sticks, chain P\n"
            "  color orange, chain P\n"
            "  orient\n"
            "  zoom visible, 6\n"
        ),
        encoding="utf-8",
    )

    print(f"[DONE] manifest -> {manifest_path}", flush=True)
    print(f"[DONE] readme   -> {readme_path}", flush=True)


if __name__ == "__main__":
    main()
