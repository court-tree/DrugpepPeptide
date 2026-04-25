from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import freesasa
import gemmi
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "MSE", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
}


@dataclass
class Step1Config:
    max_resolution: float = 3.5
    min_interface_bsa: float = 200.0
    contact_prefilter_cutoff: float = 6.0


@dataclass
class Step1Result:
    file_name: str
    pdb_id: str
    passed: bool
    reason: str
    method: Optional[str] = None
    resolution: Optional[float] = None
    resolution_missing: bool = False
    assembly_name: Optional[str] = None
    n_assemblies: int = 1
    n_chain_instances: int = 0
    n_contacting_pairs_prefiltered: int = 0
    passing_pair_delta_sasa: Optional[float] = None
    passing_pair: Optional[str] = None


def _safe_float(x):
    if x in (None, "", ".", "?"):
        return None
    s = str(x).strip().strip("'").strip('"')
    if s in ("", ".", "?"):
        return None
    if "(" in s:
        s = s.split("(", 1)[0].strip()
    try:
        return float(s)
    except Exception:
        return None


def _first_valid_value(block, key: str):
    try:
        vals = block.find_values(key)
        if vals is not None:
            for v in vals:
                if v not in (None, "", ".", "?"):
                    return v
    except Exception:
        pass
    try:
        v = block.find_value(key)
        if v not in (None, "", ".", "?"):
            return v
    except Exception:
        pass
    return None


def read_mmcif_metadata(cif_path: Path) -> Tuple[Optional[str], Optional[float], List[str]]:
    doc = gemmi.cif.read_file(str(cif_path))
    block = doc.sole_block()

    methods = []
    try:
        vals = block.find_values("_exptl.method")
        if vals is not None:
            for v in vals:
                s = str(v).strip().strip("'").strip('"')
                if s and s not in {".", "?"}:
                    methods.append(s)
    except Exception:
        pass

    if not methods:
        try:
            single = block.find_value("_exptl.method")
            if single not in (None, "", ".", "?"):
                methods.append(str(single).strip().strip("'").strip('"'))
        except Exception:
            pass

    method = "; ".join(methods) if methods else None

    resolution = None
    for key in (
        "_refine.ls_d_res_high",
        "_reflns.d_resolution_high",
        "_refine_hist.d_res_high",
        "_diffrn_reflns.d_resolution_high",
        "_diffrn_reflns_shell.d_res_high",
        "_reflns_shell.d_res_high",
        "_database_PDB_remark.angstroms",
        "_em_3d_reconstruction.resolution",
    ):
        raw = _first_valid_value(block, key)
        resolution = _safe_float(raw)
        if resolution is not None:
            break

    return method, resolution, [str(cif_path.stem).lower()]


def is_protein_residue(residue: gemmi.Residue) -> bool:
    return residue.name.strip().upper() in STANDARD_AA


def chain_coords(chain: gemmi.Chain) -> np.ndarray:
    pts = []
    for res in chain:
        if not is_protein_residue(res):
            continue
        for atom in res:
            if atom.element.name == "H":
                continue
            pts.append([atom.pos.x, atom.pos.y, atom.pos.z])
    if not pts:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


def write_chain_to_pdb(chain: gemmi.Chain, out_path: Path, new_chain_id: str) -> None:
    lines = []
    serial = 1
    for res in chain:
        if not is_protein_residue(res):
            continue
        for atom in res:
            if atom.element.name == "H":
                continue
            seqid = res.seqid.num if res.seqid.num is not None else 1
            icode = res.seqid.icode if res.seqid.icode != "\x00" else " "
            occ = atom.occ if atom.occ is not None else 1.0
            b = atom.b_iso if atom.b_iso is not None else 0.0
            line = (
                f"ATOM  {serial:5d} {atom.name.rjust(4)} {res.name.rjust(3)} {new_chain_id[:1]}"
                f"{seqid:4d}{icode:1s}   {atom.pos.x:8.3f}{atom.pos.y:8.3f}{atom.pos.z:8.3f}"
                f"{occ:6.2f}{b:6.2f}          {atom.element.name.rjust(2):>2s}"
            )
            lines.append(line)
            serial += 1
    lines.append("TER")
    lines.append("END")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_two_chains_to_pdb(chain_a: gemmi.Chain, chain_b: gemmi.Chain, out_path: Path) -> None:
    lines = []
    serial = 1
    for chain, chain_id in ((chain_a, "A"), (chain_b, "B")):
        for res in chain:
            if not is_protein_residue(res):
                continue
            for atom in res:
                if atom.element.name == "H":
                    continue
                seqid = res.seqid.num if res.seqid.num is not None else 1
                icode = res.seqid.icode if res.seqid.icode != "\x00" else " "
                occ = atom.occ if atom.occ is not None else 1.0
                b = atom.b_iso if atom.b_iso is not None else 0.0
                line = (
                    f"ATOM  {serial:5d} {atom.name.rjust(4)} {res.name.rjust(3)} {chain_id}"
                    f"{seqid:4d}{icode:1s}   {atom.pos.x:8.3f}{atom.pos.y:8.3f}{atom.pos.z:8.3f}"
                    f"{occ:6.2f}{b:6.2f}          {atom.element.name.rjust(2):>2s}"
                )
                lines.append(line)
                serial += 1
        lines.append("TER")
    lines.append("END")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sasa_of_pdb(pdb_path: Path) -> float:
    structure = freesasa.Structure(str(pdb_path))
    result = freesasa.calc(structure)
    return float(result.totalArea())


def pairwise_delta_sasa(chain_a: gemmi.Chain, chain_b: gemmi.Chain) -> float:
    shm_dir = "/dev/shm" if os.path.exists("/dev/shm") else None
    with tempfile.TemporaryDirectory(dir=shm_dir) as tmpdir:
        tmpdir_path = Path(tmpdir)
        path_a = tmpdir_path / "a.pdb"
        path_b = tmpdir_path / "b.pdb"
        path_ab = tmpdir_path / "ab.pdb"
        write_chain_to_pdb(chain_a, path_a, "A")
        write_chain_to_pdb(chain_b, path_b, "B")
        write_two_chains_to_pdb(chain_a, chain_b, path_ab)
        return sasa_of_pdb(path_a) + sasa_of_pdb(path_b) - sasa_of_pdb(path_ab)


def chains_have_contact(coords_a: np.ndarray, coords_b: np.ndarray, cutoff: float) -> bool:
    if len(coords_a) == 0 or len(coords_b) == 0:
        return False
    tree = cKDTree(coords_a)
    hits = tree.query_ball_point(coords_b, r=cutoff)
    return any(len(x) > 0 for x in hits)


def get_native_model_chains(structure: gemmi.Structure):
    if len(structure) == 0:
        return None, [], []
    chains = []
    coords_cache = []
    for chain in structure[0]:
        if len(chain) == 0:
            continue
        coords = chain_coords(chain)
        if len(coords) == 0:
            continue
        chains.append(chain)
        coords_cache.append(coords)
    return "native_assembly", chains, coords_cache


def evaluate_one(cif_path: Path, cfg: Step1Config) -> Step1Result:
    try:
        method, resolution, entry_ids = read_mmcif_metadata(cif_path)
        pdb_id = entry_ids[0]
    except Exception as e:
        return Step1Result(
            file_name=cif_path.name,
            pdb_id=cif_path.stem.lower(),
            passed=False,
            reason=f"metadata_error:{type(e).__name__}",
        )

    resolution_missing = resolution is None
    if method is None or "X-RAY DIFFRACTION" not in method.upper():
        return Step1Result(
            file_name=cif_path.name,
            pdb_id=pdb_id,
            passed=False,
            reason="non_xray_method",
            method=method,
            resolution=resolution,
            resolution_missing=resolution_missing,
        )
    if resolution is not None and resolution > cfg.max_resolution:
        return Step1Result(
            file_name=cif_path.name,
            pdb_id=pdb_id,
            passed=False,
            reason="low_resolution",
            method=method,
            resolution=resolution,
            resolution_missing=resolution_missing,
        )

    try:
        structure = gemmi.read_structure(str(cif_path))
        structure.setup_entities()
    except Exception as e:
        return Step1Result(
            file_name=cif_path.name,
            pdb_id=pdb_id,
            passed=False,
            reason=f"read_structure_error:{type(e).__name__}",
            method=method,
            resolution=resolution,
            resolution_missing=resolution_missing,
        )

    assembly_name, chains, coords_cache = get_native_model_chains(structure)
    if len(chains) < 2:
        return Step1Result(
            file_name=cif_path.name,
            pdb_id=pdb_id,
            passed=False,
            reason="too_few_protein_chains",
            method=method,
            resolution=resolution,
            resolution_missing=resolution_missing,
            assembly_name=assembly_name,
            n_chain_instances=len(chains),
        )

    contacting_pairs = []
    for i in range(len(chains)):
        for j in range(i + 1, len(chains)):
            if chains_have_contact(coords_cache[i], coords_cache[j], cfg.contact_prefilter_cutoff):
                contacting_pairs.append((i, j))

    if not contacting_pairs:
        return Step1Result(
            file_name=cif_path.name,
            pdb_id=pdb_id,
            passed=False,
            reason="no_contacting_pairs_prefilter",
            method=method,
            resolution=resolution,
            resolution_missing=resolution_missing,
            assembly_name=assembly_name,
            n_chain_instances=len(chains),
            n_contacting_pairs_prefiltered=0,
        )

    best_delta = None
    best_pair = None
    for i, j in contacting_pairs:
        try:
            delta = pairwise_delta_sasa(chains[i], chains[j])
        except Exception:
            continue
        if best_delta is None or delta > best_delta:
            best_delta = float(delta)
            best_pair = f"{chains[i].name}__{chains[j].name}"

    if best_delta is None:
        return Step1Result(
            file_name=cif_path.name,
            pdb_id=pdb_id,
            passed=False,
            reason="bsa_calc_failed",
            method=method,
            resolution=resolution,
            resolution_missing=resolution_missing,
            assembly_name=assembly_name,
            n_chain_instances=len(chains),
            n_contacting_pairs_prefiltered=len(contacting_pairs),
        )

    passed = best_delta >= cfg.min_interface_bsa
    return Step1Result(
        file_name=cif_path.name,
        pdb_id=pdb_id,
        passed=passed,
        reason="passed" if passed else "small_interface_bsa",
        method=method,
        resolution=resolution,
        resolution_missing=resolution_missing,
        assembly_name=assembly_name,
        n_chain_instances=len(chains),
        n_contacting_pairs_prefiltered=len(contacting_pairs),
        passing_pair_delta_sasa=best_delta,
        passing_pair=best_pair,
    )


def write_outputs(results: List[Step1Result], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "step1_results.jsonl"
    csv_path = out_dir / "step1_results.csv"
    passed_path = out_dir / "step1_passed_files.txt"
    summary_path = out_dir / "step1_summary.json"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    if results:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
            writer.writeheader()
            for row in results:
                writer.writerow(asdict(row))

    passed_files = [row.file_name for row in results if row.passed]
    passed_path.write_text("\n".join(passed_files) + ("\n" if passed_files else ""), encoding="utf-8")

    reason_counts = {}
    for row in results:
        reason_counts[row.reason] = reason_counts.get(row.reason, 0) + 1
    summary = {
        "n_total": len(results),
        "n_passed": sum(1 for row in results if row.passed),
        "n_failed": sum(1 for row in results if not row.passed),
        "reason_counts": reason_counts,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase1 Step1: structure QC")
    parser.add_argument("--pdb_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--glob", default="*.cif")
    parser.add_argument("--max_resolution", type=float, default=3.5)
    parser.add_argument("--min_interface_bsa", type=float, default=200.0)
    parser.add_argument("--contact_prefilter_cutoff", type=float, default=6.0)
    args = parser.parse_args()

    cfg = Step1Config(
        max_resolution=args.max_resolution,
        min_interface_bsa=args.min_interface_bsa,
        contact_prefilter_cutoff=args.contact_prefilter_cutoff,
    )
    pdb_dir = Path(args.pdb_dir)
    cif_files = sorted(pdb_dir.glob(args.glob))
    results = [evaluate_one(path, cfg) for path in tqdm(cif_files, desc="step1_qc")]
    write_outputs(results, Path(args.output_dir))


if __name__ == "__main__":
    main()
