from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import gemmi
import freesasa
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


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


STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "MSE", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL"
}


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


def read_mmcif_metadata(cif_path: Path) -> tuple[Optional[str], Optional[float], list[str]]:
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

    entry_id = cif_path.stem
    return method, resolution, [str(entry_id).lower()]


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
            pos = atom.pos
            pts.append([pos.x, pos.y, pos.z])

    if not pts:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


def write_chain_to_pdb(chain: gemmi.Chain, out_path: Path, new_chain_id: str):
    lines = []
    serial = 1

    for res in chain:
        if not is_protein_residue(res):
            continue
        for atom in res:
            if atom.element.name == "H":
                continue

            pos = atom.pos
            atom_name = atom.name.rjust(4)
            resname = res.name.rjust(3)
            chain_id = new_chain_id[:1]
            seqid = res.seqid.num if res.seqid.num is not None else 1
            icode = res.seqid.icode if res.seqid.icode != "\x00" else " "
            occ = atom.occ if atom.occ is not None else 1.0
            b = atom.b_iso if atom.b_iso is not None else 0.0
            element = atom.element.name.rjust(2)

            line = (
                f"ATOM  {serial:5d} {atom_name} "
                f"{resname} {chain_id}{seqid:4d}{icode:1s}   "
                f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}"
                f"{occ:6.2f}{b:6.2f}          {element:>2s}"
            )
            lines.append(line)
            serial += 1

    lines.append("TER")
    lines.append("END")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_two_chains_to_pdb(chain_a: gemmi.Chain, chain_b: gemmi.Chain, out_path: Path):
    lines = []
    serial = 1

    def emit_chain(chain: gemmi.Chain, new_chain_id: str, start_serial: int):
        local_lines = []
        serial_now = start_serial

        for res in chain:
            if not is_protein_residue(res):
                continue
            for atom in res:
                if atom.element.name == "H":
                    continue

                pos = atom.pos
                atom_name = atom.name.rjust(4)
                resname = res.name.rjust(3)
                seqid = res.seqid.num if res.seqid.num is not None else 1
                icode = res.seqid.icode if res.seqid.icode != "\x00" else " "
                occ = atom.occ if atom.occ is not None else 1.0
                b = atom.b_iso if atom.b_iso is not None else 0.0
                element = atom.element.name.rjust(2)

                line = (
                    f"ATOM  {serial_now:5d} {atom_name} "
                    f"{resname} {new_chain_id[:1]}{seqid:4d}{icode:1s}   "
                    f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}"
                    f"{occ:6.2f}{b:6.2f}          {element:>2s}"
                )
                local_lines.append(line)
                serial_now += 1

        local_lines.append("TER")
        return local_lines, serial_now

    block_a, serial = emit_chain(chain_a, "A", serial)
    block_b, serial = emit_chain(chain_b, "B", serial)

    lines.extend(block_a)
    lines.extend(block_b)
    lines.append("END")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sasa_of_pdb(pdb_path: Path) -> float:
    structure = freesasa.Structure(str(pdb_path))
    result = freesasa.calc(structure)
    return float(result.totalArea())


def pairwise_delta_sasa(chain_a: gemmi.Chain, chain_b: gemmi.Chain) -> float:
    shm_dir = "/dev/shm" if os.path.exists("/dev/shm") else None

    with tempfile.TemporaryDirectory(dir=shm_dir) as tmpdir:
        tmpdir = Path(tmpdir)
        path_a = tmpdir / "a.pdb"
        path_b = tmpdir / "b.pdb"
        path_ab = tmpdir / "ab.pdb"

        write_chain_to_pdb(chain_a, path_a, "A")
        write_chain_to_pdb(chain_b, path_b, "B")
        write_two_chains_to_pdb(chain_a, chain_b, path_ab)

        sasa_a = sasa_of_pdb(path_a)
        sasa_b = sasa_of_pdb(path_b)
        sasa_ab = sasa_of_pdb(path_ab)

    return sasa_a + sasa_b - sasa_ab


def chains_have_contact(coords_a: np.ndarray, coords_b: np.ndarray, cutoff: float) -> bool:
    if len(coords_a) == 0 or len(coords_b) == 0:
        return False
    tree = cKDTree(coords_a)
    hits = tree.query_ball_point(coords_b, r=cutoff)
    return any(len(x) > 0 for x in hits)


def get_first_assembly_model(structure: gemmi.Structure):
    """
    当前假设：输入已经是下载好的 -assembly1.cif
    因此直接读取第一模型，不再调用 gemmi.make_assembly，
    也不依赖 structure.assemblies 元数据。
    """
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
    except Exception:
        return Step1Result(
            file_name=cif_path.name,
            pdb_id=pdb_id,
            passed=False,
            reason="parse_error",
            method=method,
            resolution=resolution,
            resolution_missing=resolution_missing,
        )

    try:
        assembly_name, chains, coords_cache = get_first_assembly_model(structure)
    except Exception as e:
        return Step1Result(
            file_name=cif_path.name,
            pdb_id=pdb_id,
            passed=False,
            reason=f"assembly_error:{type(e).__name__}",
            method=method,
            resolution=resolution,
            resolution_missing=resolution_missing,
            n_assemblies=1,
        )

    if assembly_name is None or len(chains) == 0:
        return Step1Result(
            file_name=cif_path.name,
            pdb_id=pdb_id,
            passed=False,
            reason="empty_structure",
            method=method,
            resolution=resolution,
            resolution_missing=resolution_missing,
            n_assemblies=1,
        )

    if len(chains) < 2:
        return Step1Result(
            file_name=cif_path.name,
            pdb_id=pdb_id,
            passed=False,
            reason="insufficient_chain_instances",
            method=method,
            resolution=resolution,
            resolution_missing=resolution_missing,
            assembly_name=assembly_name,
            n_assemblies=1,
            n_chain_instances=len(chains),
        )

    contacting_pairs = []
    for i in range(len(chains)):
        for j in range(i + 1, len(chains)):
            try:
                if chains_have_contact(coords_cache[i], coords_cache[j], cfg.contact_prefilter_cutoff):
                    contacting_pairs.append((i, j))
            except Exception:
                continue

    if len(contacting_pairs) == 0:
        return Step1Result(
            file_name=cif_path.name,
            pdb_id=pdb_id,
            passed=False,
            reason="no_contacting_chain_pairs",
            method=method,
            resolution=resolution,
            resolution_missing=resolution_missing,
            assembly_name=assembly_name,
            n_assemblies=1,
            n_chain_instances=len(chains),
            n_contacting_pairs_prefiltered=0,
        )

    passing_bsa = -1.0
    passing_pair = None

    for i, j in contacting_pairs:
        try:
            bsa = pairwise_delta_sasa(chains[i], chains[j])
        except Exception:
            continue

        if bsa > passing_bsa:
            passing_bsa = bsa
            passing_pair = f"{chains[i].name}|{chains[j].name}"

        if passing_bsa >= cfg.min_interface_bsa:
            break

    if passing_bsa < 0:
        return Step1Result(
            file_name=cif_path.name,
            pdb_id=pdb_id,
            passed=False,
            reason="delta_sasa_failed",
            method=method,
            resolution=resolution,
            resolution_missing=resolution_missing,
            assembly_name=assembly_name,
            n_assemblies=1,
            n_chain_instances=len(chains),
            n_contacting_pairs_prefiltered=len(contacting_pairs),
        )

    if passing_bsa < cfg.min_interface_bsa:
        return Step1Result(
            file_name=cif_path.name,
            pdb_id=pdb_id,
            passed=False,
            reason="low_interface_buried_area",
            method=method,
            resolution=resolution,
            resolution_missing=resolution_missing,
            assembly_name=assembly_name,
            n_assemblies=1,
            n_chain_instances=len(chains),
            n_contacting_pairs_prefiltered=len(contacting_pairs),
            passing_pair_delta_sasa=passing_bsa,
            passing_pair=passing_pair,
        )

    return Step1Result(
        file_name=cif_path.name,
        pdb_id=pdb_id,
        passed=True,
        reason="pass",
        method=method,
        resolution=resolution,
        resolution_missing=resolution_missing,
        assembly_name=assembly_name,
        n_assemblies=1,
        n_chain_instances=len(chains),
        n_contacting_pairs_prefiltered=len(contacting_pairs),
        passing_pair_delta_sasa=passing_bsa,
        passing_pair=passing_pair,
    )


def write_outputs(results: list[Step1Result], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "step1_results.jsonl"
    csv_path = output_dir / "step1_results.csv"
    passed_path = output_dir / "step1_passed_files.txt"
    summary_path = output_dir / "step1_summary.json"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in results:
            # 🚀 修复双斜杠转义，使用正常的换行符
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    if results:
        fieldnames = list(asdict(results[0]).keys())
    else:
        fieldnames = list(asdict(Step1Result("", "", False, "")).keys())

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    with passed_path.open("w", encoding="utf-8") as f:
        for r in results:
            if r.passed:
                # 🚀 同上，修复双斜杠
                f.write(r.file_name + "\n")

    reason_counts = {}
    n_passed = 0
    for r in results:
        if r.passed:
            n_passed += 1
        reason_counts[r.reason] = reason_counts.get(r.reason, 0) + 1

    summary = {
        "n_total": len(results),
        "n_passed": n_passed,
        "n_failed": len(results) - n_passed,
        "pass_rate": (n_passed / len(results)) if results else 0.0,
        "reason_counts": reason_counts,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_resolution", type=float, default=3.5)
    parser.add_argument("--min_interface_bsa", type=float, default=200.0)
    parser.add_argument("--contact_prefilter_cutoff", type=float, default=6.0)
    args = parser.parse_args()

    cfg = Step1Config(
        max_resolution=args.max_resolution,
        min_interface_bsa=args.min_interface_bsa,
        contact_prefilter_cutoff=args.contact_prefilter_cutoff,
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    cif_files = sorted(input_dir.glob("*.cif"))

    results = []
    for cif_path in tqdm(cif_files, desc="Step1 QC"):
        try:
            result = evaluate_one(cif_path, cfg)
        except Exception as e:
            result = Step1Result(
                file_name=cif_path.name,
                pdb_id=cif_path.stem.lower(),
                passed=False,
                reason=f"exception:{type(e).__name__}",
            )
        results.append(result)

    write_outputs(results, output_dir)


if __name__ == "__main__":
    main()