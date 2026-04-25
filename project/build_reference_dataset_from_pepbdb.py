from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import shutil
import time
import traceback
import uuid
from pathlib import Path
from typing import List, Tuple, Dict, Any

import gemmi
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:
    import freesasa
except ImportError:
    freesasa = None


STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "MSE", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL"
}


# =========================================================
# 基础工具函数
# =========================================================
def is_standard_residue(res: gemmi.Residue) -> bool:
    return res.name in STANDARD_AA


def get_heavy_atoms(residue: gemmi.Residue) -> List[gemmi.Atom]:
    return [a for a in residue if a.element.name != "H"]


def load_first_model(path: Path) -> gemmi.Model:
    st = gemmi.read_structure(str(path))
    if len(st) == 0:
        raise ValueError(f"Empty structure: {path}")
    return st[0]


def collect_standard_residues_all_chains(model: gemmi.Model) -> List[Tuple[str, gemmi.Residue]]:
    """
    收集模型中所有链的标准蛋白残基。
    返回 (chain_name, residue) 列表，保留链信息。
    适用于 receptor，因为真实口袋可能由多条链共同组成。
    """
    out: List[Tuple[str, gemmi.Residue]] = []
    for chain in model:
        for res in chain:
            if is_standard_residue(res):
                out.append((chain.name, res))
    return out


def collect_standard_residues_single_chain(
    model: gemmi.Model,
    preferred_chain_name: str = "",
) -> Tuple[List[Tuple[str, gemmi.Residue]], str, int]:
    """
    选择 peptide 侧的单目标链：
    1) 若 preferred_chain_name 存在且匹配到有效蛋白链，则优先使用
    2) 否则回退到标准残基数最多的那条链

    返回:
    - [(chain_name, residue), ...]
    - selected_chain_name
    - chain_count
    """
    chain_candidates: List[Tuple[str, List[Tuple[str, gemmi.Residue]]]] = []

    for chain in model:
        residues = [(chain.name, res) for res in chain if is_standard_residue(res)]
        if residues:
            chain_candidates.append((chain.name, residues))

    chain_count = len(list(model))

    if not chain_candidates:
        return [], "", chain_count

    if preferred_chain_name:
        for chain_name, residues in chain_candidates:
            if chain_name == preferred_chain_name:
                return residues, chain_name, chain_count

    selected_chain_name, selected_residues = max(chain_candidates, key=lambda x: len(x[1]))
    return selected_residues, selected_chain_name, chain_count


def residue_id_string(chain_name: str, res: gemmi.Residue) -> str:
    num = int(res.seqid.num)
    icode = str(res.seqid.icode).strip()
    if icode:
        return f"{chain_name}:{num}{icode}:{res.name}"
    return f"{chain_name}:{num}:{res.name}"


def extract_coords_from_chain_residues(chain_residues: List[Tuple[str, gemmi.Residue]]) -> np.ndarray:
    coords = []
    for _, res in chain_residues:
        for atom in get_heavy_atoms(res):
            coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
    if not coords:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(coords, dtype=np.float32)


# =========================================================
# 物理计算核心
# =========================================================
def compute_pocket_num_residues(
    peptide_chain_residues: List[Tuple[str, gemmi.Residue]],
    receptor_chain_residues: List[Tuple[str, gemmi.Residue]],
    cutoff: float = 6.0,
) -> Tuple[int, List[str]]:
    pep_coords = extract_coords_from_chain_residues(peptide_chain_residues)
    if len(pep_coords) == 0:
        return 0, []

    rec_coords_list = []
    rec_res_map = []

    for i, (_, res) in enumerate(receptor_chain_residues):
        for atom in get_heavy_atoms(res):
            rec_coords_list.append([atom.pos.x, atom.pos.y, atom.pos.z])
            rec_res_map.append(i)

    if not rec_coords_list:
        return 0, []

    rec_coords = np.asarray(rec_coords_list, dtype=np.float32)
    tree = cKDTree(rec_coords)
    hits = tree.query_ball_point(pep_coords, r=cutoff)

    pocket_atom_indices = set()
    for h in hits:
        pocket_atom_indices.update(h)

    pocket_res_indices = sorted(set(rec_res_map[i] for i in pocket_atom_indices))
    pocket_chain_residues = [receptor_chain_residues[i] for i in pocket_res_indices]
    pocket_ids = [residue_id_string(chain_name, res) for chain_name, res in pocket_chain_residues]

    return len(pocket_chain_residues), pocket_ids


def build_safe_single_chain_pdb_from_chain_residues(
    chain_residues: List[Tuple[str, gemmi.Residue]],
    out_path: Path,
    forced_chain_name: str,
) -> None:
    """
    将给定残基集合写成单链 PDB，并连续重编号。
    用于 FreeSASA 的稳定几何对象，不表示真实拓扑链关系。
    """
    if not chain_residues:
        raise ValueError(f"No residues to write: {out_path}")

    st_out = gemmi.Structure()
    st_out.name = out_path.stem
    model_out = gemmi.Model("1")
    chain_out = gemmi.Chain(forced_chain_name)

    for i, (_, res) in enumerate(chain_residues, start=1):
        cloned_res = res.clone()
        cloned_res.seqid = gemmi.SeqId(str(i))
        chain_out.add_residue(cloned_res)

    model_out.add_chain(chain_out)
    st_out.add_model(model_out)
    st_out.write_pdb(str(out_path))


def build_safe_complex_pdb_from_residues(
    peptide_chain_residues: List[Tuple[str, gemmi.Residue]],
    receptor_chain_residues: List[Tuple[str, gemmi.Residue]],
    out_path: Path,
) -> None:
    pep_safe = out_path.parent / f"{out_path.stem}_pep_only_tmp.pdb"
    rec_safe = out_path.parent / f"{out_path.stem}_rec_only_tmp.pdb"

    try:
        build_safe_single_chain_pdb_from_chain_residues(peptide_chain_residues, pep_safe, "P")
        build_safe_single_chain_pdb_from_chain_residues(receptor_chain_residues, rec_safe, "R")

        st_pep = gemmi.read_structure(str(pep_safe))
        st_rec = gemmi.read_structure(str(rec_safe))

        st_comp = gemmi.Structure()
        st_comp.name = "merged_complex"
        model = gemmi.Model("1")

        for chain in st_pep[0]:
            model.add_chain(chain.clone())
        for chain in st_rec[0]:
            model.add_chain(chain.clone())

        st_comp.add_model(model)
        st_comp.write_pdb(str(out_path))
    finally:
        if pep_safe.exists():
            pep_safe.unlink()
        if rec_safe.exists():
            rec_safe.unlink()


def compute_rbsa_from_residues(
    peptide_chain_residues: List[Tuple[str, gemmi.Residue]],
    receptor_chain_residues: List[Tuple[str, gemmi.Residue]],
    temp_dir: Path,
    entry_name: str,
) -> float:
    """
    使用 FreeSASA 计算:
    rBSA = (SASA_free - SASA_bound) / SASA_free

    注意：
    - peptide 使用单目标链
    - receptor 可为多链聚合
    - 写成单链几何对象仅用于 SASA 稳定计算
    """
    if freesasa is None:
        raise RuntimeError("freesasa not installed.")

    safe_uid = uuid.uuid4().hex[:8]
    complex_pdb = temp_dir / f"{entry_name}_{safe_uid}_comp.pdb"
    pep_safe_pdb = temp_dir / f"{entry_name}_{safe_uid}_pep.pdb"

    try:
        build_safe_single_chain_pdb_from_chain_residues(peptide_chain_residues, pep_safe_pdb, "P")
        pep_structure = freesasa.Structure(str(pep_safe_pdb))
        sasa_free = freesasa.calc(pep_structure).totalArea()

        if sasa_free <= 1e-8:
            return 0.0

        build_safe_complex_pdb_from_residues(peptide_chain_residues, receptor_chain_residues, complex_pdb)
        complex_structure = freesasa.Structure(str(complex_pdb))
        complex_result = freesasa.calc(complex_structure)

        # FreeSASA Python 接口要求传入“命令字符串列表”，不是 dict
        selection = freesasa.selectArea(
            ["peptide, chain P"],
            complex_structure,
            complex_result,
        )
        sasa_bound = selection["peptide"]

        rbsa = (sasa_free - sasa_bound) / sasa_free
        return float(max(0.0, min(rbsa, 1.0)))

    finally:
        if complex_pdb.exists():
            complex_pdb.unlink()
        if pep_safe_pdb.exists():
            pep_safe_pdb.unlink()


def compute_rbsa_simple_proxy(
    peptide_chain_residues: List[Tuple[str, gemmi.Residue]],
    receptor_chain_residues: List[Tuple[str, gemmi.Residue]],
    cutoff: float = 6.0,
) -> float:
    """
    当 FreeSASA 不可用或失败时，用 peptide 残基接触覆盖率近似 rBSA_proxy。
    """
    pep_coords_by_res = []
    for _, res in peptide_chain_residues:
        coords = [[atom.pos.x, atom.pos.y, atom.pos.z] for atom in get_heavy_atoms(res)]
        pep_coords_by_res.append(np.asarray(coords, dtype=np.float32))

    rec_coords = extract_coords_from_chain_residues(receptor_chain_residues)
    if len(rec_coords) == 0 or len(pep_coords_by_res) == 0:
        return 0.0

    rec_tree = cKDTree(rec_coords)
    contact_flags = []

    for coords in pep_coords_by_res:
        if len(coords) == 0:
            contact_flags.append(False)
            continue
        dists, _ = rec_tree.query(coords, k=1)
        contact_flags.append(float(np.min(dists)) < cutoff)

    return float(sum(contact_flags) / len(contact_flags)) if contact_flags else 0.0


# =========================================================
# 单条样本处理
# =========================================================
def process_one_complex_worker(payload: Tuple[str, str, float]) -> Dict[str, Any]:
    entry_dir_str, temp_dir_str, pocket_cutoff = payload
    entry_dir = Path(entry_dir_str)
    temp_dir = Path(temp_dir_str)

    try:
        peptide_pdb = entry_dir / "peptide.pdb"
        receptor_pdb = entry_dir / "receptor.pdb"

        if not peptide_pdb.exists() or not receptor_pdb.exists():
            raise FileNotFoundError(f"Missing peptide.pdb / receptor.pdb in {entry_dir}")

        pep_model = load_first_model(peptide_pdb)
        rec_model = load_first_model(receptor_pdb)

        entry_name = entry_dir.name
        pdb_id = entry_name.split("_")[0]
        peptide_chain_hint = entry_name.split("_", 1)[1] if "_" in entry_name else ""

        # peptide：单目标链
        peptide_chain_residues, peptide_chain_selected, peptide_chain_count = collect_standard_residues_single_chain(
            pep_model,
            preferred_chain_name=peptide_chain_hint,
        )

        # receptor：全链聚合
        receptor_chain_residues = collect_standard_residues_all_chains(rec_model)
        receptor_chain_count = len(list(rec_model))

        if not peptide_chain_residues:
            raise ValueError("No peptide standard residues found.")
        if not receptor_chain_residues:
            raise ValueError("No receptor standard residues found.")

        pocket_num_residues, pocket_residue_ids = compute_pocket_num_residues(
            peptide_chain_residues,
            receptor_chain_residues,
            cutoff=pocket_cutoff,
        )

        rbsa_fallback_reason = ""
        if freesasa is not None:
            try:
                rbsa_proxy = compute_rbsa_from_residues(
                    peptide_chain_residues,
                    receptor_chain_residues,
                    temp_dir,
                    entry_name=entry_name,
                )
                rbsa_source = "freesasa"
            except Exception as e:
                rbsa_proxy = compute_rbsa_simple_proxy(
                    peptide_chain_residues,
                    receptor_chain_residues,
                    cutoff=pocket_cutoff,
                )
                rbsa_source = "contact_coverage_proxy_fallback"
                rbsa_fallback_reason = f"{type(e).__name__}: {str(e)}"
        else:
            rbsa_proxy = compute_rbsa_simple_proxy(
                peptide_chain_residues,
                receptor_chain_residues,
                cutoff=pocket_cutoff,
            )
            rbsa_source = "contact_coverage_proxy"
            rbsa_fallback_reason = "freesasa_not_installed"

        peptide_length = int(len(peptide_chain_residues))
        num_receptor_residues_total = int(len(receptor_chain_residues))

        # ==========================================
        # 真实参考集过滤：只保留真正的多肽复合物
        # 当前最小规则：4 <= peptide_length <= 30
        # ==========================================
        if peptide_length < 4 or peptide_length > 30:
            raise ValueError(f"Peptide length out of range: {peptide_length}")

        return {
            "ok": True,
            "entry_name": entry_name,
            "row": {
                "entry_name": entry_name,
                "pdb_id": pdb_id,
                "peptide_chain_hint": peptide_chain_hint,
                "peptide_chain_selected": peptide_chain_selected,
                "peptide_chain_count": peptide_chain_count,
                "receptor_chain_count": receptor_chain_count,
                "peptide_length": peptide_length,
                "num_receptor_residues_total": num_receptor_residues_total,
                "pocket_cutoff": float(pocket_cutoff),
                "pocket_num_residues": int(pocket_num_residues),
                "pocket_residue_ids": json.dumps(pocket_residue_ids, ensure_ascii=False),
                "rBSA_proxy": float(rbsa_proxy),
                "rBSA_source": rbsa_source,
                "rBSA_fallback_reason": rbsa_fallback_reason,
            },
        }

    except Exception as e:
        return {
            "ok": False,
            "entry_name": Path(entry_dir_str).name,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(limit=2),
        }


# =========================================================
# 主程序
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Build real reference dataset from PepBDB (Robust & Auditable)")
    parser.add_argument("--pepbdb_root", type=str, required=True, help="PepBDB root dir")
    parser.add_argument("--out_csv", type=str, required=True, help="Output ref_dataset.csv")
    parser.add_argument("--out_errors_jsonl", type=str, required=True, help="Output error log JSONL")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of entries")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2), help="Number of CPU cores to use")
    parser.add_argument("--pocket_cutoff", type=float, default=6.0, help="Pocket residue distance cutoff (Å)")
    args = parser.parse_args()

    pepbdb_root = Path(args.pepbdb_root)
    out_csv = Path(args.out_csv)
    out_errors = Path(args.out_errors_jsonl)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_errors.parent.mkdir(parents=True, exist_ok=True)

    entry_dirs = sorted([p for p in pepbdb_root.iterdir() if p.is_dir()])
    if args.limit > 0:
        entry_dirs = entry_dirs[:args.limit]

    temp_dir = out_csv.parent / "_tmp_rbsa_workers"
    temp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print(f"[START] Processing PepBDB Database (Workers: {args.workers})", flush=True)
    print(f"[START] Target entries: {len(entry_dirs)}", flush=True)
    print(f"[START] Pocket cutoff : {args.pocket_cutoff}", flush=True)
    print("=" * 80, flush=True)

    rows = []
    error_count = 0
    rbsa_source_counter: Dict[str, int] = {}
    start_time = time.time()

    payloads = [(str(ed), str(temp_dir), float(args.pocket_cutoff)) for ed in entry_dirs]

    with out_errors.open("w", encoding="utf-8") as ferr, mp.Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(process_one_complex_worker, payloads, chunksize=20), 1):
            if res["ok"]:
                row = res["row"]
                rows.append(row)
                src = row["rBSA_source"]
                rbsa_source_counter[src] = rbsa_source_counter.get(src, 0) + 1
            else:
                error_count += 1
                ferr.write(json.dumps(res, ensure_ascii=False) + "\n")

            if i % 500 == 0:
                elapsed = (time.time() - start_time) / 60.0
                print(
                    f"[PROGRESS] {i}/{len(entry_dirs)} | "
                    f"Success: {len(rows)} | Errors: {error_count} | "
                    f"Time: {elapsed:.1f}m",
                    flush=True
                )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    try:
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()
        elif temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass

    print("=" * 80, flush=True)
    print("[DONE] Reference dataset built successfully.", flush=True)
    print(f"[DONE] Total parsed        : {len(rows)}", flush=True)
    print(f"[DONE] Error count         : {error_count}", flush=True)
    print(f"[DONE] rBSA source counts  : {rbsa_source_counter}", flush=True)
    print(f"[DONE] Saved CSV           : {out_csv}", flush=True)
    print(f"[DONE] Saved error log     : {out_errors}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()