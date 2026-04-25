from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
import traceback
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import gemmi
import numpy as np
from scipy.spatial import cKDTree


STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "MSE", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL"
}


@dataclass
class Step2Config:
    # pair-level prefilter: whether two chains are considered interacting at all
    chain_contact_cutoff: float = 6.0
    # directed-sample eligibility: peptide-source chain must contribute enough receptor-contacting residues
    directed_contact_cutoff: float = 6.0
    min_source_contact_residues: int = 2
    # source chain must be long enough to support downstream 8-20 aa window extraction
    min_source_chain_residues: int = 8


@dataclass
class ChainContext:
    chain: gemmi.Chain
    heavy_atom_coords: np.ndarray
    residue_coords: List[np.ndarray]
    residue_ids: List[str]
    n_residues: int
    n_atoms: int


@dataclass
class Step2Task:
    task_id: str
    pdb_id: str
    source_file: str
    assembly_id: str
    chain_pair_id: str
    direction: str

    receptor_chain_id: str
    peptide_source_chain_id: str
    masked_other_chain_ids: List[str]

    receptor_n_residues: int
    receptor_n_atoms: int
    peptide_source_n_residues: int
    peptide_source_n_atoms: int

    pair_min_interchain_distance: float
    source_to_receptor_min_distance: float
    source_contact_residue_count: int
    source_contact_residue_fraction: float
    source_contact_residue_ids: List[str]


_WORKER_CFG: Optional[Step2Config] = None


def is_protein_residue(resname: str) -> bool:
    return resname.strip().upper() in STANDARD_AA


def residue_id_string(residue: gemmi.Residue) -> str:
    num = residue.seqid.num
    icode = residue.seqid.icode
    num_str = str(int(num)) if num is not None else "?"
    icode_str = str(icode).strip() if icode is not None else ""
    return f"{num_str}{icode_str}:{residue.name}"


def residue_heavy_atom_coords(residue: gemmi.Residue) -> np.ndarray:
    coords: List[List[float]] = []
    for atom in residue:
        if atom.element.name == "H":
            continue
        coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
    if not coords:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(coords, dtype=float)


def build_chain_context(chain: gemmi.Chain) -> Optional[ChainContext]:
    residue_coords: List[np.ndarray] = []
    residue_ids: List[str] = []
    atom_blocks: List[np.ndarray] = []

    for residue in chain:
        if not is_protein_residue(residue.name):
            continue
        coords = residue_heavy_atom_coords(residue)
        residue_coords.append(coords)
        residue_ids.append(residue_id_string(residue))
        if len(coords) > 0:
            atom_blocks.append(coords)

    n_residues = len(residue_coords)
    if n_residues == 0:
        return None

    heavy_atom_coords = np.concatenate(atom_blocks, axis=0) if atom_blocks else np.zeros((0, 3), dtype=float)
    return ChainContext(
        chain=chain,
        heavy_atom_coords=heavy_atom_coords,
        residue_coords=residue_coords,
        residue_ids=residue_ids,
        n_residues=n_residues,
        n_atoms=int(len(heavy_atom_coords)),
    )


def load_structure(cif_path: Path) -> gemmi.Structure:
    st = gemmi.read_structure(str(cif_path))
    st.setup_entities()
    return st


def get_model_chain_contexts(structure: gemmi.Structure) -> List[ChainContext]:
    if len(structure) == 0:
        return []

    model = structure[0]
    out: List[ChainContext] = []
    for chain in model:
        ctx = build_chain_context(chain)
        if ctx is not None:
            out.append(ctx)
    return out


def chains_have_contact(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    cutoff: float,
) -> Tuple[bool, float]:
    if len(coords_a) == 0 or len(coords_b) == 0:
        return False, float("inf")

    tree = cKDTree(coords_b)
    dists, _ = tree.query(coords_a, k=1)
    min_dist = float(np.min(dists))
    return min_dist <= cutoff, min_dist


def count_source_contact_residues(
    source_residue_coords: List[np.ndarray],
    source_residue_ids: List[str],
    receptor_coords: np.ndarray,
    cutoff: float,
) -> Tuple[int, float, List[str], float]:
    if len(receptor_coords) == 0:
        return 0, 0.0, [], float("inf")

    receptor_tree = cKDTree(receptor_coords)
    n_contact = 0
    contact_ids: List[str] = []
    global_min = float("inf")

    for residue_id, coords in zip(source_residue_ids, source_residue_coords):
        if len(coords) == 0:
            continue
        dists, _ = receptor_tree.query(coords, k=1)
        dmin = float(np.min(dists))
        if dmin < global_min:
            global_min = dmin
        if dmin <= cutoff:
            n_contact += 1
            contact_ids.append(residue_id)

    n_total = len(source_residue_coords)
    frac = (n_contact / n_total) if n_total > 0 else 0.0
    return n_contact, frac, contact_ids, global_min


def make_pair_id(chain_a_name: str, chain_b_name: str) -> str:
    return "__".join(sorted([chain_a_name, chain_b_name]))


def make_directed_task(
    pdb_id: str,
    source_file: str,
    assembly_id: str,
    pair_id: str,
    pair_min_dist: float,
    receptor_ctx: ChainContext,
    source_ctx: ChainContext,
    masked_other_chain_ids: List[str],
    cfg: Step2Config,
) -> Optional[Step2Task]:
    if source_ctx.n_residues < cfg.min_source_chain_residues:
        return None

    n_contact_residues, contact_frac, contact_ids, source_min_dist = count_source_contact_residues(
        source_residue_coords=source_ctx.residue_coords,
        source_residue_ids=source_ctx.residue_ids,
        receptor_coords=receptor_ctx.heavy_atom_coords,
        cutoff=cfg.directed_contact_cutoff,
    )

    if n_contact_residues < cfg.min_source_contact_residues:
        return None

    return Step2Task(
        task_id=str(uuid.uuid4()),
        pdb_id=pdb_id,
        source_file=source_file,
        assembly_id=assembly_id,
        chain_pair_id=pair_id,
        direction=f"{source_ctx.chain.name}_as_peptide__{receptor_ctx.chain.name}_as_receptor",
        receptor_chain_id=receptor_ctx.chain.name,
        peptide_source_chain_id=source_ctx.chain.name,
        masked_other_chain_ids=masked_other_chain_ids,
        receptor_n_residues=receptor_ctx.n_residues,
        receptor_n_atoms=receptor_ctx.n_atoms,
        peptide_source_n_residues=source_ctx.n_residues,
        peptide_source_n_atoms=source_ctx.n_atoms,
        pair_min_interchain_distance=pair_min_dist,
        source_to_receptor_min_distance=source_min_dist,
        source_contact_residue_count=n_contact_residues,
        source_contact_residue_fraction=contact_frac,
        source_contact_residue_ids=contact_ids,
    )


def build_bidirectional_tasks(cif_path: Path, cfg: Step2Config) -> List[Step2Task]:
    structure = load_structure(cif_path)
    chain_contexts = get_model_chain_contexts(structure)
    pdb_id = cif_path.stem.lower()
    assembly_id = "native_file"

    if len(chain_contexts) < 2:
        return []

    tasks: List[Step2Task] = []
    all_chain_names = [ctx.chain.name for ctx in chain_contexts]

    for i in range(len(chain_contexts)):
        for j in range(i + 1, len(chain_contexts)):
            ctx_a = chain_contexts[i]
            ctx_b = chain_contexts[j]

            ok, pair_min_dist = chains_have_contact(
                ctx_a.heavy_atom_coords,
                ctx_b.heavy_atom_coords,
                cfg.chain_contact_cutoff,
            )
            if not ok:
                continue

            pair_id = make_pair_id(ctx_a.chain.name, ctx_b.chain.name)
            masked = [x for x in all_chain_names if x not in {ctx_a.chain.name, ctx_b.chain.name}]

            task_ab = make_directed_task(
                pdb_id=pdb_id,
                source_file=cif_path.name,
                assembly_id=assembly_id,
                pair_id=pair_id,
                pair_min_dist=pair_min_dist,
                receptor_ctx=ctx_a,
                source_ctx=ctx_b,
                masked_other_chain_ids=masked,
                cfg=cfg,
            )
            if task_ab is not None:
                tasks.append(task_ab)

            task_ba = make_directed_task(
                pdb_id=pdb_id,
                source_file=cif_path.name,
                assembly_id=assembly_id,
                pair_id=pair_id,
                pair_min_dist=pair_min_dist,
                receptor_ctx=ctx_b,
                source_ctx=ctx_a,
                masked_other_chain_ids=masked,
                cfg=cfg,
            )
            if task_ba is not None:
                tasks.append(task_ba)

    return tasks


def init_worker(cfg: Step2Config) -> None:
    global _WORKER_CFG
    _WORKER_CFG = cfg


def worker(cif_path_str: str) -> Dict[str, Any]:
    cif_path = Path(cif_path_str)

    try:
        if _WORKER_CFG is None:
            raise RuntimeError("Worker config is not initialized.")

        tasks = build_bidirectional_tasks(cif_path, _WORKER_CFG)
        return {
            "ok": True,
            "source_file": cif_path.name,
            "tasks": [asdict(t) for t in tasks],
            "error": None,
        }

    except Exception as e:
        return {
            "ok": False,
            "source_file": cif_path.name,
            "tasks": [],
            "error": {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(limit=3),
            },
        }


def read_passed_files(step1_dir: Path, pdb_dir: Path) -> List[Path]:
    passed_file = step1_dir / "step1_passed_files.txt"
    if not passed_file.exists():
        raise FileNotFoundError(f"Missing: {passed_file}")

    files: List[Path] = []
    for line in passed_file.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue

        p = Path(s)

        if p.is_absolute() and p.exists():
            files.append(p)
            continue

        p1 = step1_dir / s
        if p1.exists():
            files.append(p1)
            continue

        p2 = pdb_dir / s
        if p2.exists():
            files.append(p2)
            continue

        p3 = pdb_dir / p.name
        if p3.exists():
            files.append(p3)
            continue

        print(f"[WARN] Missing CIF: {s}", flush=True)

    return files


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="PeptideCLIP Phase-1 Step-2 Task Generation (improved directed-task version)"
    )
    parser.add_argument("--step1_dir", type=str, required=True)
    parser.add_argument("--pdb_dir", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--error_jsonl", type=str, default="")
    parser.add_argument("--chain_contact_cutoff", type=float, default=6.0)
    parser.add_argument("--directed_contact_cutoff", type=float, default=6.0)
    parser.add_argument("--min_source_contact_residues", type=int, default=2)
    parser.add_argument("--min_source_chain_residues", type=int, default=8)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    parser.add_argument("--chunksize", type=int, default=50)
    parser.add_argument("--progress_every", type=int, default=100)
    args = parser.parse_args()

    cfg = Step2Config(
        chain_contact_cutoff=args.chain_contact_cutoff,
        directed_contact_cutoff=args.directed_contact_cutoff,
        min_source_contact_residues=args.min_source_contact_residues,
        min_source_chain_residues=args.min_source_chain_residues,
    )
    step1_dir = Path(args.step1_dir)
    pdb_dir = Path(args.pdb_dir)
    output_jsonl = Path(args.output_jsonl)
    error_jsonl = Path(args.error_jsonl) if args.error_jsonl else None

    start_time = time.time()
    pid = os.getpid()

    print("=" * 80, flush=True)
    print(f"[START] PID={pid}", flush=True)
    print(f"[START] step1_dir                  = {step1_dir}", flush=True)
    print(f"[START] pdb_dir                    = {pdb_dir}", flush=True)
    print(f"[START] output_jsonl               = {output_jsonl}", flush=True)
    print(f"[START] error_jsonl                = {error_jsonl}", flush=True)
    print(f"[START] chain_contact_cutoff       = {args.chain_contact_cutoff}", flush=True)
    print(f"[START] directed_contact_cutoff    = {args.directed_contact_cutoff}", flush=True)
    print(f"[START] min_source_contact_residues= {args.min_source_contact_residues}", flush=True)
    print(f"[START] min_source_chain_residues  = {args.min_source_chain_residues}", flush=True)
    print(f"[START] workers                    = {args.workers}", flush=True)
    print(f"[START] chunksize                  = {args.chunksize}", flush=True)
    print(f"[START] progress_every             = {args.progress_every}", flush=True)
    print("=" * 80, flush=True)

    cif_files = read_passed_files(step1_dir, pdb_dir)
    if not cif_files:
        print("[STOP] No valid CIF files found. Check your paths.", flush=True)
        return

    print(f"[INFO] Found {len(cif_files)} valid complexes.", flush=True)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if error_jsonl is not None:
        error_jsonl.parent.mkdir(parents=True, exist_ok=True)

    total_tasks = 0
    processed_files = 0
    error_count = 0

    f_err = None
    try:
        with output_jsonl.open("w", encoding="utf-8") as f_out:
            if error_jsonl is not None:
                f_err = error_jsonl.open("w", encoding="utf-8")

            with mp.Pool(
                processes=args.workers,
                initializer=init_worker,
                initargs=(cfg,)
            ) as pool:
                jobs = (str(p) for p in cif_files)

                for result in pool.imap_unordered(worker, jobs, chunksize=args.chunksize):
                    processed_files += 1

                    if result["ok"]:
                        task_dicts = result["tasks"]
                        total_tasks += len(task_dicts)

                        for task_dict in task_dicts:
                            f_out.write(json.dumps(task_dict, ensure_ascii=False) + "\n")
                    else:
                        error_count += 1
                        if f_err is not None:
                            f_err.write(json.dumps(result, ensure_ascii=False) + "\n")

                    if processed_files % args.progress_every == 0:
                        elapsed = time.time() - start_time
                        speed = processed_files / elapsed if elapsed > 0 else 0.0
                        print(
                            f"[PROGRESS] {processed_files}/{len(cif_files)} | "
                            f"tasks={total_tasks} | errors={error_count} | "
                            f"elapsed={elapsed/60:.1f} min | speed={speed:.2f} files/s",
                            flush=True
                        )

        elapsed = time.time() - start_time
        print("=" * 80, flush=True)
        print("[DONE] Step-2 finished successfully.", flush=True)
        print(f"[DONE] Processed files : {processed_files}", flush=True)
        print(f"[DONE] Generated tasks : {total_tasks}", flush=True)
        print(f"[DONE] Worker errors   : {error_count}", flush=True)
        print(f"[DONE] Elapsed time    : {elapsed/60:.2f} min", flush=True)
        print(f"[DONE] Output JSONL    : {output_jsonl}", flush=True)
        if error_jsonl is not None:
            print(f"[DONE] Error JSONL     : {error_jsonl}", flush=True)
        print("=" * 80, flush=True)

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print("\n" + "=" * 80, flush=True)
        print("[INTERRUPTED] Received Ctrl+C.", flush=True)
        print(f"[INTERRUPTED] Processed files : {processed_files}", flush=True)
        print(f"[INTERRUPTED] Generated tasks : {total_tasks}", flush=True)
        print(f"[INTERRUPTED] Worker errors   : {error_count}", flush=True)
        print(f"[INTERRUPTED] Elapsed time    : {elapsed/60:.2f} min", flush=True)
        print("=" * 80, flush=True)
        raise

    finally:
        if f_err is not None:
            f_err.close()


if __name__ == "__main__":
    main()
