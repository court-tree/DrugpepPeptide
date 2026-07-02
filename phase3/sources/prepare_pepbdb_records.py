"""Prepare PepBDB records for the Phase-3 V1 source table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import gemmi


STANDARD_AA = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}


def load_model(path: Path) -> gemmi.Model:
    structure = gemmi.read_structure(str(path))
    if len(structure) == 0:
        raise ValueError(f"Empty structure: {path}")
    return structure[0]


def protein_residues(chain: gemmi.Chain) -> List[gemmi.Residue]:
    return [residue for residue in chain if residue.name.strip().upper() in STANDARD_AA]


def residue_num(residue: gemmi.Residue) -> int:
    if residue.seqid.num is None:
        raise ValueError(f"Residue missing seqid.num: {residue.name}")
    return int(residue.seqid.num)


def next_chain_name(preferred: str, used: set[str]) -> str:
    if preferred and preferred not in used:
        return preferred
    for name in "PQRSTUVWXYZABCDEFGHIJKLMNO":
        if name not in used:
            return name
    raise ValueError("Could not find an unused one-letter chain name")


def clone_chain_with_name(chain: gemmi.Chain, new_name: str) -> gemmi.Chain:
    cloned = chain.clone()
    cloned.name = new_name
    return cloned


def write_merged_complex(
    receptor_pdb: Path,
    peptide_pdb: Path,
    output_pdb: Path,
    peptide_chain_hint: str,
) -> Tuple[List[str], str, int, int]:
    receptor_model = load_model(receptor_pdb)
    peptide_model = load_model(peptide_pdb)
    receptor_chains = [chain for chain in receptor_model if protein_residues(chain)]
    peptide_chains = [chain for chain in peptide_model if protein_residues(chain)]
    if not receptor_chains:
        raise ValueError("No receptor protein chains")
    if not peptide_chains:
        raise ValueError("No peptide protein chains")

    peptide_chain = None
    if peptide_chain_hint:
        peptide_chain = next((chain for chain in peptide_chains if str(chain.name).strip() == peptide_chain_hint), None)
    if peptide_chain is None:
        peptide_chain = max(peptide_chains, key=lambda chain: len(protein_residues(chain)))

    structure = gemmi.Structure()
    structure.name = output_pdb.stem
    model = gemmi.Model("1")
    used: set[str] = set()
    receptor_out_names: List[str] = []
    for chain in receptor_chains:
        new_name = next_chain_name(str(chain.name).strip(), used)
        used.add(new_name)
        receptor_out_names.append(new_name)
        model.add_chain(clone_chain_with_name(chain, new_name))
    peptide_out_name = next_chain_name(peptide_chain_hint or str(peptide_chain.name).strip() or "P", used)
    model.add_chain(clone_chain_with_name(peptide_chain, peptide_out_name))
    structure.add_model(model)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    structure.write_pdb(str(output_pdb))

    residues = protein_residues(peptide_chain)
    return receptor_out_names, peptide_out_name, residue_num(residues[0]), residue_num(residues[-1])


def iter_pepbdb_entries(pepbdb_root: Path) -> Iterable[Path]:
    for path in sorted(pepbdb_root.iterdir()):
        if path.is_dir() and (path / "peptide.pdb").exists() and (path / "receptor.pdb").exists():
            yield path


def build_records(pepbdb_root: Path, output_dir: Path, limit: int | None, progress_every: int) -> dict[str, Any]:
    structures_dir = output_dir / "structures"
    records_path = output_dir / "pepbdb_phase3_records.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path.write_text("", encoding="utf-8")

    processed = 0
    written_records = 0
    failed: list[dict[str, str]] = []
    for entry_dir in iter_pepbdb_entries(pepbdb_root):
        if limit is not None and processed >= limit:
            break
        processed += 1
        entry_name = entry_dir.name
        pdb_id = entry_name.split("_", 1)[0].lower()
        peptide_hint = entry_name.split("_", 1)[1] if "_" in entry_name else ""
        try:
            complex_rel = Path("structures") / f"{entry_name}.pdb"
            complex_path = output_dir / complex_rel
            receptor_chains, peptide_chain, pep_start, pep_end = write_merged_complex(
                receptor_pdb=entry_dir / "receptor.pdb",
                peptide_pdb=entry_dir / "peptide.pdb",
                output_pdb=complex_path,
                peptide_chain_hint=peptide_hint,
            )
            with records_path.open("a", encoding="utf-8") as handle:
                for receptor_chain in receptor_chains:
                    row: dict[str, Any] = {
                        "source_database": "PepBDB",
                        "source_entry_id": entry_name,
                        "pdb_id": pdb_id,
                        "biological_assembly_id": "pepbdb_curated_complex",
                        "assembly_confidence": "pepbdb_curated_complex",
                        "source_confidence_tier": "tier_2_curated_positive",
                        "complex_structure_file": str(complex_path),
                        "receptor_chain_id": receptor_chain,
                        "peptide_chain_id": peptide_chain,
                        "peptide_residue_start": pep_start,
                        "peptide_residue_end": pep_end,
                    }
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    written_records += 1
        except Exception as exc:
            failed.append({"entry": entry_name, "error": f"{type(exc).__name__}: {exc}"})
        if progress_every > 0 and processed % progress_every == 0:
            print(f"[pepbdb] processed={processed} records={written_records} failed={len(failed)}", flush=True)

    summary = {
        "source_database": "PepBDB",
        "source_policy_tier": "tier_2_curated_positive",
        "pepbdb_root": str(pepbdb_root),
        "output_dir": str(output_dir),
        "records_jsonl": str(records_path),
        "structures_dir": str(structures_dir),
        "processed_entries": processed,
        "written_records": written_records,
        "failed_entries": len(failed),
        "failed_preview": failed[:50],
    }
    (output_dir / "pepbdb_prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pepbdb_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress_every", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_records(Path(args.pepbdb_root), Path(args.output_dir), args.limit, args.progress_every)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
