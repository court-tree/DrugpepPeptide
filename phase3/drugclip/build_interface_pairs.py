"""Build DrugCLIP positives at receptor-interface plus peptide level."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from phase3.drugclip.io_utils import read_jsonl, write_json, write_jsonl
from phase3.drugclip.structure_qc import coordinate_qc


SCHEMA_VERSION = "drugclip-interface-positive-v2"


def _key(sequence: str, peptide: str) -> tuple[str, str]:
    return sequence.upper(), peptide.upper()


def _interface_id(interface: dict[str, Any]) -> str:
    residue_ids = sorted({str(atom["residue_id"]) for atom in interface["receptor_atoms"]})
    material = "|".join(
        [
            str(interface.get("source_pdb_id") or "").lower(),
            str(interface.get("source_chain_id") or ""),
            ",".join(residue_ids),
        ]
    )
    return f"iface:{hashlib.sha256(material.encode('utf-8')).hexdigest()[:20]}"


def _pair_id(interface_id: str, peptide_sequence: str) -> str:
    material = f"{interface_id}|{peptide_sequence}"
    return f"interface_pair:{hashlib.sha256(material.encode('ascii')).hexdigest()[:20]}"


def _collect_evidence(
    biological_rows: list[dict[str, Any]],
    candidate_evidence_jsonl: str | Path,
    expanded_evidence_jsonl: str | Path,
    mmcif_root: str | Path,
) -> dict[str, list[dict[str, Any]]]:
    by_exact: dict[tuple[str, str], str] = {}
    for row in biological_rows:
        for sequence in row.get("receptor_sequences", []):
            by_exact[_key(str(sequence), str(row["peptide_sequence"]))] = str(row["pair_id"])
    evidence: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, str, str]] = set()
    for path in (candidate_evidence_jsonl, expanded_evidence_jsonl):
        for raw in read_jsonl(path):
            if not raw.get("pdb_id") or not raw.get("receptor_chain_id") or not raw.get("peptide_chain_id"):
                continue
            biological_pair_id = by_exact.get(
                _key(str(raw["receptor_sequence"]), str(raw["peptide_sequence"]))
            )
            if biological_pair_id is None:
                continue
            key = (
                biological_pair_id,
                str(raw["pdb_id"]).lower(),
                str(raw["receptor_chain_id"]),
                str(raw["peptide_chain_id"]),
            )
            if key in seen:
                continue
            seen.add(key)
            row = dict(raw)
            mmcif = Path(mmcif_root) / f"{str(row['pdb_id']).lower()}.cif.gz"
            if mmcif.is_file() and str(row.get("source_database") or "") != "Q-BioLiP_PIII":
                row["complex_structure_file"] = str(mmcif.resolve())
            if not row.get("experimental_method") and row.get("structure_method"):
                row["experimental_method"] = row["structure_method"]
            evidence[biological_pair_id].append(row)
    return evidence


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    biological_rows = list(read_jsonl(args.biological_pairs_jsonl))
    biological_by_id = {str(row["pair_id"]): row for row in biological_rows}
    evidence = _collect_evidence(
        biological_rows,
        args.candidate_evidence_jsonl,
        args.expanded_evidence_jsonl,
        args.mmcif_root,
    )
    interfaces: dict[str, dict[str, Any]] = {}
    relations: dict[tuple[str, str], dict[str, Any]] = {}
    rejects: list[dict[str, Any]] = []
    valid_evidence = 0
    for biological_pair_id, rows in evidence.items():
        biological = biological_by_id[biological_pair_id]
        for row in rows:
            try:
                interface, metrics = coordinate_qc(
                    row,
                    biological_pair_id,
                    Path(args.qbiolip_root),
                    Path(args.biolip_root),
                    args.contact_cutoff,
                    args.min_interface_residues,
                )
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
                rejects.append({"evidence_id": row.get("evidence_id", ""), "reason": str(exc)})
                continue
            valid_evidence += 1
            interface_id = _interface_id(interface)
            interface = {
                **interface,
                "pair_id": interface_id,
                "receptor_interface_id": interface_id,
                "structure_id": str(row.get("pdb_id") or "").lower(),
                "peptide_chain_id": str(row.get("peptide_chain_id") or ""),
            }
            previous = interfaces.get(interface_id)
            if previous is None or int(interface["atom_contact_count"]) > int(previous["atom_contact_count"]):
                interfaces[interface_id] = interface
            peptide = str(row["peptide_sequence"]).upper()
            key = (interface_id, peptide)
            relation = relations.get(key)
            if relation is None:
                relation = {
                    "schema_version": SCHEMA_VERSION,
                    "pair_id": _pair_id(interface_id, peptide),
                    "receptor_id": interface_id,
                    "receptor_interface_id": interface_id,
                    "biological_receptor_id": biological["biological_receptor_id"],
                    "receptor_sequence": str(row["receptor_sequence"]),
                    "peptide_sequence": peptide,
                    "source_databases": set(),
                    "evidence_ids": set(),
                    "structure_pdb_ids": set(),
                    "experimental_evidence": {"coordinate_validated_complex"},
                }
                relations[key] = relation
            relation["source_databases"].add(str(row.get("source_database") or ""))
            relation["evidence_ids"].add(str(row.get("evidence_id") or ""))
            relation["structure_pdb_ids"].add(str(row.get("pdb_id") or "").lower())
    pair_rows = []
    for relation in relations.values():
        pair_rows.append(
            {
                **relation,
                "source_database": ";".join(sorted(value for value in relation["source_databases"] if value)),
                "source_databases": sorted(value for value in relation["source_databases"] if value),
                "evidence_ids": sorted(value for value in relation["evidence_ids"] if value),
                "structure_pdb_ids": sorted(value for value in relation["structure_pdb_ids"] if value),
                "experimental_evidence": sorted(relation["experimental_evidence"]),
            }
        )
    pair_rows.sort(key=lambda row: row["pair_id"])
    receptor_to_peptides: defaultdict[str, set[str]] = defaultdict(set)
    peptide_to_interfaces: defaultdict[str, set[str]] = defaultdict(set)
    for row in pair_rows:
        receptor_to_peptides[str(row["receptor_interface_id"])].add(str(row["peptide_sequence"]))
        peptide_to_interfaces[str(row["peptide_sequence"])].add(str(row["receptor_interface_id"]))
    write_jsonl(output / "interface_pairs.jsonl", pair_rows)
    write_jsonl(output / "receptor_interfaces.jsonl", interfaces.values())
    write_jsonl(output / "rejects.jsonl", rejects)
    write_json(
        output / "known_positive_groups.json",
        {
            "receptor_to_peptides": {key: sorted(value) for key, value in receptor_to_peptides.items()},
            "peptide_to_receptors": {key: sorted(value) for key, value in peptide_to_interfaces.items()},
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "biological_pairs_examined": len(biological_rows),
        "coordinate_valid_evidence": valid_evidence,
        "receptor_interfaces": len(interfaces),
        "interface_peptide_pairs": len(pair_rows),
        "rejects": len(rejects),
        "source_databases": dict(Counter(source for row in pair_rows for source in row["source_databases"])),
    }
    write_json(output / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--biological_pairs_jsonl", required=True)
    parser.add_argument("--candidate_evidence_jsonl", required=True)
    parser.add_argument("--expanded_evidence_jsonl", required=True)
    parser.add_argument("--mmcif_root", required=True)
    parser.add_argument("--qbiolip_root", required=True)
    parser.add_argument("--biolip_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--contact_cutoff", type=float, default=6.0)
    parser.add_argument("--min_interface_residues", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
