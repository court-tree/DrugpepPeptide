"""Cluster receptors and create leakage-safe Phase-3 train/valid/test splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from phase3.drugclip.split_utils import connected_components


def receptor_id_of(row: dict[str, Any]) -> str:
    return str(row.get("biological_receptor_id") or row.get("receptor_id") or "")


def receptor_sequence_of(row: dict[str, Any]) -> str:
    sequences = row.get("receptor_sequences")
    if isinstance(sequences, list) and sequences:
        return max((str(item) for item in sequences if str(item)), key=lambda item: (len(item), item))
    return str(row.get("receptor_sequence") or "")


def structure_ids_of(row: dict[str, Any]) -> list[str]:
    explicit = row.get("structure_pdb_ids", []) or []
    if explicit:
        return sorted({str(item).lower() for item in explicit if str(item)})
    return sorted(
        {
            str(item).split("|", 1)[0].lower()
            for item in (row.get("structure_receptor_ids", []) or [])
            if str(item)
        }
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def _win_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    rest = str(resolved)[3:].replace("\\", "/")
    return f"/mnt/{drive}/{rest}"


def _write_receptor_fasta(rows: list[dict[str, Any]], path: Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    for row in rows:
        receptor_id = receptor_id_of(row)
        sequence = receptor_sequence_of(row)
        if not receptor_id or not sequence:
            raise ValueError(f"Missing receptor identity or sequence for pair {row.get('pair_id')}")
        previous = sequences.get(receptor_id)
        if previous is None or (len(sequence), sequence) > (len(previous), previous):
            sequences[receptor_id] = sequence
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for receptor_id, sequence in sorted(sequences.items()):
            handle.write(f">{receptor_id}\n")
            for offset in range(0, len(sequence), 80):
                handle.write(sequence[offset : offset + 80] + "\n")
    return sequences


def _run_mmseqs(
    fasta: Path,
    work_dir: Path,
    executable: Path,
    min_identity: float,
    coverage: float,
) -> Path:
    prefix = work_dir / "receptor_cluster"
    tmp = work_dir / "mmseqs_tmp"
    command = " ".join(
        shlex.quote(item)
        for item in (
            _win_to_wsl(executable),
            "easy-cluster",
            _win_to_wsl(fasta),
            _win_to_wsl(prefix),
            _win_to_wsl(tmp),
            "--min-seq-id",
            str(min_identity),
            "-c",
            str(coverage),
            "--cov-mode",
            "0",
            "--cluster-mode",
            "0",
            "-v",
            "1",
        )
    )
    subprocess.run(["wsl.exe", "bash", "-lc", command], check=True)
    cluster_tsv = prefix.with_name(prefix.name + "_cluster.tsv")
    if not cluster_tsv.is_file():
        raise FileNotFoundError(cluster_tsv)
    return cluster_tsv


def _parse_clusters(path: Path, receptor_ids: Iterable[str]) -> dict[str, str]:
    representative_by_member: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            representative, member = line.rstrip("\n").split("\t")[:2]
            representative_by_member[member] = representative
    result: dict[str, str] = {}
    for receptor_id in receptor_ids:
        representative = representative_by_member.get(receptor_id, receptor_id)
        digest = hashlib.sha256(representative.encode("utf-8")).hexdigest()[:16]
        result[receptor_id] = f"rfam_{digest}"
    return result


def _connect_group(edges: dict[str, set[str]], ids: list[str]) -> None:
    if len(ids) < 2:
        return
    anchor = ids[0]
    for member in ids[1:]:
        edges[anchor].add(member)
        edges[member].add(anchor)


def build_pair_components(
    rows: list[dict[str, Any]],
    receptor_families: dict[str, str],
) -> tuple[list[set[str]], dict[str, int]]:
    edges = {str(row["pair_id"]): set() for row in rows}
    by_family: defaultdict[str, list[str]] = defaultdict(list)
    by_peptide: defaultdict[str, list[str]] = defaultdict(list)
    by_structure: defaultdict[str, list[str]] = defaultdict(list)

    for row in rows:
        pair_id = str(row["pair_id"])
        peptide = str(row["peptide_sequence"])
        by_family[receptor_families[receptor_id_of(row)]].append(pair_id)
        by_peptide[peptide].append(pair_id)
        for structure_id in structure_ids_of(row):
            by_structure[str(structure_id).lower()].append(pair_id)

    for ids in by_family.values():
        _connect_group(edges, ids)
    for ids in by_peptide.values():
        _connect_group(edges, ids)
    for ids in by_structure.values():
        _connect_group(edges, ids)

    components = connected_components(edges)
    return components, {
        "receptor_family_count": len(by_family),
        "exact_peptide_groups": len(by_peptide),
        "structure_groups": len(by_structure),
        "component_count": len(components),
        "largest_component": max(map(len, components), default=0),
    }


def assign_components(
    components: list[set[str]],
    total: int,
    train_fraction: float,
    valid_fraction: float,
) -> dict[str, str]:
    targets = {
        "train": total * train_fraction,
        "valid": total * valid_fraction,
        "test": total * (1.0 - train_fraction - valid_fraction),
    }
    counts = {split: 0 for split in targets}
    assignment: dict[str, str] = {}
    ordered = sorted(components, key=lambda group: (-len(group), min(group)))
    for component in ordered:
        split = max(
            targets,
            key=lambda name: (
                targets[name] - counts[name],
                targets[name],
                name,
            ),
        )
        for pair_id in component:
            assignment[pair_id] = split
        counts[split] += len(component)
    return assignment


def _hash_baseline(pair_id: str) -> str:
    bucket = int(hashlib.sha256(pair_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "valid" if bucket < 90 else "test"


def audit_assignment(
    rows: list[dict[str, Any]],
    assignment: dict[str, str],
    receptor_families: dict[str, str],
) -> dict[str, Any]:
    split_rows: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        split_rows[assignment[str(row["pair_id"])]].append(row)

    counts = {name: len(split_rows[name]) for name in ("train", "valid", "test")}
    pair_sets = {
        name: {(receptor_id_of(row), str(row["peptide_sequence"])) for row in split_rows[name]}
        for name in counts
    }
    receptor_sets = {
        name: {receptor_id_of(row) for row in split_rows[name]} for name in counts
    }
    family_sets = {
        name: {receptor_families[receptor_id_of(row)] for row in split_rows[name]} for name in counts
    }
    peptide_sets = {
        name: {str(row["peptide_sequence"]) for row in split_rows[name]} for name in counts
    }
    structure_sets = {
        name: {
            str(structure_id).lower()
            for row in split_rows[name]
            for structure_id in structure_ids_of(row)
        }
        for name in counts
    }

    comparisons: dict[str, Any] = {}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        comparisons[f"{left}_vs_{right}"] = {
            "exact_pair_overlap": len(pair_sets[left] & pair_sets[right]),
            "exact_receptor_overlap": len(receptor_sets[left] & receptor_sets[right]),
            "receptor_family_overlap": len(family_sets[left] & family_sets[right]),
            "exact_peptide_overlap": len(peptide_sets[left] & peptide_sets[right]),
            "structure_pdb_overlap": len(structure_sets[left] & structure_sets[right]),
        }
    return {"split_counts": counts, "comparisons": comparisons}


def _render_report(summary: dict[str, Any]) -> str:
    clean = summary["leakage_audit"]["component_split"]
    baseline = summary["leakage_audit"]["hash_baseline"]
    lines = [
        "# Phase-3 receptor homology clustering and leakage audit",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        "## Clustering contract",
        "",
        f"- MMseqs2 receptor identity: >= {summary['parameters']['receptor_min_identity']:.0%}",
        f"- Bidirectional receptor coverage: >= {summary['parameters']['receptor_coverage']:.0%}",
        "- Exact peptide sequences are kept within one split.",
        "- Receptor-family, exact-peptide, and shared-PDB edges form the components.",
        "- Similar but non-identical peptides do not create edges or labels.",
        "- A connected component is assigned wholly to one split.",
        "",
        "## Dataset",
        "",
        f"- Logical pairs: {summary['pair_count']}",
        f"- Receptors: {summary['receptor_count']}",
        f"- Receptor families: {summary['component_graph']['receptor_family_count']}",
        f"- Leakage components: {summary['component_graph']['component_count']}",
        f"- Largest component: {summary['component_graph']['largest_component']} pairs",
        "",
        "## Split sizes",
        "",
        "| Split | Pair count |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {clean['split_counts'][name]} |" for name in ("train", "valid", "test"))
    lines.extend(
        [
            "",
            "## Leakage comparison",
            "",
            "A deterministic row-level 80/10/10 hash split is shown as the unsafe baseline.",
            "",
            "| Assignment | Comparison | Exact receptor | Receptor family | Exact peptide | PDB structure |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for label, audit in (("Hash baseline", baseline), ("Component split", clean)):
        for comparison, values in audit["comparisons"].items():
            lines.append(
                f"| {label} | {comparison} | {values['exact_receptor_overlap']} | "
                f"{values['receptor_family_overlap']} | {values['exact_peptide_overlap']} | "
                f"{values['structure_pdb_overlap']} |"
            )
    lines.extend(
        [
            "",
            "The component split is acceptable only when every cross-split overlap column is zero.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.real_pairs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(input_path)
    if not rows:
        raise ValueError("No real pairs found")

    fasta = output_dir / "receptors.fasta"
    receptor_sequences = _write_receptor_fasta(rows, fasta)
    cluster_tsv = _run_mmseqs(
        fasta,
        output_dir,
        Path(args.mmseqs),
        args.receptor_min_identity,
        args.receptor_coverage,
    )
    receptor_families = _parse_clusters(cluster_tsv, receptor_sequences)
    components, graph_summary = build_pair_components(rows, receptor_families)
    assignment = assign_components(
        components, len(rows), args.train_fraction, args.valid_fraction
    )
    baseline_assignment = {str(row["pair_id"]): _hash_baseline(str(row["pair_id"])) for row in rows}

    annotated = [
        {
            **row,
            "receptor_id": receptor_id_of(row),
            "receptor_sequence": receptor_sequence_of(row),
            "receptor_family": receptor_families[receptor_id_of(row)],
            "split": assignment[str(row["pair_id"])],
        }
        for row in rows
    ]
    annotated.sort(key=lambda row: (row["split"], row["pair_id"]))
    _write_jsonl(output_dir / "pair_splits.jsonl", annotated)
    for split in ("train", "valid", "test"):
        _write_jsonl(
            output_dir / f"{split}.jsonl",
            (row for row in annotated if row["split"] == split),
        )
    _write_jsonl(
        output_dir / "receptor_families.jsonl",
        (
            {
                "receptor_id": receptor_id,
                "receptor_family": receptor_families[receptor_id],
                "receptor_length": len(sequence),
                "method": "mmseqs_easy_cluster",
            }
            for receptor_id, sequence in sorted(receptor_sequences.items())
        ),
    )

    component_sizes = Counter(map(len, components))
    summary = {
        "schema_version": "phase3-drugclip-exact-peptide-split-audit-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path.resolve()),
        "pair_count": len(rows),
        "receptor_count": len(receptor_sequences),
        "parameters": {
            "receptor_min_identity": args.receptor_min_identity,
            "receptor_coverage": args.receptor_coverage,
            "peptide_rule": "exact_sequence_only",
            "train_fraction": args.train_fraction,
            "valid_fraction": args.valid_fraction,
            "test_fraction": 1.0 - args.train_fraction - args.valid_fraction,
            "mmseqs_cov_mode": 0,
            "mmseqs_cluster_mode": 0,
        },
        "component_graph": {
            **graph_summary,
            "component_size_distribution": {
                str(size): count for size, count in sorted(component_sizes.items())
            },
        },
        "leakage_audit": {
            "hash_baseline": audit_assignment(rows, baseline_assignment, receptor_families),
            "component_split": audit_assignment(rows, assignment, receptor_families),
        },
        "outputs": {
            "pair_splits": str((output_dir / "pair_splits.jsonl").resolve()),
            "receptor_families": str((output_dir / "receptor_families.jsonl").resolve()),
            "train": str((output_dir / "train.jsonl").resolve()),
            "valid": str((output_dir / "valid.jsonl").resolve()),
            "test": str((output_dir / "test.jsonl").resolve()),
            "mmseqs_cluster_tsv": str(cluster_tsv.resolve()),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "LEAKAGE_AUDIT.md").write_text(_render_report(summary), encoding="utf-8")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real_pairs",
        required=True,
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mmseqs", required=True)
    parser.add_argument("--receptor_min_identity", type=float, default=0.40)
    parser.add_argument("--receptor_coverage", type=float, default=0.60)
    parser.add_argument("--train_fraction", type=float, default=0.80)
    parser.add_argument("--valid_fraction", type=float, default=0.10)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.train_fraction <= 0 or args.valid_fraction <= 0:
        raise ValueError("train_fraction and valid_fraction must be positive")
    if args.train_fraction + args.valid_fraction >= 1:
        raise ValueError("train_fraction + valid_fraction must be less than 1")
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
