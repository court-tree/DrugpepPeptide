"""Merge curated Phase-3 source records under the active source policy."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ALLOWED_SOURCES = {
    "BioLiP_peptide",
    "Q-BioLiP_PIII",
    "PepBDB",
    "Propedia",
}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc


def record_key(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("pdb_id", "")).lower(),
        str(row.get("biological_assembly_id", row.get("assembly_id", ""))),
        str(row.get("receptor_chain_id", "")),
        str(row.get("peptide_chain_id", "")),
        str(row.get("peptide_residue_start", "")),
        str(row.get("peptide_residue_end", "")),
        str(row.get("receptor_structure_file") or row.get("complex_structure_file") or ""),
        str(row.get("peptide_structure_file") or ""),
    )


def source_name(row: dict[str, Any]) -> str:
    return str(row.get("source_database") or row.get("source_db") or "unknown")


def merge_sources(
    inputs: list[Path],
    output: Path,
    allowed_sources: set[str],
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, ...]] = set()
    source_counts: Counter[str] = Counter()
    rejected_source_counts: Counter[str] = Counter()
    input_rows = 0
    duplicates = 0
    written = 0
    with output.open("w", encoding="utf-8", newline="\n") as out:
        for path in inputs:
            for row in read_jsonl(path):
                input_rows += 1
                src = source_name(row)
                if src not in allowed_sources:
                    rejected_source_counts[src] += 1
                    continue
                key = record_key(row)
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                row["source_policy_tier"] = source_policy_tier(src)
                source_counts[src] += 1
                out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                written += 1

    summary = {
        "inputs": [str(path) for path in inputs],
        "output": str(output),
        "input_rows": input_rows,
        "written_records": written,
        "dropped_exact_duplicates": duplicates,
        "allowed_sources": sorted(allowed_sources),
        "source_counts": dict(sorted(source_counts.items())),
        "rejected_source_counts": dict(sorted(rejected_source_counts.items())),
        "source_policy": {
            "tier_1": ["BioLiP_peptide", "Q-BioLiP_PIII"],
            "tier_2": ["PepBDB", "Propedia"],
            "not_label_source": ["raw_PDB"],
            "coordinate_reservoir": ["PDB/mmCIF referenced by curated sources"],
        },
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def source_policy_tier(src: str) -> str:
    if src in {"BioLiP_peptide", "Q-BioLiP_PIII"}:
        return "tier_1_curated_positive"
    if src in {"PepBDB", "Propedia"}:
        return "tier_2_curated_positive"
    return "not_allowed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allowed_source",
        action="append",
        default=None,
        help="Allowed source_database value. Defaults to curated V1 sources.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allowed = set(args.allowed_source) if args.allowed_source else set(DEFAULT_ALLOWED_SOURCES)
    summary = merge_sources([Path(path) for path in args.input], Path(args.output), allowed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
