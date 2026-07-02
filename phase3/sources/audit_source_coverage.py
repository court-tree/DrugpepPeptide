"""Audit Phase-3 source coverage and source-policy status."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .merge_source_records import DEFAULT_ALLOWED_SOURCES, read_jsonl, source_name


def audit_sources(input_jsonl: Path, allowed_sources: set[str] | None = None) -> dict[str, Any]:
    allowed_sources = set(allowed_sources or DEFAULT_ALLOWED_SOURCES)
    counts: Counter[str] = Counter()
    rows = 0
    for row in read_jsonl(input_jsonl):
        rows += 1
        counts[source_name(row)] += 1
    missing_allowed = sorted(allowed_sources - set(counts))
    unexpected = sorted(set(counts) - allowed_sources)
    return {
        "input_jsonl": str(input_jsonl),
        "rows": rows,
        "source_counts": dict(sorted(counts.items())),
        "allowed_sources": sorted(allowed_sources),
        "missing_allowed_sources": missing_allowed,
        "unexpected_sources": unexpected,
        "raw_pdb_as_label_source_present": any(src.lower() in {"pdb", "raw_pdb"} for src in counts),
        "source_policy": {
            "tier_1": ["BioLiP_peptide", "Q-BioLiP_PIII"],
            "tier_2": ["PepBDB", "Propedia"],
            "not_label_source": ["raw_PDB"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--allowed_source", action="append", default=None)
    args = parser.parse_args()
    allowed = set(args.allowed_source) if args.allowed_source else None
    result = audit_sources(Path(args.input_jsonl), allowed)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
