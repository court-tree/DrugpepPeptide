"""Write v3 release documentation and DATA_MANIFEST.json after all audits."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def record_count(path: Path) -> int | None:
    if path.suffix in {".jsonl", ".fasta", ".tsv", ".md", ".py"}:
        with path.open("rb") as handle:
            return sum(1 for _ in handle)
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list): return len(payload)
        if isinstance(payload, dict):
            if set(payload) == {"receptor_to_peptides", "peptide_to_receptors"}:
                return sum(len(payload[key]) for key in payload)
            return 1
    return None


def finalize(v2_dir: str | Path, v3_dir: str | Path, biological_pairs_jsonl: str | Path, repo_root: str | Path) -> dict[str, Any]:
    v2, root, repo = Path(v2_dir).resolve(), Path(v3_dir).resolve(), Path(repo_root).resolve()
    build = json.loads((root / "build_summary.json").read_text(encoding="utf-8"))
    audit = json.loads((root / "full_audit.json").read_text(encoding="utf-8"))
    deterministic = json.loads((root / "determinism_audit.json").read_text(encoding="utf-8"))
    if not audit["passed"] or not deterministic["passed"]:
        raise RuntimeError("cannot finalize a failed v3 audit")
    dependencies = root / "dependencies"
    dependencies.mkdir(exist_ok=True)
    biological_target = dependencies / "biological_pairs.jsonl"
    shutil.copy2(Path(biological_pairs_jsonl), biological_target)
    code_names = [
        "random_conformer_v3.py", "build_random_conformer_v3.py",
        "audit_random_conformer_v3.py", "verify_random_conformer_v3_determinism.py",
        "finalize_random_conformer_v3_release.py", "validate_random_conformer_v3_release.py",
    ]
    for name in code_names:
        shutil.copy2(repo / "phase3/drugclip" / name, dependencies / name)

    validation = f"""# random_conformer_v3 validation report

Status: **PASS**

- Parent dataset: `random_conformer_v2`
- Generator/QC: `{build['generator_id']}`
- Caches: {audit['counts']['caches']:,}
- Conformers: {audit['counts']['conformers']:,} (exactly 10 per cache)
- Replaced conformers: {build['replacement_count']:,}
- Unchanged conformers: {build['unchanged_count']:,}
- Unchanged coordinate SHA256 mismatches: 0
- Interface pairs: {audit['counts']['pairs']:,}
- Pair/split/interface/peptide/known-positive semantic differences: 0
- Missing or mismatched cache references: 0
- Independent clash15 count: 0
- Minimum accepted non-local backbone distance: {audit['counts']['minimum_nonlocal_backbone_distance_angstrom']:.9f} A
- Determinism samples checked: {deterministic['total_checked']:,}
- Determinism mismatches: {deterministic['mismatch_count']}

No training or Pilot was run as part of this data-hardening release.
"""
    (root / "VALIDATION_REPORT.md").write_text(validation, encoding="utf-8")
    development = f"""# PepCLIP Phase-3 DrugCLIP Development Database v3

This release is a geometry-only hardening of `random_conformer_v2`.
Pair identities, receptor interfaces, peptide sequences, splits, biological
mappings, and known-positive groups are unchanged. The parent remains preserved
and read-only.

Formal training inputs:

- `04_training_input/random_conformer_pairs.jsonl`
- `03_random_conformer_cache/random_conformer_cache.jsonl`

The cache contract is `{build['cache_schema']}` and the relation contract is
`{build['database_contract']}`. The formal QC examines peptide N/CA/C atoms only,
excludes same and adjacent residues, and rejects any residue-gap >=2 atom pair
with distance strictly below 1.5 A using unrounded coordinates.

Replacement attempt seed rule:

`SHA256(generator_id|split|peptide_sequence|conformer_index|attempt_index)`,
using the first eight hex digits modulo 2,147,483,646 plus one. Attempt indices
start at 1; attempt 0 is the stored v2 seed used to prove parent reproducibility.

See `VALIDATION_REPORT.md`, `conformer_replacement_audit.jsonl`,
`unchanged_conformer_summary.json`, `full_audit.json`,
`determinism_audit.json`, and `DATA_MANIFEST.json`.
"""
    (root / "DEVELOPMENT_DATABASE.md").write_text(development, encoding="utf-8")

    roles: dict[str, tuple[str, bool, bool, bool, str, str]] = {
        "04_training_input/random_conformer_pairs.jsonl": ("formal interface-peptide training relations", True, True, False, build["relation_schema"], "random_conformer_v2 semantic snapshot"),
        "03_random_conformer_cache/random_conformer_cache.jsonl": ("formal sequence-only conformer cache", True, True, False, build["cache_schema"], "random_conformer_v2 seeds with clash15 replacements"),
        "dependencies/biological_pairs.jsonl": ("formal biological relation mapping", True, True, False, "biological-pairs-jsonl", "receptor_identity_mapping_v1"),
        "02_leakage_safe_split/pair_splits.jsonl": ("formal leakage-safe pair split", True, True, False, "drugclip-interface-positive-v2", "random_conformer_v2 byte-identical snapshot"),
        "01_interface_pairs/known_positive_groups.json": ("formal exact known-positive graph", True, True, False, "known-positive-groups-json", "random_conformer_v2 byte-identical snapshot"),
        "01_interface_pairs/receptor_interfaces.jsonl": ("formal receptor interface coordinates", True, True, False, "receptor-interface-jsonl", "random_conformer_v2 byte-identical snapshot"),
        "02_leakage_safe_split/LEAKAGE_AUDIT.md": ("leakage audit", False, True, True, "markdown", "random_conformer_v2 byte-identical snapshot"),
        "conformer_replacement_audit.jsonl": ("changed-conformer provenance", False, True, True, "conformer-replacement-audit-v1", "random_conformer_v3 build"),
        "unchanged_conformer_summary.json": ("unchanged-coordinate consistency summary", False, True, True, "unchanged-summary-v1", "random_conformer_v3 build"),
        "full_audit.json": ("independent full-data audit", False, True, True, audit["audit_id"], "independent v3 auditor"),
        "determinism_audit.json": ("deterministic regeneration audit", False, True, True, deterministic["audit_id"], "v3 regeneration verifier"),
        "DEVELOPMENT_DATABASE.md": ("release contract and usage", False, True, False, "markdown", "random_conformer_v3 release"),
        "VALIDATION_REPORT.md": ("release validation report", False, True, True, "markdown", "random_conformer_v3 release"),
        "03_random_conformer_cache/random_conformer_summary.json": ("formal conformer build summary", False, True, True, "cache-summary-json", "random_conformer_v3 build"),
        "03_random_conformer_cache/random_conformer_rejects.jsonl": ("formal conformer reject ledger", False, True, True, "reject-jsonl", "random_conformer_v3 build"),
        "04_training_input/summary.json": ("formal relation build summary", False, True, True, "relation-summary-json", "random_conformer_v3 build"),
        "04_training_input/rejects.jsonl": ("formal relation reject ledger", False, True, True, "reject-jsonl", "random_conformer_v3 build"),
        "build_summary.json": ("v3 build summary and parent hashes", False, True, True, "build-summary-json", "random_conformer_v3 build"),
    }
    copied_roles = {
        "01_interface_pairs/interface_pairs.jsonl": "formal interface-peptide identity table",
        "01_interface_pairs/rejects.jsonl": "parent interface reject ledger",
        "01_interface_pairs/summary.json": "parent interface build summary",
        "02_leakage_safe_split/receptors.fasta": "receptor split-audit FASTA",
        "02_leakage_safe_split/receptor_cluster_all_seqs.fasta": "MMseqs all-sequence snapshot",
        "02_leakage_safe_split/receptor_cluster_cluster.tsv": "MMseqs cluster mapping snapshot",
        "02_leakage_safe_split/receptor_cluster_rep_seq.fasta": "MMseqs representative snapshot",
        "02_leakage_safe_split/receptor_families.jsonl": "receptor family mapping",
        "02_leakage_safe_split/summary.json": "leakage-safe split summary",
        "02_leakage_safe_split/train.jsonl": "train split snapshot",
        "02_leakage_safe_split/valid.jsonl": "validation split snapshot",
        "02_leakage_safe_split/test.jsonl": "test split snapshot",
    }
    for relative, role in copied_roles.items():
        suffix = Path(relative).suffix.lstrip(".") or "file"
        roles[relative] = (role, False, True, True, suffix, "random_conformer_v2 byte-identical snapshot")
    for name in code_names:
        roles[f"dependencies/{name}"] = ("build/audit source code", False, True, True, "python", "PepCLIP repository snapshot")
    files = []
    for relative, metadata in sorted(roles.items()):
        path = root / relative
        role, training, validation_required, audit_only, schema, source = metadata
        files.append({
            "role": role, "training_required": training,
            "validation_required": validation_required, "audit_only": audit_only,
            "relative_path": relative.replace("\\", "/"), "sha256": sha256_file(path),
            "bytes": path.stat().st_size, "records_or_lines": record_count(path),
            "schema": schema, "source_dataset_version": source,
        })
    parent_core = dict(build["parent_core_sha256"])
    for relative in ("03_random_conformer_cache/random_conformer_cache.jsonl", "04_training_input/random_conformer_pairs.jsonl", "DEVELOPMENT_DATABASE.md", "VALIDATION_REPORT.md", "final_summary.json"):
        parent_core[relative] = sha256_file(v2 / relative)
    manifest = {
        "manifest_schema": "pepclip-data-manifest-v2",
        "dataset_name": build["dataset_name"], "dataset_version": "random_conformer_v3",
        "parent_dataset": "random_conformer_v2", "created_at": datetime.now(timezone.utc).isoformat(),
        "relation_schema": build["relation_schema"], "cache_schema": build["cache_schema"],
        "generator_id": build["generator_id"],
        "generator_parameters": {"conformers_per_cache": 10, "max_replacement_attempts": build["max_replacement_attempts"], "attempt_seed_rule": "SHA256(generator_id|split|sequence|index|attempt)[0:8] mod 2147483646 + 1"},
        "clash_rule": {"atoms": ["N", "CA", "C"], "minimum_residue_index_gap": 2, "reject_if_distance_angstrom_less_than": 1.5, "boundary_1_50_accepted": True, "coordinates": "unrounded formal coordinates"},
        "split_rule": "unchanged random_conformer_v2 leakage-safe split; exact peptide identity only",
        "exact_peptide_identity_rule": "uppercase full-sequence equality",
        "counts": {"pairs": build["pair_count"], "caches": build["cache_count"], "conformers": build["conformer_count"], "conformers_per_cache": 10, "replaced": build["replacement_count"], "unchanged": build["unchanged_count"]},
        "build_code_version": "pepclip-random-conformer-v3-build-v1",
        "parent_core_sha256": dict(sorted(parent_core.items())),
        "formal_files": files,
        "absolute_paths_recorded": False,
    }
    (root / "DATA_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2_dir", required=True); parser.add_argument("--v3_dir", required=True)
    parser.add_argument("--biological_pairs_jsonl", required=True); parser.add_argument("--repo_root", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    manifest = finalize(**vars(parse_args()))
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
