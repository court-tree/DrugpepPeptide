"""Read-only fixed-512 peptide input-domain ablation.

This module never trains, calls backward, constructs an optimizer, or mutates a
checkpoint.  It compares true-bound heavy atoms, the exact bound N/CA/C
subset, random conformer 0, and arithmetic mean scores across conformers 0-9.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import shlex
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from phase2.pepclip.train_concat_fusion import move_1d_batch, move_3d_batch
from phase3.drugclip.batching import (
    PHASE2_MAX_PEPTIDE_ATOMS,
    PHASE2_MAX_RECEPTOR_ATOMS,
    UniquePeptideBatchSampler,
    collate_phase3,
)
from phase3.drugclip.build_interface_pairs import _collect_evidence, _interface_id
from phase3.drugclip.evaluate_full_retrieval import (
    _file_hash,
    _load_fixed_dataset,
    _read_json,
    _sequence_hash,
    _validate_pilot_contract,
)
from phase3.drugclip.io_utils import read_jsonl, write_json, write_jsonl
from phase3.drugclip.structure_qc import coordinate_qc, extract_bound_peptide_atoms
from phase3.drugclip.train import load_phase2_fusion_model, load_source_configs, resolve_path


EXPECTED = {
    "interface_pair_sha256": "6BF3206C391BEB590D2C9ED033D947E489CFBBEC5219D33B0F0383AA7D466BE4",
    "plan_file_sha256": "45D4748CB56AB0DF91D8A0DE8FBE4F9013C4B30253E6F080DB04B3D3BB6C9A21",
    "plan_canonical_sha256": "9B132B4AE88D851C7A1709444C78F848D5343312163080E327CF8AD492034DE8",
    "manifest_sha256": "043278F18EFC9B9C3238788D4C6B34C35641C9C26895E5045D8598FA99D5C309",
    "phase2_checkpoint_sha256": "9FB16C48BA715C6273341609D60725AE796AD4A78771744E19ECF2C13D38AE20",
    "phase3_checkpoint_sha256": "5D2D326B634B38A4412950B08093F2152C974403A344AD8A7B2A59EAF8F33599",
    "r2p_candidate_ids_sha256": "B3F50E0A5457F1997D5EF060A1B6B43306B580031A9F38867AB579587FE34311",
    "p2r_candidate_ids_sha256": "5E9A1D761AC70F782BD0C6BF7A7473307830AC23C7FF87D983A23FC3B87F6E91",
}
VARIANTS = ("bound_all_heavy", "bound_backbone_ncac", "random_backbone_conformer0")
REPRESENTATIONS = ("1d_only", "3d_only", "learned_fusion")
METRICS = ("recall_at_1", "recall_at_5", "recall_at_10", "mrr", "median_rank", "mean_rank")


def _json_sha(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def canonical_atom_sha(atoms: list[dict[str, Any]]) -> str:
    canonical = [
        {
            "atom_name": str(atom["atom_name"]),
            "element": str(atom["element"]),
            "residue_id": str(atom["residue_id"]),
            "residue_name": str(atom["residue_name"]),
            "x": float(atom["x"]), "y": float(atom["y"]), "z": float(atom["z"]),
        }
        for atom in atoms
    ]
    return _json_sha(canonical)


def tensor_sha(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest().upper()


def model_state_sha(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor_sha(value).encode("ascii"))
    return digest.hexdigest().upper()


def assert_finite(*values: torch.Tensor) -> None:
    if any(not torch.isfinite(value).all() for value in values):
        raise FloatingPointError("non-finite embedding/score")


def assert_1d_embeddings_identical(reference: torch.Tensor, other: torch.Tensor) -> None:
    if not torch.equal(reference, other):
        raise RuntimeError("1D embedding changed across input variants")


def arithmetic_mean_score(matrices: list[torch.Tensor]) -> torch.Tensor:
    if len(matrices) != 10:
        raise ValueError("Cmean10 requires exactly ten score matrices")
    shape = matrices[0].shape
    if any(matrix.shape != shape or not torch.isfinite(matrix).all() for matrix in matrices):
        raise ValueError("Cmean10 score matrices differ in shape or contain non-finite values")
    return torch.stack(matrices, dim=0).mean(dim=0)


def validate_candidate_bank(candidate_ids: list[str], direction: str) -> None:
    expected = EXPECTED[f"{direction}_candidate_ids_sha256"]
    if _sequence_hash(candidate_ids) != expected:
        raise RuntimeError(f"candidate bank drift:{direction}")


def validate_named_hashes(actual: dict[str, str], expected: dict[str, str]) -> None:
    for key, expected_value in expected.items():
        if actual.get(key) != expected_value:
            raise ValueError(f"SHA256 mismatch:{key}:{actual.get(key)}")


def select_exact_evidence(
    rows: list[dict[str, Any]], evidence_id: str, interface_pair_id: str
) -> dict[str, Any]:
    matches = [row for row in rows if str(row.get("evidence_id") or "") == evidence_id]
    if len(matches) != 1:
        raise ValueError(f"exact evidence join is not 1:1:{interface_pair_id}:{len(matches)}")
    return matches[0]


def atom_cap_audit(atoms: list[dict[str, Any]], cap: int) -> dict[str, Any]:
    return {
        "before": len(atoms), "after": min(len(atoms), cap),
        "touched_cap": len(atoms) >= cap, "truncated": len(atoms) > cap,
    }


def call_structure_extractor(
    extractor: Any, interface_pair_id: str, evidence_id: str, *args: Any
) -> Any:
    try:
        return extractor(*args)
    except Exception as exc:
        raise RuntimeError(
            f"interface_pair_id={interface_pair_id};evidence_id={evidence_id};"
            f"original_exception={type(exc).__name__}: {exc}"
        ) from exc


def _clone_item(item: dict[str, Any], peptide_atoms: list[dict[str, Any]]) -> dict[str, Any]:
    return {**item, "peptide_atoms": [dict(atom) for atom in peptide_atoms]}


class _ItemDataset:
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self.items = items
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def peptide_sequence_for_index(self, index: int) -> str:
        return str(self.items[index]["peptide_sequence"])


def _input_sha(item: dict[str, Any]) -> tuple[str, str]:
    receptor = canonical_atom_sha(item["receptor_atoms"][:PHASE2_MAX_RECEPTOR_ATOMS])
    one_d = _json_sha({
        "receptor_patch_sequence": str(item["receptor_patch_sequence"]),
        "peptide_sequence": str(item["peptide_sequence"]),
    })
    return receptor, one_d


def canonical_ncac_subset(atoms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return source atoms as residue-ordered canonical N/CA/C triplets."""

    residue_order: list[str] = []
    by_residue: dict[str, dict[str, dict[str, Any]]] = {}
    for atom in atoms:
        residue_id = str(atom["residue_id"])
        if residue_id not in by_residue:
            residue_order.append(residue_id)
            by_residue[residue_id] = {}
        atom_name = str(atom["atom_name"])
        if atom_name not in {"N", "CA", "C"}:
            continue
        if atom_name in by_residue[residue_id]:
            raise ValueError(f"duplicate backbone atom:{residue_id}:{atom_name}")
        by_residue[residue_id][atom_name] = atom
    output = []
    for residue_id in residue_order:
        named = by_residue[residue_id]
        missing = [name for name in ("N", "CA", "C") if name not in named]
        if missing:
            raise ValueError(
                f"A complete-residue atoms lack canonical backbone:{residue_id}:"
                f"missing={','.join(missing)}"
            )
        output.extend(dict(named[name]) for name in ("N", "CA", "C"))
    return output


def exact_evidence_join(
    plan: list[dict[str, Any]],
    base_dataset: Any,
    candidate_evidence_jsonl: Path,
    expanded_evidence_jsonl: Path,
    mmcif_root: Path,
    qbiolip_root: Path,
    biolip_root: Path,
    biological_pairs_jsonl: Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    """Reproduce the original evidence join and build all four input domains."""

    biological_rows = list(read_jsonl(biological_pairs_jsonl))
    evidence = _collect_evidence(
        biological_rows, candidate_evidence_jsonl, expanded_evidence_jsonl, mmcif_root
    )
    variants: dict[str, list[dict[str, Any]]] = {name: [] for name in VARIANTS}
    variants.update({f"random_backbone_conformer{i}": [] for i in range(1, 10)})
    audits: list[dict[str, Any]] = []
    source_counts: defaultdict[str, int] = defaultdict(int)
    for index, plan_row in enumerate(plan):
        pair_id = str(plan_row["interface_pair_id"])
        biological_pair_id = str(plan_row["biological_pair_id"])
        base_dataset.base.fixed_conformer_index = 0
        base_dataset.set_epoch(0)
        c0 = base_dataset[index]
        formal_row = base_dataset.base.interface_pair_rows[pair_id]
        formal_interface = formal_row["interface"]
        evidence_id = str(formal_interface["evidence"])
        evidence_row = select_exact_evidence(
            evidence.get(biological_pair_id, []), evidence_id, pair_id
        )
        regenerated, _ = call_structure_extractor(
            coordinate_qc, pair_id, evidence_id,
            evidence_row, biological_pair_id, qbiolip_root, biolip_root
        )
        if _interface_id(regenerated) != str(formal_interface["receptor_interface_id"]):
            raise ValueError(f"exact evidence does not regenerate formal interface:{pair_id}")
        if canonical_atom_sha(regenerated["receptor_atoms"]) != canonical_atom_sha(formal_interface["receptor_atoms"]):
            raise ValueError(f"regenerated receptor atoms differ from formal input:{pair_id}")
        bound = call_structure_extractor(
            extract_bound_peptide_atoms, pair_id, evidence_id,
            evidence_row, qbiolip_root, biolip_root
        )
        sequence = str(c0["peptide_sequence"])
        if bound["expected_sequence"] != sequence or bound["observed_sequence"] != sequence:
            raise ValueError(f"peptide sequence mismatch:{pair_id}")
        a_raw = bound["all_heavy_atoms"]
        b_atoms = bound["backbone_ncac_atoms"]
        if len(b_atoms) != 3 * len(sequence):
            raise ValueError(f"B is not exactly three atoms per residue:{pair_id}")
        expected_subset = canonical_ncac_subset(a_raw)
        if canonical_atom_sha(b_atoms) != canonical_atom_sha(expected_subset):
            raise ValueError(f"B is not the exact A N/CA/C subset:{pair_id}")
        a_cap = atom_cap_audit(a_raw, PHASE2_MAX_PEPTIDE_ATOMS)
        a_used = a_raw[:PHASE2_MAX_PEPTIDE_ATOMS]
        variants["bound_all_heavy"].append(_clone_item(c0, a_used))
        variants["bound_backbone_ncac"].append(_clone_item(c0, b_atoms))
        variants["random_backbone_conformer0"].append(_clone_item(c0, c0["peptide_atoms"]))
        c_hashes = [canonical_atom_sha(c0["peptide_atoms"])]
        c_counts = [len(c0["peptide_atoms"])]
        for conformer_index in range(1, 10):
            base_dataset.base.fixed_conformer_index = conformer_index
            base_dataset.set_epoch(0)
            item = base_dataset[index]
            if str(item["peptide_sequence"]) != sequence:
                raise ValueError(f"random conformer sequence drift:{pair_id}:{conformer_index}")
            variants[f"random_backbone_conformer{conformer_index}"].append(item)
            c_hashes.append(canonical_atom_sha(item["peptide_atoms"]))
            c_counts.append(len(item["peptide_atoms"]))
        receptor_sha, one_d_sha = _input_sha(c0)
        for name, items in variants.items():
            if len(items) == index + 1:
                item_receptor_sha, item_one_d_sha = _input_sha(items[index])
                if item_receptor_sha != receptor_sha or item_one_d_sha != one_d_sha:
                    raise RuntimeError(f"receptor/1D input drift:{pair_id}:{name}")
        source_counts[str(evidence_row.get("source_database") or "")] += 1
        audits.append({
            "interface_pair_id": pair_id,
            "evidence_id": evidence_id,
            "source_database": str(evidence_row.get("source_database") or ""),
            "structure_path": bound["peptide_structure_path"],
            "structure_type": bound["structure_type"],
            "peptide_chain": bound["peptide_chain"],
            "expected_sequence": sequence,
            "observed_sequence": bound["observed_sequence"],
            "excluded_incomplete_residue_count": bound["excluded_incomplete_residue_count"],
            "excluded_incomplete_residues": bound["excluded_incomplete_residues"],
            "source_backbone_order_canonical": bound["source_backbone_order_canonical"],
            "reordered_backbone_residue_count": bound["reordered_backbone_residue_count"],
            "reordered_backbone_residues": bound["reordered_backbone_residues"],
            "a_atom_count_before_truncation": a_cap["before"],
            "a_atom_count_after_truncation": a_cap["after"],
            "a_touched_192_atom_cap": a_cap["touched_cap"],
            "a_was_truncated": a_cap["truncated"],
            "b_atom_count": len(b_atoms),
            "c_atom_counts": c_counts,
            "a_raw_canonical_atom_sha256": canonical_atom_sha(a_raw),
            "b_canonical_atom_sha256": canonical_atom_sha(b_atoms),
            "c0_to_c9_canonical_atom_sha256": c_hashes,
            "receptor_input_sha256": receptor_sha,
            "one_d_input_sha256": one_d_sha,
        })
    base_dataset.base.fixed_conformer_index = 0
    base_dataset.set_epoch(0)
    summary = {
        "query_count": len(plan),
        "exact_evidence_matches": len(audits),
        "sequence_mismatches": 0,
        "missing_structure_files": 0,
        "bound_backbone_subset_failures": 0,
        "source_database_counts": dict(sorted(source_counts.items())),
        "a_touched_192_atom_cap": sum(row["a_touched_192_atom_cap"] for row in audits),
        "a_truncated_over_192": sum(row["a_was_truncated"] for row in audits),
        "pairs_with_excluded_incomplete_residues": sum(
            row["excluded_incomplete_residue_count"] > 0 for row in audits
        ),
        "excluded_incomplete_residue_count": sum(
            row["excluded_incomplete_residue_count"] for row in audits
        ),
        "pairs_with_reordered_backbone_residues": sum(
            row["reordered_backbone_residue_count"] > 0 for row in audits
        ),
        "reordered_backbone_residue_count": sum(
            row["reordered_backbone_residue_count"] for row in audits
        ),
    }
    if len(audits) != 512:
        raise ValueError("not 512/512 exact evidence matches")
    return variants, audits, summary


def _model_args(pilot_config: dict[str, Any], device: torch.device) -> argparse.Namespace:
    args = argparse.Namespace(**pilot_config["args"])
    args.device = str(device)
    return args


def _load_model(
    label: str, checkpoint: Path, phase2_checkpoint: Path, source_configs: dict[str, Any],
    train_args: argparse.Namespace, repo_root: Path, device: torch.device,
) -> torch.nn.Module:
    model = load_phase2_fusion_model(
        phase2_checkpoint, source_configs, device, train_args, repo_root
    )
    if label == "phase3_v1_epoch0":
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    return model


def _sampler_batches(items: list[dict[str, Any]], batch_size: int) -> list[list[int]]:
    sampler = UniquePeptideBatchSampler(_ItemDataset(items), batch_size=batch_size, seed=17, epoch=0)
    return [list(batch) for batch in sampler]


def _encode_common(
    model: torch.nn.Module, items: list[dict[str, Any]], batches: list[list[int]], device: torch.device
) -> dict[str, dict[str, torch.Tensor]]:
    rows: dict[str, dict[str, torch.Tensor]] = {}
    with torch.inference_mode():
        for indices in batches:
            batch = collate_phase3([items[index] for index in indices])
            one_d = move_1d_batch(batch["one_d"], device)
            three_d = move_3d_batch(batch["three_d"], device)
            receptor_1d = model.model_1d.encode_receptor(one_d["receptor_tokens"], one_d["receptor_sequence"])
            peptide_1d = model.model_1d.encode_peptide(one_d["peptide_tokens"], one_d["peptide_sequence"])
            receptor_3d = model.model_3d.encode_receptor(
                three_d["receptor_coords"], three_d["receptor_elements"], three_d["receptor_mask"],
                three_d["receptor_atom_names"], three_d["receptor_residue_names"],
            )
            assert_finite(receptor_1d, peptide_1d, receptor_3d)
            for offset, index in enumerate(indices):
                rows[str(items[index]["interface_pair_id"])] = {
                    "receptor_1d": receptor_1d[offset].float().cpu(),
                    "peptide_1d": peptide_1d[offset].float().cpu(),
                    "receptor_3d": receptor_3d[offset].float().cpu(),
                }
    return rows


def _encode_variant(
    model: torch.nn.Module, items: list[dict[str, Any]], batches: list[list[int]],
    common: dict[str, dict[str, torch.Tensor]], device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    output: dict[str, dict[str, torch.Tensor]] = {}
    with torch.inference_mode():
        for indices in batches:
            batch = collate_phase3([items[index] for index in indices])
            three_d = move_3d_batch(batch["three_d"], device)
            peptide_3d = model.model_3d.encode_peptide(
                three_d["peptide_coords"], three_d["peptide_elements"], three_d["peptide_mask"],
                three_d["peptide_atom_names"], three_d["peptide_residue_names"],
            )
            assert_finite(peptide_3d)
            pair_ids = [str(items[index]["interface_pair_id"]) for index in indices]
            receptor_concat = torch.stack([
                torch.cat([common[pair_id]["receptor_1d"], common[pair_id]["receptor_3d"]])
                for pair_id in pair_ids
            ]).to(device)
            peptide_concat = torch.stack([
                torch.cat([common[pair_id]["peptide_1d"], peptide_3d[offset].float().cpu()])
                for offset, pair_id in enumerate(pair_ids)
            ]).to(device)
            receptor_fused = model.receptor_fusion(receptor_concat)
            peptide_fused = model.peptide_fusion(peptide_concat)
            assert_finite(receptor_fused, peptide_fused)
            for offset, pair_id in enumerate(pair_ids):
                output[pair_id] = {
                    "receptor_1d": common[pair_id]["receptor_1d"],
                    "peptide_1d": common[pair_id]["peptide_1d"],
                    "receptor_3d": common[pair_id]["receptor_3d"],
                    "peptide_3d": peptide_3d[offset].float().cpu(),
                    "receptor_fused": receptor_fused[offset].float().cpu(),
                    "peptide_fused": peptide_fused[offset].float().cpu(),
                }
    return output


def _candidate_contract(
    items: list[dict[str, Any]], insertion_order: list[str], direction: Literal["r2p", "p2r"]
) -> tuple[list[str], dict[str, str]]:
    by_pair = {str(item["interface_pair_id"]): item for item in items}
    field = "peptide_sequence" if direction == "r2p" else "receptor_id"
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for pair_id in insertion_order:
        groups[str(by_pair[pair_id][field])].append(pair_id)
    candidate_ids = sorted(groups)
    validate_candidate_bank(candidate_ids, direction)
    return candidate_ids, {candidate_id: groups[candidate_id][0] for candidate_id in candidate_ids}


def _score_matrix(
    encoded: dict[str, dict[str, torch.Tensor]], plan_ids: list[str], candidate_ids: list[str],
    representatives: dict[str, str], direction: Literal["r2p", "p2r"], representation: str,
    temperature: float,
) -> torch.Tensor:
    fields = {
        "1d_only": ("receptor_1d", "peptide_1d"),
        "3d_only": ("receptor_3d", "peptide_3d"),
        "learned_fusion": ("receptor_fused", "peptide_fused"),
    }
    receptor_field, peptide_field = fields[representation]
    query_field = receptor_field if direction == "r2p" else peptide_field
    candidate_field = peptide_field if direction == "r2p" else receptor_field
    queries = torch.stack([encoded[pair_id][query_field] for pair_id in plan_ids])
    candidates = torch.stack([encoded[representatives[candidate_id]][candidate_field] for candidate_id in candidate_ids])
    scores = (queries @ candidates.t()) / float(temperature)
    assert_finite(scores)
    return scores


def _metric_summary(ranks: list[int]) -> dict[str, float]:
    if not ranks:
        raise ValueError("no ranks")
    tensor = torch.tensor(ranks, dtype=torch.float64)
    return {
        "recall_at_1": sum(rank <= 1 for rank in ranks) / len(ranks),
        "recall_at_5": sum(rank <= 5 for rank in ranks) / len(ranks),
        "recall_at_10": sum(rank <= 10 for rank in ranks) / len(ranks),
        "mrr": sum(1.0 / rank for rank in ranks) / len(ranks),
        "median_rank": float(tensor.median().item()),
        "mean_rank": float(tensor.mean().item()),
    }


def _rank_scores(
    scores: torch.Tensor, items: list[dict[str, Any]], candidate_ids: list[str],
    direction: Literal["r2p", "p2r"],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_index = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
    candidate_set = set(candidate_ids)
    target_field = "peptide_sequence" if direction == "r2p" else "receptor_id"
    known_field = "receptor_peptides" if direction == "r2p" else "peptide_receptors"
    records = []
    for query_index, item in enumerate(items):
        target = str(item[target_field])
        declared = {str(value) for value in item["known_positive_group"].get(known_field, [])}
        if target not in declared or target not in candidate_index:
            raise ValueError(f"target/known-positive policy drift:{direction}:{target}")
        excluded = (declared - {target}) & candidate_set
        allowed = [candidate for candidate in candidate_ids if candidate not in excluded]
        ranked = sorted(allowed, key=lambda value: (-float(scores[query_index, candidate_index[value]]), value))
        records.append({
            "interface_pair_id": str(item["interface_pair_id"]), "target_id": target,
            "rank": ranked.index(target) + 1,
            "target_score": float(scores[query_index, candidate_index[target]]),
            "candidate_count": len(allowed), "known_positive_candidates_excluded": len(excluded),
        })
    metrics = _metric_summary([int(row["rank"]) for row in records])
    metrics.update({
        "queries": len(records),
        "candidate_count_min": min(row["candidate_count"] for row in records),
        "candidate_count_max": max(row["candidate_count"] for row in records),
        "known_positive_exclusion_total": sum(row["known_positive_candidates_excluded"] for row in records),
        "tie_policy": "descending_score_then_lexicographic_candidate_id",
    })
    return records, metrics


def _metric_delta(later: list[int], earlier: list[int]) -> dict[str, float]:
    later_metrics, earlier_metrics = _metric_summary(later), _metric_summary(earlier)
    return {name: later_metrics[name] - earlier_metrics[name] for name in METRICS}


def paired_bootstrap(
    later: list[int], earlier: list[int], indices: np.ndarray, seed: int
) -> dict[str, Any]:
    if len(later) != len(earlier) or indices.shape[1] != len(later):
        raise ValueError("paired bootstrap query mismatch")
    a = np.asarray(later, dtype=np.float64)[indices]
    b = np.asarray(earlier, dtype=np.float64)[indices]
    values = {
        "recall_at_1": (a <= 1).mean(1) - (b <= 1).mean(1),
        "recall_at_5": (a <= 5).mean(1) - (b <= 5).mean(1),
        "recall_at_10": (a <= 10).mean(1) - (b <= 10).mean(1),
        "mrr": (1.0 / a).mean(1) - (1.0 / b).mean(1),
        "median_rank": np.partition(a, (a.shape[1] - 1) // 2, axis=1)[:, (a.shape[1] - 1) // 2]
                       - np.partition(b, (b.shape[1] - 1) // 2, axis=1)[:, (b.shape[1] - 1) // 2],
        "mean_rank": a.mean(1) - b.mean(1),
    }
    point = _metric_delta(later, earlier)
    return {
        "queries": len(later), "resamples": int(indices.shape[0]), "seed": seed,
        "later_minus_earlier": {
            name: {
                "point_estimate": point[name],
                "ci_95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
                "crosses_zero": bool(np.quantile(samples, 0.025) <= 0 <= np.quantile(samples, 0.975)),
            }
            for name, samples in values.items()
        },
    }


def _embedding_shift(encoded: dict[str, dict[str, dict[str, torch.Tensor]]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    pairs = (("bound_all_heavy", "bound_backbone_ncac"), ("bound_backbone_ncac", "random_backbone_conformer0"))
    for later, earlier in pairs:
        key = f"{later}_vs_{earlier}"
        output[key] = {}
        ids = sorted(encoded[later])
        for field in ("peptide_3d", "peptide_fused"):
            cosine = [float(torch.dot(encoded[later][pair_id][field], encoded[earlier][pair_id][field])) for pair_id in ids]
            output[key][field] = {
                "mean_cosine": statistics.mean(cosine), "median_cosine": statistics.median(cosine),
                "min_cosine": min(cosine), "max_cosine": max(cosine),
            }
    return output


def _preflight(cli: argparse.Namespace, repo_root: Path) -> dict[str, Any]:
    output = resolve_path(cli.output_dir, repo_root).resolve()
    if output.exists():
        raise FileExistsError(f"formal output directory already exists:{output}")
    pilot = resolve_path(cli.pilot_output, repo_root).resolve()
    dataset_root = resolve_path(cli.dataset_root, repo_root).resolve()
    phase2_checkpoint = resolve_path(cli.phase2_checkpoint, repo_root).resolve()
    phase3_checkpoint = resolve_path(cli.phase3_checkpoint, repo_root).resolve()
    plan_path = pilot / "validation_sampling_plan.jsonl"
    manifest_path = dataset_root / "DATA_MANIFEST.json"
    checks = {
        "plan_file_sha256": _file_hash(plan_path), "manifest_sha256": _file_hash(manifest_path),
        "phase2_checkpoint_sha256": _file_hash(phase2_checkpoint),
        "phase3_checkpoint_sha256": _file_hash(phase3_checkpoint),
    }
    validate_named_hashes(checks, {key: EXPECTED[key] for key in checks})
    config, subset, plan = _validate_pilot_contract(pilot)
    pair_ids = [str(row["interface_pair_id"]) for row in plan]
    if _sequence_hash(pair_ids) != EXPECTED["interface_pair_sha256"]:
        raise ValueError("fixed interface-pair SHA256 mismatch")
    if str(config["fixed_validation_plan_sha256"]) != EXPECTED["plan_canonical_sha256"]:
        raise ValueError("canonical fixed-plan SHA256 mismatch")
    base_dataset = _load_fixed_dataset(config, subset, plan)
    variants, audits, source_summary = exact_evidence_join(
        plan, base_dataset,
        resolve_path(cli.candidate_evidence_jsonl, repo_root).resolve(),
        resolve_path(cli.expanded_evidence_jsonl, repo_root).resolve(),
        resolve_path(cli.mmcif_root, repo_root).resolve(),
        resolve_path(cli.qbiolip_root, repo_root).resolve(),
        resolve_path(cli.biolip_root, repo_root).resolve(),
        dataset_root / "dependencies" / "biological_pairs.jsonl",
    )
    return {
        "output": output, "pilot": pilot, "dataset_root": dataset_root,
        "phase2_checkpoint": phase2_checkpoint, "phase3_checkpoint": phase3_checkpoint,
        "config": config, "plan": plan, "variants": variants, "audits": audits,
        "source_summary": source_summary, "hash_checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-output", "--pilot_output", required=True)
    parser.add_argument("--dataset-root", "--dataset_root", required=True)
    parser.add_argument("--candidate-evidence-jsonl", "--candidate_evidence_jsonl", required=True)
    parser.add_argument("--expanded-evidence-jsonl", "--expanded_evidence_jsonl", required=True)
    parser.add_argument("--mmcif-root", "--mmcif_root", required=True)
    parser.add_argument("--qbiolip-root", "--qbiolip_root", required=True)
    parser.add_argument("--biolip-root", "--biolip_root", required=True)
    parser.add_argument("--phase2-checkpoint", "--phase2_checkpoint", required=True)
    parser.add_argument("--phase3-checkpoint", "--phase3_checkpoint", required=True)
    parser.add_argument("--source-model-configs", "--source_model_configs", required=True)
    parser.add_argument("--conformer-indices", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", "--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def _summary_markdown(metrics: dict[str, Any], differences: dict[str, Any], source: dict[str, Any]) -> str:
    lines = [
        "# Phase-3 v1 fixed-512 input-domain ablation", "",
        "> Diagnostic read-only evaluation. Bound A/B are true-bound upper-bound diagnostics, not deployable screening performance. Tower-only metrics are diagnostic and no model-selection claim is made.", "",
        f"- Exact evidence coverage: {source['exact_evidence_matches']}/512",
        f"- A touched 192-atom boundary: {source['a_touched_192_atom_cap']}; truncated above cap: {source['a_truncated_over_192']}", "",
        "## Retrieval metrics", "",
    ]
    for model, model_rows in metrics.items():
        lines.extend([f"### {model}", ""])
        for representation, representation_rows in model_rows.items():
            lines.extend([f"#### {representation}", "", "| Variant | Direction | R@1 | R@5 | R@10 | MRR | Median rank | Mean rank |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
            for variant, directions in representation_rows.items():
                for direction, row in directions.items():
                    lines.append(f"| {variant} | {direction} | {row['recall_at_1']:.5f} | {row['recall_at_5']:.5f} | {row['recall_at_10']:.5f} | {row['mrr']:.5f} | {row['median_rank']:.1f} | {row['mean_rank']:.3f} |")
            lines.append("")
    lines.extend(["## Pre-registered paired differences", "", "See `paired_metric_differences.json` and `bootstrap_confidence_intervals.json` for all point estimates and deterministic 95% intervals.", ""])
    return "\n".join(lines)


def main() -> None:
    cli = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    if tuple(int(value) for value in cli.conformer_indices.split(",")) != tuple(range(10)):
        raise ValueError("conformer indices must be exactly 0,1,2,3,4,5,6,7,8,9")
    if cli.bootstrap_seed != 20260723 or cli.bootstrap_resamples != 10000:
        raise ValueError("formal bootstrap contract requires seed 20260723 and 10000 resamples")
    preflight = _preflight(cli, repo_root)
    if cli.preflight_only:
        print(json.dumps({"status": "PASS", "preflight": preflight["source_summary"], "hashes": preflight["hash_checks"]}, ensure_ascii=False, indent=2))
        return
    output: Path = preflight["output"]
    if output.exists():
        raise FileExistsError(f"formal output directory already exists:{output}")
    output.mkdir(parents=True)
    command = shlex.join([sys.executable, "-m", "phase3.drugclip.evaluate_input_domain_ablation", *sys.argv[1:]])
    (output / "launcher.log").write_text(command + "\n", encoding="utf-8")
    (output / "stdout.log").write_text("preflight PASS; starting read-only evaluation\n", encoding="utf-8")
    (output / "stderr.log").write_text("", encoding="utf-8")
    logger = logging.getLogger("phase3.input_domain_ablation")
    logger.handlers.clear(); logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s")); logger.addHandler(handler)
    try:
        device = torch.device(cli.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        write_json(output / "source_structure_audit.json", preflight["source_summary"])
        write_jsonl(output / "input_variant_audit.jsonl", preflight["audits"])
        config = preflight["config"]
        source_configs = load_source_configs(
            preflight["phase2_checkpoint"], resolve_path(cli.source_model_configs, repo_root), repo_root
        )
        train_args = _model_args(config, device)
        plan = preflight["plan"]
        plan_ids = [str(row["interface_pair_id"]) for row in plan]
        items0 = preflight["variants"]["random_backbone_conformer0"]
        batches = _sampler_batches(items0, cli.batch_size)
        insertion_order = [str(items0[index]["interface_pair_id"]) for batch in batches for index in batch]
        candidate_contracts = {
            direction: _candidate_contract(items0, insertion_order, direction)
            for direction in ("r2p", "p2r")
        }
        all_metrics: dict[str, Any] = {}
        all_records: dict[str, Any] = {}
        shifts: dict[str, Any] = {}
        checkpoint_audit: dict[str, Any] = {"read_only": True, "checkpoints": {}}
        score_store: dict[str, Any] = {}
        rank_store: dict[str, Any] = {}
        for label, checkpoint in (
            ("phase2_baseline", preflight["phase2_checkpoint"]),
            ("phase3_v1_epoch0", preflight["phase3_checkpoint"]),
        ):
            logger.info("loading %s", label)
            model = _load_model(label, checkpoint, preflight["phase2_checkpoint"], source_configs, train_args, repo_root, device)
            before_sha = model_state_sha(model)
            common = _encode_common(model, items0, batches, device)
            encoded: dict[str, Any] = {}
            for variant, items in preflight["variants"].items():
                encoded[variant] = _encode_variant(model, items, batches, common, device)
                for pair_id in plan_ids:
                    assert_1d_embeddings_identical(common[pair_id]["peptide_1d"], encoded[variant][pair_id]["peptide_1d"])
            model_metrics: dict[str, Any] = {representation: {} for representation in REPRESENTATIONS}
            model_records: dict[str, Any] = {representation: {} for representation in REPRESENTATIONS}
            score_store[label] = {}; rank_store[label] = {}
            for representation in REPRESENTATIONS:
                score_store[label][representation] = {}; rank_store[label][representation] = {}
                variant_names = list(VARIANTS) + ["random_backbone_mean10"]
                for variant in variant_names:
                    model_metrics[representation][variant] = {}
                    model_records[representation][variant] = {}
                    score_store[label][representation][variant] = {}
                    rank_store[label][representation][variant] = {}
                    for direction in ("r2p", "p2r"):
                        candidate_ids, representatives = candidate_contracts[direction]
                        if variant == "random_backbone_mean10":
                            matrices = [
                                _score_matrix(encoded[f"random_backbone_conformer{i}"], plan_ids, candidate_ids, representatives, direction, representation, model.temperature)
                                for i in range(10)
                            ]
                            scores = arithmetic_mean_score(matrices)
                        else:
                            scores = _score_matrix(encoded[variant], plan_ids, candidate_ids, representatives, direction, representation, model.temperature)
                        records, metric = _rank_scores(scores, items0, candidate_ids, direction)
                        model_metrics[representation][variant][direction] = metric
                        model_records[representation][variant][direction] = records
                        rank_store[label][representation][variant][direction] = [int(row["rank"]) for row in records]
            after_sha = model_state_sha(model)
            if before_sha != after_sha:
                raise RuntimeError(f"model tensor changed during read-only evaluation:{label}")
            checkpoint_audit["checkpoints"][label] = {
                "path": str(checkpoint), "file_sha256": _file_hash(checkpoint),
                "model_state_sha256_before": before_sha, "model_state_sha256_after": after_sha,
                "unchanged": True, "finite": True,
            }
            all_metrics[label] = model_metrics
            all_records[label] = model_records
            shifts[label] = _embedding_shift(encoded)
            del model
            if device.type == "cuda": torch.cuda.empty_cache()
        rng = np.random.default_rng(cli.bootstrap_seed)
        bootstrap_indices = rng.integers(0, len(plan_ids), size=(cli.bootstrap_resamples, len(plan_ids)), dtype=np.int32)
        point_differences: dict[str, Any] = {}
        intervals: dict[str, Any] = {}
        comparisons = []
        for label in all_metrics:
            for representation in REPRESENTATIONS:
                for direction in ("r2p", "p2r"):
                    for later, earlier, name in (
                        ("bound_all_heavy", "bound_backbone_ncac", "A_minus_B"),
                        ("bound_backbone_ncac", "random_backbone_conformer0", "B_minus_C0"),
                        ("random_backbone_mean10", "random_backbone_conformer0", "Cmean10_minus_C0"),
                    ):
                        comparisons.append((f"{label}/{representation}/{direction}/{name}", label, label, representation, direction, later, earlier))
        for representation in REPRESENTATIONS:
            for direction in ("r2p", "p2r"):
                for variant in list(VARIANTS) + ["random_backbone_mean10"]:
                    comparisons.append((f"phase3_minus_phase2/{representation}/{direction}/{variant}", "phase3_v1_epoch0", "phase2_baseline", representation, direction, variant, variant))
        for key, later_model, earlier_model, representation, direction, later_variant, earlier_variant in comparisons:
            later = rank_store[later_model][representation][later_variant][direction]
            earlier = rank_store[earlier_model][representation][earlier_variant][direction]
            point_differences[key] = _metric_delta(later, earlier)
            intervals[key] = paired_bootstrap(later, earlier, bootstrap_indices, cli.bootstrap_seed)
        flattened_records = []
        for model, model_rows in all_records.items():
            for representation, representation_rows in model_rows.items():
                for variant, directions in representation_rows.items():
                    for direction, rows in directions.items():
                        flattened_records.extend({"model": model, "representation": representation, "variant": variant, "direction": direction, **row} for row in rows)
        write_json(output / "evaluation_config.json", {
            "schema_version": "phase3-input-domain-ablation-fixed512-v1", "command": command,
            "device": str(device), "query_count": 512, "conformer_indices": list(range(10)),
            "bootstrap_seed": cli.bootstrap_seed, "bootstrap_resamples": cli.bootstrap_resamples,
            "temperature_policy": "checkpoint learned-fusion temperature applied to all normalized representations",
            "candidate_policy": "exact identity deduplication; first pair in fixed sampler insertion order represents duplicate candidate",
            "known_positive_policy": "target retained; other exact formal in-bank positives excluded",
            "tie_policy": "descending_score_then_lexicographic_candidate_id",
            "hash_checks": preflight["hash_checks"],
        })
        write_json(output / "checkpoint_audit.json", checkpoint_audit)
        write_json(output / "embedding_shift_summary.json", shifts)
        write_json(output / "retrieval_metrics.json", all_metrics)
        write_json(output / "paired_metric_differences.json", point_differences)
        write_json(output / "bootstrap_confidence_intervals.json", intervals)
        write_jsonl(output / "per_query_ranks.jsonl", flattened_records)
        (output / "summary.md").write_text(_summary_markdown(all_metrics, point_differences, preflight["source_summary"]), encoding="utf-8")
        with (output / "stdout.log").open("a", encoding="utf-8") as handle:
            handle.write("evaluation PASS; all model states unchanged\n")
        logger.info("evaluation complete")
        print(json.dumps({"status": "PASS", "output": str(output)}, ensure_ascii=False))
    except Exception as exc:
        with (output / "stderr.log").open("a", encoding="utf-8") as handle:
            handle.write(f"{type(exc).__name__}: {exc}\n")
        logger.exception("evaluation failed")
        raise


if __name__ == "__main__":
    main()
