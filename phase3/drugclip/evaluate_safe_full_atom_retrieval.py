"""Frozen-model safe373 full-atom retrieval evaluation and CPU preflight.

The evaluator is read-only.  It never trains, calls backward, constructs an
optimizer, or mutates a checkpoint.  Bound-heavy inputs remain target-bound
diagnostic upper bounds and are not deployable screening inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import shlex
import statistics
import sys
from typing import Any

import numpy as np
import torch

from phase2.pepclip.data import (
    ATOM_NAME_UNK_TOKEN,
    ELEMENT_UNK_TOKEN,
    RESIDUE_NAME_UNK_TOKEN,
    atom_tensors,
)
from phase3.drugclip.build_safe_full_atom_conformer_coverage import (
    derive_candidate_contract,
)
from phase3.drugclip.evaluate_full_retrieval import (
    _file_hash,
    _load_fixed_dataset,
    _sequence_hash,
    _validate_pilot_contract,
)
from phase3.drugclip.evaluate_input_domain_ablation import (
    EXPECTED as FIXED512_EXPECTED,
    REPRESENTATIONS,
    _clone_item,
    _encode_common,
    _encode_variant,
    _input_sha,
    _json_sha,
    _load_model,
    _model_args,
    _sampler_batches,
    _score_matrix,
    arithmetic_mean_score,
    assert_1d_embeddings_identical,
    exact_evidence_join,
    model_state_sha,
)
from phase3.drugclip.io_utils import read_jsonl, write_json, write_jsonl
from phase3.drugclip.train import load_source_configs, resolve_path


SCHEMA_VERSION = "phase3-v2-safe373-full-atom-retrieval-v1"
PLAN_SCHEMA_VERSION = "phase3-v2-safe373-evaluation-plan-v1"
SAFE_QUERY_COUNT = 373
SAFE_PEPTIDE_CANDIDATE_COUNT = 265
RECEPTOR_CANDIDATE_COUNT = 512
EXPECTED_CACHE_MANIFEST_SHA256 = (
    "1BD875D3AA33CA5FADDC0A39F8C1594BF53A89944582CA700B3B5D3B65733A4C"
)
EXPECTED_DETERMINISTIC_MANIFEST_SHA256 = (
    "E213B177A98E484BAC9F3516B899EB0FF167B6A7A28A97FF0998F48C1B1C84F8"
)
EXPECTED_VALIDATION_MANIFEST_SHA256 = (
    "B159FF3B8C06465F0E678904915B591C6C1DF4564EADB093EB01F08147E972D8"
)
VARIANT_SINGLE = (
    "bound_heavy_A",
    "bound_ncac_B",
    "random_ncac_C0",
    "random_full_heavy_D0",
)
VARIANT_MEAN = ("random_ncac_Cmean10", "random_full_heavy_Dmean10")
EVALUATION_VARIANTS = VARIANT_SINGLE + VARIANT_MEAN
METRICS = (
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "median_rank",
    "mean_rank",
    "rank_standard_deviation",
    "worst_rank",
)
KNOWN_POSITIVE_POLICY = (
    "target retained; other exact formal positives intersected with the "
    "direction-specific candidate bank are excluded"
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verify_canonical_object(
    value: dict[str, Any], sha_field: str, error_prefix: str
) -> str:
    recorded = str(value.get(sha_field) or "")
    core = dict(value)
    core.pop(sha_field, None)
    if _json_sha(core) != recorded:
        raise ValueError(f"{error_prefix}_canonical_sha256_mismatch")
    return recorded


def build_safe373_plan(
    fixed_plan: list[dict[str, Any]],
    chemistry_rows: list[dict[str, Any]],
    *,
    fixed_plan_file_sha256: str,
    fixed_plan_canonical_sha256: str,
    chemistry_audit_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    contract = derive_candidate_contract(chemistry_rows)
    safe_pair_ids = set(contract["safe_query_interface_pair_ids"])
    safe_plan = [
        dict(row)
        for row in fixed_plan
        if str(row["interface_pair_id"]) in safe_pair_ids
    ]
    if len(safe_plan) != SAFE_QUERY_COUNT:
        raise ValueError(f"safe_query_count_mismatch:{len(safe_plan)}")
    if {str(row["interface_pair_id"]) for row in safe_plan} != safe_pair_ids:
        raise ValueError("safe_query_plan_membership_mismatch")
    peptide_ids = sorted(
        str(row["peptide_sequence"])
        for row in contract["safe_candidates"]
    )
    receptor_ids = sorted(
        {
            str(row["receptor_interface_id"])
            for row in fixed_plan
        }
    )
    if len(peptide_ids) != SAFE_PEPTIDE_CANDIDATE_COUNT:
        raise ValueError("safe_peptide_candidate_count_mismatch")
    if len(receptor_ids) != RECEPTOR_CANDIDATE_COUNT:
        raise ValueError("receptor_candidate_count_mismatch")
    core = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "original_fixed512_plan_file_sha256": fixed_plan_file_sha256,
        "original_fixed512_plan_canonical_sha256": (
            fixed_plan_canonical_sha256
        ),
        "chemistry_audit_file_sha256": chemistry_audit_sha256,
        "safe_query_interface_pair_ids": [
            str(row["interface_pair_id"]) for row in safe_plan
        ],
        "safe_peptide_candidate_ids": peptide_ids,
        "receptor_candidate_ids": receptor_ids,
        "known_positive_policy": KNOWN_POSITIVE_POLICY,
        "query_order_policy": "original fixed-512 relative order",
        "peptide_candidate_order_policy": "lexicographic exact sequence ID",
        "receptor_candidate_order_policy": (
            "lexicographic formal receptor interface ID"
        ),
        "counts": {
            "queries": len(safe_plan),
            "peptide_candidates": len(peptide_ids),
            "receptor_candidates": len(receptor_ids),
        },
    }
    return {
        **core,
        "plan_canonical_sha256": _json_sha(core),
    }, safe_plan


def full_atom_atoms(
    atom_identity: list[dict[str, Any]],
    coordinates: list[list[float]],
) -> list[dict[str, Any]]:
    if len(atom_identity) != len(coordinates):
        raise ValueError("full_atom_identity_coordinate_count_mismatch")
    atoms = []
    for identity, coordinate in zip(
        atom_identity, coordinates, strict=True
    ):
        if len(coordinate) != 3 or any(
            not math.isfinite(float(value)) for value in coordinate
        ):
            raise ValueError("full_atom_coordinate_nonfinite_or_wrong_shape")
        atoms.append({
            "atom_name": str(identity["atom_name"]),
            "element": str(identity["element"]),
            "residue_id": str(identity["residue_index"]),
            "residue_index": int(identity["residue_index"]),
            "residue_name": str(identity["residue_name"]),
            "x": float(coordinate[0]),
            "y": float(coordinate[1]),
            "z": float(coordinate[2]),
        })
    tensors = atom_tensors(atoms)
    if (
        (tensors["elements"] == ELEMENT_UNK_TOKEN).any()
        or (tensors["atom_names"] == ATOM_NAME_UNK_TOKEN).any()
        or (tensors["residue_names"] == RESIDUE_NAME_UNK_TOKEN).any()
    ):
        raise ValueError("full_atom_tensorization_contains_UNK")
    return atoms


def load_safe265_cache(
    cache_dir: Path,
) -> tuple[dict[str, list[list[dict[str, Any]]]], dict[str, Any]]:
    manifest = _load_json(cache_dir / "cache_manifest.json")
    if _verify_canonical_object(
        manifest, "manifest_canonical_sha256", "cache_manifest"
    ) != EXPECTED_CACHE_MANIFEST_SHA256:
        raise ValueError("safe265_cache_manifest_contract_mismatch")
    deterministic = _load_json(
        cache_dir / "deterministic_generation_manifest.json"
    )
    if _verify_canonical_object(
        deterministic,
        "deterministic_manifest_sha256",
        "deterministic_manifest",
    ) != EXPECTED_DETERMINISTIC_MANIFEST_SHA256:
        raise ValueError("safe265_deterministic_manifest_contract_mismatch")
    validation = _load_json(cache_dir / "validation_report.json")
    if _verify_canonical_object(
        validation,
        "validation_manifest_canonical_sha256",
        "validation_manifest",
    ) != EXPECTED_VALIDATION_MANIFEST_SHA256:
        raise ValueError("safe265_validation_manifest_contract_mismatch")
    if validation.get("classification") != "SAFE265_FULL_ATOM_CACHE_PASS":
        raise ValueError("safe265_validation_status_not_pass")
    index_rows = list(read_jsonl(cache_dir / "cache_index.jsonl"))
    if len(index_rows) != SAFE_PEPTIDE_CANDIDATE_COUNT:
        raise ValueError("safe265_cache_index_count_mismatch")
    output: dict[str, list[list[dict[str, Any]]]] = {}
    atom_identity_hashes = set()
    conformer_count = 0
    for index_row in index_rows:
        sequence = str(index_row["peptide_sequence"])
        cache_path = Path(index_row["cache_path"])
        if not cache_path.is_absolute():
            cache_path = cache_dir / cache_path
        if _file_hash(cache_path) != str(index_row["cache_file_sha256"]):
            raise ValueError(f"safe265_cache_file_sha_mismatch:{sequence}")
        payload = _load_json(cache_path)
        if payload["peptide_sequence"] != sequence:
            raise ValueError(f"safe265_cache_sequence_mismatch:{sequence}")
        if int(payload["atom_count"]) >= 192:
            raise ValueError(f"safe265_atom_cap_violation:{sequence}")
        if len(payload["conformers"]) != 10:
            raise ValueError(f"safe265_conformer_count_mismatch:{sequence}")
        identities = payload["atom_identity"]
        atom_identity_hashes.add(str(payload["atom_identity_sha256"]))
        conformers = []
        for expected_index, conformer in enumerate(payload["conformers"]):
            if int(conformer["conformer_index"]) != expected_index:
                raise ValueError(f"safe265_conformer_order_mismatch:{sequence}")
            conformers.append(
                full_atom_atoms(identities, conformer["coordinates"])
            )
            conformer_count += 1
        output[sequence] = conformers
    if conformer_count != 2650:
        raise ValueError("safe265_total_conformer_count_mismatch")
    return output, {
        "sequence_count": len(output),
        "conformer_count": conformer_count,
        "all_readable": True,
        "no_unk": True,
        "atom_identity_hash_count": len(atom_identity_hashes),
        "cache_manifest_canonical_sha256": EXPECTED_CACHE_MANIFEST_SHA256,
        "deterministic_manifest_sha256": (
            EXPECTED_DETERMINISTIC_MANIFEST_SHA256
        ),
        "validation_manifest_canonical_sha256": (
            EXPECTED_VALIDATION_MANIFEST_SHA256
        ),
    }


def build_full_atom_variants(
    safe_items: list[dict[str, Any]],
    cache: dict[str, list[list[dict[str, Any]]]],
) -> dict[str, list[dict[str, Any]]]:
    variants = {
        f"random_full_heavy_D{index}": [] for index in range(10)
    }
    for item in safe_items:
        sequence = str(item["peptide_sequence"])
        if sequence not in cache:
            raise ValueError(f"safe_query_target_cache_missing:{sequence}")
        for index, atoms in enumerate(cache[sequence]):
            variants[f"random_full_heavy_D{index}"].append(
                _clone_item(item, atoms)
            )
    return variants


def candidate_contract(
    items: list[dict[str, Any]],
    insertion_order: list[str],
    direction: str,
) -> tuple[list[str], dict[str, str]]:
    field = "peptide_sequence" if direction == "r2p" else "receptor_id"
    by_pair = {str(item["interface_pair_id"]): item for item in items}
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for pair_id in insertion_order:
        if pair_id in by_pair:
            groups[str(by_pair[pair_id][field])].append(pair_id)
    candidate_ids = sorted(groups)
    return candidate_ids, {
        candidate_id: groups[candidate_id][0]
        for candidate_id in candidate_ids
    }


def metric_summary(ranks: list[int]) -> dict[str, float]:
    values = np.asarray(ranks, dtype=np.float64)
    return {
        "recall_at_1": float((values <= 1).mean()),
        "recall_at_5": float((values <= 5).mean()),
        "recall_at_10": float((values <= 10).mean()),
        "mrr": float((1.0 / values).mean()),
        "median_rank": float(np.median(values)),
        "mean_rank": float(values.mean()),
        "rank_standard_deviation": float(values.std(ddof=0)),
        "worst_rank": float(values.max()),
    }


def rank_scores(
    scores: torch.Tensor,
    query_items: list[dict[str, Any]],
    candidate_ids: list[str],
    direction: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if scores.shape != (len(query_items), len(candidate_ids)):
        raise ValueError(f"score_matrix_shape_mismatch:{direction}")
    if not torch.isfinite(scores).all():
        raise ValueError("score_matrix_nonfinite")
    target_field = "peptide_sequence" if direction == "r2p" else "receptor_id"
    known_field = (
        "receptor_peptides" if direction == "r2p" else "peptide_receptors"
    )
    index = {candidate_id: offset for offset, candidate_id in enumerate(candidate_ids)}
    bank = set(candidate_ids)
    records = []
    target_missing = 0
    for query_index, item in enumerate(query_items):
        target = str(item[target_field])
        if target not in index:
            target_missing += 1
            raise ValueError(f"target_missing:{direction}:{target}")
        declared = {
            str(value)
            for value in item["known_positive_group"].get(known_field, [])
        }
        if target not in declared:
            raise ValueError(f"target_not_declared_positive:{direction}:{target}")
        excluded = (declared - {target}) & bank
        allowed = [candidate for candidate in candidate_ids if candidate not in excluded]
        ranked = sorted(
            allowed,
            key=lambda candidate: (
                -float(scores[query_index, index[candidate]]),
                candidate,
            ),
        )
        records.append({
            "interface_pair_id": str(item["interface_pair_id"]),
            "target_id": target,
            "rank": ranked.index(target) + 1,
            "target_score": float(scores[query_index, index[target]]),
            "candidate_count": len(allowed),
            "known_positive_candidates_excluded": len(excluded),
        })
    metrics: dict[str, Any] = metric_summary(
        [int(row["rank"]) for row in records]
    )
    metrics.update({
        "queries": len(records),
        "candidate_count_min": min(row["candidate_count"] for row in records),
        "candidate_count_max": max(row["candidate_count"] for row in records),
        "known_positive_exclusion_total": sum(
            row["known_positive_candidates_excluded"] for row in records
        ),
        "target_missing": target_missing,
    })
    return records, metrics


def paired_bootstrap(
    later: list[int],
    earlier: list[int],
    indices: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    if len(later) != len(earlier) or indices.shape[1] != len(later):
        raise ValueError("paired_bootstrap_query_mismatch")
    later_array = np.asarray(later, dtype=np.float64)[indices]
    earlier_array = np.asarray(earlier, dtype=np.float64)[indices]
    samples = {
        "recall_at_1": (later_array <= 1).mean(1) - (earlier_array <= 1).mean(1),
        "recall_at_5": (later_array <= 5).mean(1) - (earlier_array <= 5).mean(1),
        "recall_at_10": (later_array <= 10).mean(1) - (earlier_array <= 10).mean(1),
        "mrr": (1.0 / later_array).mean(1) - (1.0 / earlier_array).mean(1),
        "median_rank": np.median(later_array, axis=1) - np.median(earlier_array, axis=1),
        "mean_rank": later_array.mean(1) - earlier_array.mean(1),
        "rank_standard_deviation": later_array.std(1) - earlier_array.std(1),
        "worst_rank": later_array.max(1) - earlier_array.max(1),
    }
    point_later = metric_summary(later)
    point_earlier = metric_summary(earlier)
    return {
        "queries": len(later),
        "resamples": int(indices.shape[0]),
        "seed": seed,
        "later_minus_earlier": {
            name: {
                "point_estimate": point_later[name] - point_earlier[name],
                "ci_95": [
                    float(value)
                    for value in np.quantile(values, [0.025, 0.975])
                ],
                "crosses_zero": bool(
                    np.quantile(values, 0.025)
                    <= 0
                    <= np.quantile(values, 0.975)
                ),
            }
            for name, values in samples.items()
        },
    }


def preregistered_comparisons() -> list[dict[str, str]]:
    rows = []
    for model in ("phase2_baseline", "phase3_v1_epoch0"):
        for direction in ("r2p", "p2r"):
            for representation in ("3d_only", "learned_fusion"):
                for later, earlier, label in (
                    ("random_full_heavy_D0", "random_ncac_C0", "D0_minus_C0"),
                    (
                        "random_full_heavy_Dmean10",
                        "random_ncac_Cmean10",
                        "Dmean10_minus_Cmean10",
                    ),
                    (
                        "random_full_heavy_Dmean10",
                        "bound_heavy_A",
                        "Dmean10_minus_A",
                    ),
                    (
                        "random_full_heavy_Dmean10",
                        "random_full_heavy_D0",
                        "Dmean10_minus_D0",
                    ),
                ):
                    rows.append({
                        "label": label,
                        "later_model": model,
                        "earlier_model": model,
                        "later_representation": representation,
                        "earlier_representation": representation,
                        "direction": direction,
                        "later_variant": later,
                        "earlier_variant": earlier,
                    })
            rows.append({
                "label": "Dmean10_minus_one_d_only",
                "later_model": model,
                "earlier_model": model,
                "later_representation": "learned_fusion",
                "earlier_representation": "one_d_only",
                "direction": direction,
                "later_variant": "random_full_heavy_Dmean10",
                "earlier_variant": "one_d_only",
            })
    for direction in ("r2p", "p2r"):
        for representation in ("3d_only", "learned_fusion"):
            rows.append({
                "label": "phase3_epoch0_minus_phase2_Dmean10",
                "later_model": "phase3_v1_epoch0",
                "earlier_model": "phase2_baseline",
                "later_representation": representation,
                "earlier_representation": representation,
                "direction": direction,
                "later_variant": "random_full_heavy_Dmean10",
                "earlier_variant": "random_full_heavy_Dmean10",
            })
    return rows


def _checkpoint_finite(model: torch.nn.Module) -> bool:
    return all(
        torch.isfinite(value.detach().float()).all().item()
        for value in model.state_dict().values()
    )


def preflight(cli: argparse.Namespace, repo_root: Path) -> dict[str, Any]:
    pilot = resolve_path(cli.pilot_output, repo_root).resolve()
    dataset_root = resolve_path(cli.dataset_root, repo_root).resolve()
    chemistry_path = resolve_path(cli.chemistry_audit, repo_root).resolve()
    cache_dir = resolve_path(cli.safe265_cache_dir, repo_root).resolve()
    phase2_checkpoint = resolve_path(cli.phase2_checkpoint, repo_root).resolve()
    phase3_checkpoint = resolve_path(cli.phase3_checkpoint, repo_root).resolve()
    plan_path = pilot / "validation_sampling_plan.jsonl"
    checks = {
        "fixed512_plan_file_sha256": _file_hash(plan_path),
        "dataset_manifest_sha256": _file_hash(
            dataset_root / "DATA_MANIFEST.json"
        ),
        "phase2_checkpoint_sha256": _file_hash(phase2_checkpoint),
        "phase3_checkpoint_sha256": _file_hash(phase3_checkpoint),
        "chemistry_audit_file_sha256": _file_hash(chemistry_path),
    }
    for actual_key, expected_key in (
        ("fixed512_plan_file_sha256", "plan_file_sha256"),
        ("dataset_manifest_sha256", "manifest_sha256"),
        ("phase2_checkpoint_sha256", "phase2_checkpoint_sha256"),
        ("phase3_checkpoint_sha256", "phase3_checkpoint_sha256"),
    ):
        if checks[actual_key] != FIXED512_EXPECTED[expected_key]:
            raise ValueError(f"input_sha256_mismatch:{actual_key}")
    config, subset, fixed_plan = _validate_pilot_contract(pilot)
    chemistry_rows = list(read_jsonl(chemistry_path))
    safe_plan_record, safe_plan = build_safe373_plan(
        fixed_plan,
        chemistry_rows,
        fixed_plan_file_sha256=checks["fixed512_plan_file_sha256"],
        fixed_plan_canonical_sha256=str(
            config["fixed_validation_plan_sha256"]
        ),
        chemistry_audit_sha256=checks["chemistry_audit_file_sha256"],
    )
    base_dataset = _load_fixed_dataset(config, subset, fixed_plan)
    full_variants, full_audits, full_source = exact_evidence_join(
        fixed_plan,
        base_dataset,
        resolve_path(cli.candidate_evidence_jsonl, repo_root).resolve(),
        resolve_path(cli.expanded_evidence_jsonl, repo_root).resolve(),
        resolve_path(cli.mmcif_root, repo_root).resolve(),
        resolve_path(cli.qbiolip_root, repo_root).resolve(),
        resolve_path(cli.biolip_root, repo_root).resolve(),
        dataset_root / "dependencies" / "biological_pairs.jsonl",
    )
    safe_ids = {
        str(row["interface_pair_id"]) for row in safe_plan
    }
    safe_indices = [
        index
        for index, row in enumerate(fixed_plan)
        if str(row["interface_pair_id"]) in safe_ids
    ]
    safe_variants = {
        "bound_heavy_A": [
            full_variants["bound_all_heavy"][index] for index in safe_indices
        ],
        "bound_ncac_B": [
            full_variants["bound_backbone_ncac"][index]
            for index in safe_indices
        ],
    }
    for conformer_index in range(10):
        safe_variants[f"random_ncac_C{conformer_index}"] = [
            full_variants[f"random_backbone_conformer{conformer_index}"][index]
            for index in safe_indices
        ]
    safe_audits = [full_audits[index] for index in safe_indices]
    if len(safe_audits) != SAFE_QUERY_COUNT:
        raise ValueError("safe373_A_B_audit_count_mismatch")
    cache, cache_audit = load_safe265_cache(cache_dir)
    safe_variants.update(
        build_full_atom_variants(
            safe_variants["random_ncac_C0"], cache
        )
    )
    full_c0 = full_variants["random_backbone_conformer0"]
    safe_c0 = safe_variants["random_ncac_C0"]
    full_batches = _sampler_batches(full_c0, cli.batch_size)
    safe_batches = _sampler_batches(safe_c0, cli.batch_size)
    full_order = [
        str(full_c0[index]["interface_pair_id"])
        for batch in full_batches for index in batch
    ]
    safe_order = [
        str(safe_c0[index]["interface_pair_id"])
        for batch in safe_batches for index in batch
    ]
    r2p_ids, r2p_representatives = candidate_contract(
        safe_c0, safe_order, "r2p"
    )
    p2r_ids, p2r_representatives = candidate_contract(
        full_c0, full_order, "p2r"
    )
    if r2p_ids != safe_plan_record["safe_peptide_candidate_ids"]:
        raise ValueError("safe265_candidate_order_or_membership_mismatch")
    if p2r_ids != safe_plan_record["receptor_candidate_ids"]:
        raise ValueError("receptor512_candidate_order_or_membership_mismatch")
    source_configs = load_source_configs(
        phase2_checkpoint,
        resolve_path(cli.source_model_configs, repo_root),
        repo_root,
    )
    device = torch.device("cpu")
    train_args = _model_args(config, device)
    checkpoint_audit = {}
    score_smoke = {}
    smoke_indices = []
    seen_sequences = set()
    for index, item in enumerate(safe_c0):
        sequence = str(item["peptide_sequence"])
        if sequence not in seen_sequences:
            smoke_indices.append(index)
            seen_sequences.add(sequence)
        if len(smoke_indices) == 2:
            break
    smoke_common_items = [safe_c0[index] for index in smoke_indices]
    smoke_batches = [list(range(len(smoke_common_items)))]
    smoke_ids = [str(item["interface_pair_id"]) for item in smoke_common_items]
    smoke_candidates = [
        str(item["peptide_sequence"]) for item in smoke_common_items
    ]
    smoke_representatives = {
        candidate: pair_id
        for candidate, pair_id in zip(
            smoke_candidates, smoke_ids, strict=True
        )
    }
    for label, checkpoint in (
        ("phase2_baseline", phase2_checkpoint),
        ("phase3_v1_epoch0", phase3_checkpoint),
    ):
        model = _load_model(
            label,
            checkpoint,
            phase2_checkpoint,
            source_configs,
            train_args,
            repo_root,
            device,
        )
        before = model_state_sha(model)
        if not _checkpoint_finite(model):
            raise ValueError(f"checkpoint_model_state_nonfinite:{label}")
        common = _encode_common(
            model, smoke_common_items, smoke_batches, device
        )
        matrices = []
        for conformer_index in range(10):
            items = [
                safe_variants[
                    f"random_full_heavy_D{conformer_index}"
                ][index]
                for index in smoke_indices
            ]
            encoded = _encode_variant(
                model, items, smoke_batches, common, device
            )
            for pair_id in smoke_ids:
                assert_1d_embeddings_identical(
                    common[pair_id]["peptide_1d"],
                    encoded[pair_id]["peptide_1d"],
                )
            matrices.append(
                _score_matrix(
                    encoded,
                    smoke_ids,
                    smoke_candidates,
                    smoke_representatives,
                    "r2p",
                    "learned_fusion",
                    model.temperature,
                )
            )
        mean_matrix = arithmetic_mean_score(matrices)
        if mean_matrix.shape != (2, 2):
            raise ValueError("preflight_score_matrix_shape_mismatch")
        after = model_state_sha(model)
        if before != after:
            raise ValueError(f"preflight_model_state_changed:{label}")
        checkpoint_audit[label] = {
            "file_sha256": _file_hash(checkpoint),
            "model_state_sha256_before": before,
            "model_state_sha256_after": after,
            "unchanged": True,
            "finite": True,
        }
        score_smoke[label] = {
            "D0_shape": list(matrices[0].shape),
            "Dmean10_shape": list(mean_matrix.shape),
            "Dmean10_is_arithmetic_mean_of_ten_score_matrices": bool(
                torch.equal(
                    mean_matrix,
                    torch.stack(matrices, dim=0).mean(dim=0),
                )
            ),
            "coordinates_were_not_averaged": True,
        }
        del model
    return {
        "safe_plan_record": safe_plan_record,
        "safe_plan": safe_plan,
        "safe_variants": safe_variants,
        "full_c0": full_c0,
        "safe_audits": safe_audits,
        "full_source_summary": full_source,
        "cache_audit": cache_audit,
        "candidate_contracts": {
            "r2p": (r2p_ids, r2p_representatives),
            "p2r": (p2r_ids, p2r_representatives),
        },
        "checkpoint_audit": checkpoint_audit,
        "score_smoke": score_smoke,
        "hash_checks": checks,
        "config": config,
        "phase2_checkpoint": phase2_checkpoint,
        "phase3_checkpoint": phase3_checkpoint,
        "source_configs": source_configs,
        "AB_contract": {
            "selection_logic": (
                "full fixed-512 exact-evidence join first, then safe-ID filter"
            ),
            "exact_evidence_matches": len(safe_audits),
            "sequence_mismatches": 0,
            "target_missing": 0,
            "subset_failures": 0,
            "candidate_representation_uses_existing_sampler_first_pair": True,
            "bound_A_is_target_bound_non_deployable_upper_bound": True,
        },
        "no_training_path": {
            "training": False,
            "backward": False,
            "optimizer": False,
            "formal_gpu_evaluation": False,
        },
    }


def _formal_evaluate(
    cli: argparse.Namespace,
    repo_root: Path,
    flight: dict[str, Any],
    output: Path,
) -> None:
    device = torch.device(cli.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output.mkdir(parents=True)
    command = shlex.join([
        sys.executable,
        "-m",
        "phase3.drugclip.evaluate_safe_full_atom_retrieval",
        *sys.argv[1:],
    ])
    write_json(output / "safe373_evaluation_plan.json", flight["safe_plan_record"])
    write_jsonl(output / "input_variant_audit.jsonl", flight["safe_audits"])
    train_args = _model_args(flight["config"], device)
    safe_items = flight["safe_variants"]["random_ncac_C0"]
    full_items = flight["full_c0"]
    safe_batches = _sampler_batches(safe_items, cli.batch_size)
    full_batches = _sampler_batches(full_items, cli.batch_size)
    safe_ids = [str(item["interface_pair_id"]) for item in safe_items]
    all_metrics: dict[str, Any] = {}
    all_records: dict[str, Any] = {}
    rank_store: dict[str, Any] = {}
    checkpoint_audit = {"read_only": True, "checkpoints": {}}
    for label, checkpoint in (
        ("phase2_baseline", flight["phase2_checkpoint"]),
        ("phase3_v1_epoch0", flight["phase3_checkpoint"]),
    ):
        model = _load_model(
            label,
            checkpoint,
            flight["phase2_checkpoint"],
            flight["source_configs"],
            train_args,
            repo_root,
            device,
        )
        before = model_state_sha(model)
        full_common = _encode_common(model, full_items, full_batches, device)
        receptor_encoded = _encode_variant(
            model, full_items, full_batches, full_common, device
        )
        safe_common = {
            pair_id: full_common[pair_id] for pair_id in safe_ids
        }
        encoded = {}
        for variant, items in flight["safe_variants"].items():
            safe_encoded = _encode_variant(
                model, items, safe_batches, safe_common, device
            )
            merged = dict(receptor_encoded)
            merged.update(safe_encoded)
            encoded[variant] = merged
        model_metrics = {
            "one_d_only": {},
            "3d_only": {},
            "learned_fusion": {},
        }
        model_records = {
            "one_d_only": {},
            "3d_only": {},
            "learned_fusion": {},
        }
        rank_store[label] = {
            "one_d_only": {},
            "3d_only": {},
            "learned_fusion": {},
        }
        for representation in ("one_d_only", "3d_only", "learned_fusion"):
            variants = (
                ("one_d_only",)
                if representation == "one_d_only"
                else EVALUATION_VARIANTS
            )
            internal_representation = (
                "1d_only"
                if representation == "one_d_only"
                else representation
            )
            for variant in variants:
                model_metrics[representation][variant] = {}
                model_records[representation][variant] = {}
                rank_store[label][representation][variant] = {}
                for direction in ("r2p", "p2r"):
                    candidate_ids, representatives = flight[
                        "candidate_contracts"
                    ][direction]
                    if variant == "one_d_only":
                        scores = _score_matrix(
                            encoded["random_ncac_C0"],
                            safe_ids,
                            candidate_ids,
                            representatives,
                            direction,
                            internal_representation,
                            model.temperature,
                        )
                    elif variant.endswith("mean10"):
                        prefix = (
                            "random_ncac_C"
                            if variant == "random_ncac_Cmean10"
                            else "random_full_heavy_D"
                        )
                        scores = arithmetic_mean_score([
                            _score_matrix(
                                encoded[f"{prefix}{index}"],
                                safe_ids,
                                candidate_ids,
                                representatives,
                                direction,
                                internal_representation,
                                model.temperature,
                            )
                            for index in range(10)
                        ])
                    else:
                        scores = _score_matrix(
                            encoded[variant],
                            safe_ids,
                            candidate_ids,
                            representatives,
                            direction,
                            internal_representation,
                            model.temperature,
                        )
                    records, metrics = rank_scores(
                        scores, safe_items, candidate_ids, direction
                    )
                    model_metrics[representation][variant][direction] = metrics
                    model_records[representation][variant][direction] = records
                    rank_store[label][representation][variant][direction] = [
                        int(row["rank"]) for row in records
                    ]
        after = model_state_sha(model)
        if before != after:
            raise RuntimeError(f"model_state_changed:{label}")
        checkpoint_audit["checkpoints"][label] = {
            "file_sha256": _file_hash(checkpoint),
            "model_state_sha256_before": before,
            "model_state_sha256_after": after,
            "unchanged": True,
        }
        all_metrics[label] = model_metrics
        all_records[label] = model_records
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    rng = np.random.default_rng(cli.bootstrap_seed)
    indices = rng.integers(
        0,
        SAFE_QUERY_COUNT,
        size=(cli.bootstrap_resamples, SAFE_QUERY_COUNT),
        dtype=np.int32,
    )
    intervals = {}
    for comparison in preregistered_comparisons():
        later = rank_store[
            comparison["later_model"]
        ][comparison["later_representation"]][
            comparison["later_variant"]
        ][comparison["direction"]]
        earlier = rank_store[
            comparison["earlier_model"]
        ][comparison["earlier_representation"]][
            comparison["earlier_variant"]
        ][comparison["direction"]]
        key = "/".join([
            comparison["label"],
            comparison["later_model"],
            comparison["later_representation"],
            comparison["direction"],
        ])
        intervals[key] = {
            "comparison": comparison,
            **paired_bootstrap(
                later, earlier, indices, cli.bootstrap_seed
            ),
        }
    flattened = []
    for model, model_rows in all_records.items():
        for representation, representation_rows in model_rows.items():
            for variant, directions in representation_rows.items():
                for direction, rows in directions.items():
                    flattened.extend({
                        "model": model,
                        "representation": representation,
                        "variant": variant,
                        "direction": direction,
                        **row,
                    } for row in rows)
    write_json(output / "evaluation_config.json", {
        "schema_version": SCHEMA_VERSION,
        "command": command,
        "query_count": SAFE_QUERY_COUNT,
        "r2p_candidate_count": SAFE_PEPTIDE_CANDIDATE_COUNT,
        "p2r_candidate_count": RECEPTOR_CANDIDATE_COUNT,
        "bootstrap_seed": cli.bootstrap_seed,
        "bootstrap_resamples": cli.bootstrap_resamples,
        "Dmean10_policy": "arithmetic mean of ten score matrices",
        "bound_A_policy": "target-bound non-deployable diagnostic upper bound",
        "hash_checks": flight["hash_checks"],
        "cache_audit": flight["cache_audit"],
    })
    write_json(output / "checkpoint_audit.json", checkpoint_audit)
    write_json(output / "retrieval_metrics.json", all_metrics)
    write_json(output / "bootstrap_confidence_intervals.json", intervals)
    write_jsonl(output / "per_query_ranks.jsonl", flattened)
    print(json.dumps({"status": "PASS", "output": str(output)}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-output", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--chemistry-audit", required=True)
    parser.add_argument("--safe265-cache-dir", required=True)
    parser.add_argument("--candidate-evidence-jsonl", required=True)
    parser.add_argument("--expanded-evidence-jsonl", required=True)
    parser.add_argument("--mmcif-root", required=True)
    parser.add_argument("--qbiolip-root", required=True)
    parser.add_argument("--biolip-root", required=True)
    parser.add_argument("--phase2-checkpoint", required=True)
    parser.add_argument("--phase3-checkpoint", required=True)
    parser.add_argument("--source-model-configs", required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> None:
    cli = build_parser().parse_args()
    if cli.bootstrap_resamples != 10000:
        raise ValueError("safe373 bootstrap requires 10000 resamples")
    repo_root = Path(__file__).resolve().parents[2]
    output = resolve_path(cli.output_dir, repo_root).resolve()
    if output.exists():
        raise FileExistsError(f"output_directory_already_exists:{output}")
    flight = preflight(cli, repo_root)
    if cli.preflight_only:
        output.mkdir(parents=True)
        write_json(
            output / "safe373_evaluation_plan.json",
            flight["safe_plan_record"],
        )
        write_jsonl(
            output / "input_variant_audit.jsonl", flight["safe_audits"]
        )
        report = {
            "status": "PASS",
            "classification": "SAFE373_FULL_ATOM_RETRIEVAL_PREFLIGHT_PASS",
            "plan": flight["safe_plan_record"],
            "cache": flight["cache_audit"],
            "AB_contract": flight["AB_contract"],
            "checkpoint_audit": flight["checkpoint_audit"],
            "score_matrix_smoke": flight["score_smoke"],
            "no_training_path": flight["no_training_path"],
            "formal_gpu_evaluation_run": False,
        }
        write_json(output / "preflight_report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    _formal_evaluate(cli, repo_root, flight, output)


if __name__ == "__main__":
    main()
