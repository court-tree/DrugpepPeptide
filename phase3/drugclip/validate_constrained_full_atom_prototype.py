"""Five-sequence validation panel for constrained side-chain completion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch

from phase2.pepclip.data import (
    ATOM_NAME_TO_ID,
    ELEMENT_TO_ID,
    RESIDUE_NAME_TO_ID,
    atom_tensors,
    pad_atom_clouds,
)
from phase2.pepclip.model_3d import PepCLIP3DModel
from phase3.drugclip.batching import PHASE2_MAX_PEPTIDE_ATOMS
from phase3.drugclip.constrained_full_atom_conformer_prototype import (
    BACKBONE_COORDINATE_TOLERANCE_ANGSTROM,
    GENERATOR_ID,
    SCHEMA_VERSION,
    BackboneConstraintError,
    ConstrainedCompletionError,
    OptimizationCoverageError,
    SidechainCompletionError,
    conformer_atoms,
    generate_constrained_full_atom_conformers,
)
from phase3.drugclip.full_atom_conformer_prototype import (
    CHEMISTRY_CLASS,
    UnsupportedPeptideChemistry,
)
from phase3.drugclip.random_conformer_v3 import (
    GENERATOR_ID as FORMAL_V3_GENERATOR_ID,
    coordinate_sha256 as backbone_coordinate_sha256,
    generate_from_seed,
)


PANEL = [
    {
        "peptide_sequence": "SAVTTVVN",
        "panel_role": "short_global_etkdg_success_control",
        "sequence_length": 8,
        "theoretical_heavy_atom_count": 55,
    },
    {
        "peptide_sequence": "TLAPADGPTTDEVTLQV",
        "panel_role": "length_15_to_17_global_etkdg_success_control",
        "sequence_length": 17,
        "theoretical_heavy_atom_count": 121,
    },
    {
        "peptide_sequence": "KVSKAAADLMAYCEAHAKE",
        "panel_role": "prior_runtime_block",
        "sequence_length": 19,
        "theoretical_heavy_atom_count": 141,
    },
    {
        "peptide_sequence": "DDFTNELKAELDRYKRENQ",
        "panel_role": "prior_optimization_coverage_block",
        "sequence_length": 19,
        "theoretical_heavy_atom_count": 168,
    },
    {
        "peptide_sequence": "ENYFQAEAYNLDKVLDEFEQ",
        "panel_role": "prior_longest_maximum_heavy_block",
        "sequence_length": 20,
        "theoretical_heavy_atom_count": 175,
    },
]
SPECIAL_CHEMISTRY_CLASSES = [
    "receptor_covalent",
    "modified_or_nonstandard",
    "chemistry_insufficient",
    "known_disulfide",
    "multiple_cys_unknown",
    "cyclic_or_crosslinked",
]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_payload(
    payload: dict[str, Any], *, expected_conformers: int = 10
) -> dict[str, Any]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("schema_version_mismatch")
    if payload.get("generator_id") != GENERATOR_ID:
        raise ValueError("generator_id_mismatch")
    if payload.get("chemistry", {}).get("chemistry_class") != CHEMISTRY_CLASS:
        raise ValueError("chemistry_class_mismatch")
    if payload.get("conformer_count") != expected_conformers:
        raise ValueError("conformer_count_mismatch")
    identities = payload.get("atom_identity")
    if not isinstance(identities, list) or len(identities) != payload.get("atom_count"):
        raise ValueError("atom_identity_contract_mismatch")
    if len(identities) > PHASE2_MAX_PEPTIDE_ATOMS:
        raise ValueError("peptide_atom_cap_exceeded")
    unknowns = []
    for row in identities:
        if row["element"] not in ELEMENT_TO_ID:
            unknowns.append(f"element:{row['element']}")
        if row["atom_name"] not in ATOM_NAME_TO_ID:
            unknowns.append(f"atom_name:{row['atom_name']}")
        if row["residue_name"] not in RESIDUE_NAME_TO_ID:
            unknowns.append(f"residue_name:{row['residue_name']}")
    if unknowns:
        raise ValueError(f"pepclip_vocabulary_unknowns:{sorted(set(unknowns))}")
    coordinate_hashes = set()
    for expected_index, conformer in enumerate(payload["conformers"]):
        if conformer["conformer_index"] != expected_index:
            raise ValueError("conformer_index_mismatch")
        if conformer["mmff_status"] != 0:
            raise ValueError("unconverged_conformer_accepted")
        if conformer["input_backbone_coordinate_sha256"] != conformer[
            "output_backbone_coordinate_sha256"
        ]:
            raise ValueError("backbone_coordinate_hash_changed")
        for key in (
            "backbone_deviation_after_embedding",
            "backbone_deviation_after_optimization",
        ):
            if conformer[key]["maximum_angstrom"] > (
                BACKBONE_COORDINATE_TOLERANCE_ANGSTROM
            ):
                raise ValueError(f"backbone_coordinate_tolerance_failed:{key}")
        geometry = conformer["geometry_audit"]
        if geometry.get("status") != "PASS":
            raise ValueError("geometry_audit_failed")
        if not geometry.get("coordinate_chirality_match"):
            raise ValueError("coordinate_chirality_failed")
        coordinates = conformer["coordinates"]
        if len(coordinates) != len(identities):
            raise ValueError("coordinate_count_mismatch")
        if not torch.isfinite(torch.tensor(coordinates)).all():
            raise ValueError("nonfinite_coordinates")
        coordinate_hashes.add(conformer["coordinate_sha256"])
    if len(coordinate_hashes) != expected_conformers:
        raise ValueError("conformers_not_all_distinct")
    if payload.get("dependency_contract", {}).get(
        "target_bound_inputs_used"
    ) is not False:
        raise ValueError("target_bound_dependency_not_false")
    return {
        "status": "PASS",
        "atom_count": len(identities),
        "conformer_count": expected_conformers,
        "unique_coordinate_hashes": len(coordinate_hashes),
        "maximum_backbone_deviation_angstrom": max(
            conformer["backbone_deviation_after_optimization"][
                "maximum_angstrom"
            ]
            for conformer in payload["conformers"]
        ),
        "pepclip_vocabulary_unknown_count": 0,
        "target_bound_inputs_used": False,
    }


def cpu_egnn_forward_all(payload: dict[str, Any]) -> dict[str, Any]:
    tensor_rows = [
        atom_tensors(conformer_atoms(payload, index))
        for index in range(payload["conformer_count"])
    ]
    padded = pad_atom_clouds(
        [row["coords"] for row in tensor_rows],
        [row["elements"] for row in tensor_rows],
        [row["atom_names"] for row in tensor_rows],
        [row["residue_names"] for row in tensor_rows],
    )
    if (
        (padded["elements"] == 1).any()
        or (padded["atom_names"] == 1).any()
        or (padded["residue_names"] == 1).any()
    ):
        raise ValueError("pepclip_tensorization_contains_unk")
    model = PepCLIP3DModel(
        num_elements=max(ELEMENT_TO_ID.values()) + 1,
        num_atom_names=max(ATOM_NAME_TO_ID.values()) + 1,
        num_residue_names=max(RESIDUE_NAME_TO_ID.values()) + 1,
        encoder_type="egnn",
        element_dim=16,
        hidden_dim=32,
        output_dim=16,
        dropout=0.0,
        num_layers=1,
        num_rbf=8,
        num_neighbors=8,
    ).cpu().eval()
    with torch.inference_mode():
        embedding = model.encode_peptide(
            padded["coords"],
            padded["elements"],
            padded["mask"],
            padded["atom_names"],
            padded["residue_names"],
        )
    expected_shape = (payload["conformer_count"], 16)
    if embedding.shape != expected_shape or not torch.isfinite(embedding).all():
        raise ValueError("egnn_cpu_forward_invalid")
    return {
        "status": "PASS",
        "device": "cpu",
        "encoder_type": "egnn",
        "input_conformer_count": payload["conformer_count"],
        "input_atom_count": payload["atom_count"],
        "embedding_shape": list(embedding.shape),
        "embedding_finite": True,
        "tensorization_unk_count": 0,
    }


def run_worker(
    sequence: str,
    base_seed: int,
    backbone_plan_path: Path,
    output_path: Path,
) -> None:
    progress_path = output_path.with_suffix(".progress.json")
    started = time.perf_counter()

    def progress(row: dict[str, Any]) -> None:
        _write_json(progress_path, row)

    try:
        plan_document = json.loads(
            backbone_plan_path.read_text(encoding="utf-8")
        )
        matches = [
            row for row in plan_document["sequences"]
            if row["peptide_sequence"] == sequence
        ]
        if len(matches) != 1:
            raise BackboneConstraintError(
                f"worker_backbone_plan_not_1_to_1:{len(matches)}"
            )
        payload = generate_constrained_full_atom_conformers(
            sequence,
            num_conformers=10,
            base_seed=base_seed,
            backbone_seed_plan=matches[0]["conformers"],
            progress_callback=progress,
        )
        validation = validate_payload(payload)
        cpu_forward = cpu_egnn_forward_all(payload)
        _write_json(output_path, {
            "status": "PASS",
            "sequence": sequence,
            "elapsed_seconds": time.perf_counter() - started,
            "atom_identity_sha256": payload["atom_identity_sha256"],
            "canonical_coordinate_set_sha256": payload[
                "canonical_coordinate_set_sha256"
            ],
            "validation": validation,
            "cpu_egnn_forward": cpu_forward,
            "payload": payload,
        })
    except ConstrainedCompletionError as error:
        if isinstance(error, BackboneConstraintError):
            classification = "BACKBONE_CONSTRAINT_FAIL"
        elif isinstance(error, OptimizationCoverageError):
            classification = "OPTIMIZATION_COVERAGE_FAIL"
        else:
            classification = "SIDECHAIN_COMPLETION_FAIL"
        _write_json(output_path, {
            "status": "FAIL",
            "classification": classification,
            "sequence": sequence,
            "elapsed_seconds": time.perf_counter() - started,
            "failure_type": type(error).__name__,
            "failure_text": str(error),
            "failure_details": error.details,
            "target_bound_inputs_used": False,
        })


def launch_worker(
    sequence: str,
    base_seed: int,
    backbone_plan_path: Path,
    output_path: Path,
    *,
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output_path.with_suffix(".progress.json")
    stdout_path = output_path.with_suffix(".stdout.log")
    stderr_path = output_path.with_suffix(".stderr.log")
    command = [
        sys.executable,
        "-m",
        "phase3.drugclip.validate_constrained_full_atom_prototype",
        "--worker-sequence",
        sequence,
        "--worker-seed",
        str(base_seed),
        "--worker-backbone-plan",
        str(backbone_plan_path),
        "--worker-output",
        str(output_path),
    ]
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parents[2]),
            stdout=stdout,
            stderr=stderr,
        )
        pid = process.pid
        timed_out = False
        try:
            exit_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            exit_code = process.wait(timeout=30)
    result: dict[str, Any] = {
        "sequence": sequence,
        "pid": pid,
        "command": command,
        "timeout_seconds": timeout_seconds,
        "timed_out": timed_out,
        "exit_code": exit_code,
        "elapsed_seconds": time.perf_counter() - started,
        "leftover_process": process.poll() is None,
        "output_path": str(output_path.resolve()),
        "progress_path": str(progress_path.resolve()),
        "stdout_path": str(stdout_path.resolve()),
        "stderr_path": str(stderr_path.resolve()),
        "status": "FAIL",
    }
    if progress_path.exists():
        result["latest_progress"] = json.loads(
            progress_path.read_text(encoding="utf-8")
        )
    if not timed_out and exit_code == 0 and output_path.exists():
        result["worker_result"] = json.loads(
            output_path.read_text(encoding="utf-8")
        )
        result["status"] = result["worker_result"]["status"]
        result["classification"] = result["worker_result"].get(
            "classification"
        )
    else:
        result["classification"] = (
            "PERFORMANCE_BLOCKED" if timed_out
            else "SIDECHAIN_COMPLETION_FAIL"
        )
        result["stderr"] = stderr_path.read_text(encoding="utf-8")
    return result


def _verify_panel_chemistry(audit_path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    output = {}
    for panel_row in PANEL:
        sequence = panel_row["peptide_sequence"]
        matches = [
            row for row in rows if row["peptide_sequence"] == sequence
        ]
        if not matches or any(
            row["chemistry_classification"] != "ordinary_linear_standard"
            for row in matches
        ):
            raise ValueError(f"panel_sequence_not_ordinary_linear:{sequence}")
        output[sequence] = {
            "fixed512_occurrence_count": len(matches),
            "all_occurrences_ordinary_linear_standard": True,
            "interface_pair_ids": sorted(
                row["interface_pair_id"] for row in matches
            ),
        }
    return output


def _special_chemistry_rejections() -> list[dict[str, Any]]:
    output = []
    for classification in SPECIAL_CHEMISTRY_CLASSES:
        sequence = (
            "ACDC" if classification == "multiple_cys_unknown" else "SAVTTVVN"
        )
        rejected = False
        exception = None
        try:
                generate_constrained_full_atom_conformers(
                    sequence,
                    num_conformers=1,
                    backbone_seed_plan=[],
                chemistry_class=(
                    CHEMISTRY_CLASS
                    if classification == "multiple_cys_unknown"
                    else classification
                ),
            )
        except UnsupportedPeptideChemistry as error:
            rejected = True
            exception = f"{type(error).__name__}:{error}"
        output.append({
            "chemistry_classification": classification,
            "generation_rejected": rejected,
            "exception": exception,
        })
    return output


def _formal_v3_backbone_seed_plan(cache_path: Path) -> dict[str, Any]:
    target = {row["peptide_sequence"] for row in PANEL}
    found: dict[str, dict[str, Any]] = {}
    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            sequence = str(row.get("peptide_sequence") or "")
            if sequence not in target:
                continue
            if row.get("generator_id") != FORMAL_V3_GENERATOR_ID:
                raise ValueError(
                    f"formal_v3_generator_id_mismatch:{sequence}"
                )
            conformers = []
            for expected_index, conformer in enumerate(row["conformers"]):
                if int(conformer["conformer_index"]) != expected_index:
                    raise ValueError(
                        f"formal_v3_conformer_index_mismatch:{sequence}"
                    )
                seed = int(conformer["seed"])
                expected_sha = backbone_coordinate_sha256(
                    conformer["backbone_atoms"]
                )
                regenerated_sha = backbone_coordinate_sha256(
                    generate_from_seed(sequence, seed)
                )
                if regenerated_sha != expected_sha:
                    raise ValueError(
                        f"formal_v3_seed_reproduction_failed:{sequence}:"
                        f"{expected_index}"
                    )
                conformers.append({
                    "conformer_index": expected_index,
                    "seed": seed,
                    "attempt_index": int(conformer["attempt_index"]),
                    "split": str(row["split"]),
                    "backbone_coordinate_sha256": expected_sha,
                })
            if len(conformers) != 10:
                raise ValueError(
                    f"formal_v3_conformer_count_not_10:{sequence}"
                )
            found[sequence] = {
                "peptide_sequence": sequence,
                "split": str(row["split"]),
                "formal_v3_generator_id": str(row["generator_id"]),
                "conformers": conformers,
            }
    if set(found) != target:
        raise ValueError(
            f"formal_v3_panel_sequences_missing:{sorted(target - set(found))}"
        )
    return {
        "schema_version": "phase3-v2-formal-v3-backbone-seed-plan-v1",
        "source_cache_path": str(cache_path.resolve()),
        "formal_v3_generator_id": FORMAL_V3_GENERATOR_ID,
        "seed_reproduction_pass_count": 50,
        "sequences": [found[row["peptide_sequence"]] for row in PANEL],
    }


def _final_classification(
    sequence_results: list[dict[str, Any]],
    special_rejections: list[dict[str, Any]],
) -> str:
    runs = [
        run
        for sequence in sequence_results
        for run in sequence["runs"]
    ]
    if any(run["timed_out"] for run in runs):
        return "PERFORMANCE_BLOCKED"
    classifications = [
        run.get("classification")
        for run in runs
        if run["status"] != "PASS"
    ]
    for classification in (
        "BACKBONE_CONSTRAINT_FAIL",
        "SIDECHAIN_COMPLETION_FAIL",
        "OPTIMIZATION_COVERAGE_FAIL",
    ):
        if classification in classifications:
            return classification
    if any(not row["deterministic_double_run"] for row in sequence_results):
        return "DETERMINISM_FAIL"
    if any(
        row["cpu_egnn_forward_status"] != "PASS"
        for row in sequence_results
    ):
        return "MODEL_INPUT_FAIL"
    if not all(row["generation_rejected"] for row in special_rejections):
        return "SIDECHAIN_COMPLETION_FAIL"
    return "CONSTRAINED_COMPLETION_PASS"


def run_panel(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"prototype_output_directory_exists:{output_dir}")
    output_dir.mkdir(parents=True)
    chemistry = _verify_panel_chemistry(
        Path(args.chemistry_audit_jsonl).resolve()
    )
    backbone_plan = _formal_v3_backbone_seed_plan(
        Path(args.formal_v3_cache_jsonl).resolve()
    )
    backbone_plan_path = output_dir / "backbone_seed_plan.json"
    _write_json(backbone_plan_path, backbone_plan)
    _write_json(output_dir / "panel.json", {
        "schema_version": "phase3-v2-constrained-completion-panel-v1",
        "panel": PANEL,
        "chemistry_eligibility": chemistry,
        "per_run_timeout_seconds": args.timeout_seconds,
        "base_seed": args.seed,
    })
    special_rejections = _special_chemistry_rejections()
    _write_json(
        output_dir / "special_chemistry_rejections.json",
        special_rejections,
    )
    sequence_results = []
    all_runs = []
    for panel_index, panel_row in enumerate(PANEL):
        sequence = panel_row["peptide_sequence"]
        runs = []
        for repeat in (1, 2):
            result = launch_worker(
                sequence,
                args.seed,
                backbone_plan_path,
                output_dir
                / "workers"
                / f"{panel_index:02d}_{sequence}"
                / f"run{repeat}.json",
                timeout_seconds=args.timeout_seconds,
            )
            runs.append(result)
            all_runs.append({
                "panel_index": panel_index,
                "repeat": repeat,
                **result,
            })
            if result["status"] != "PASS":
                break
        deterministic = bool(
            len(runs) == 2
            and all(run["status"] == "PASS" for run in runs)
            and runs[0]["worker_result"]["atom_identity_sha256"]
            == runs[1]["worker_result"]["atom_identity_sha256"]
            and runs[0]["worker_result"]["canonical_coordinate_set_sha256"]
            == runs[1]["worker_result"]["canonical_coordinate_set_sha256"]
        )
        sequence_results.append({
            **panel_row,
            "runs": runs,
            "deterministic_double_run": deterministic,
            "cpu_egnn_forward_status": (
                runs[0].get("worker_result", {})
                .get("cpu_egnn_forward", {})
                .get("status")
            ),
            "run1_generation_seconds": (
                runs[0].get("worker_result", {})
                .get("payload", {})
                .get("total_generation_seconds")
            ),
            "run2_generation_seconds": (
                runs[1].get("worker_result", {})
                .get("payload", {})
                .get("total_generation_seconds")
                if len(runs) == 2 else None
            ),
        })
    classification = _final_classification(
        sequence_results, special_rejections
    )
    _write_json(output_dir / "runs.json", all_runs)
    summary = {
        "schema_version": "phase3-v2-constrained-completion-evidence-v1",
        "classification": classification,
        "sequence_count": len(PANEL),
        "all_double_run_deterministic": all(
            row["deterministic_double_run"] for row in sequence_results
        ),
        "timeout_run_count": sum(run["timed_out"] for run in all_runs),
        "leftover_process_count": sum(
            run["leftover_process"] for run in all_runs
        ),
        "all_cpu_egnn_forward_pass": all(
            row["cpu_egnn_forward_status"] == "PASS"
            for row in sequence_results
        ),
        "all_special_chemistry_rejected": all(
            row["generation_rejected"] for row in special_rejections
        ),
        "sequence_results": sequence_results,
        "target_bound_leakage": {
            "chemistry_audit_used_only_by_orchestrator": True,
            "worker_allowed_inputs": [
                "peptide_sequence",
                "conformer_index",
                "formal_v3_backbone_seed_and_hash",
                "fixed_seed",
                "generator_version",
            ],
            "receptor_interface_contact_evidence_bound_coordinates_used_by_worker": False,
        },
        "training_run": False,
        "gpu_retrieval_run": False,
        "formal_data_version_published": False,
    }
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(
        "\n".join([
            "# Constrained full-atom side-chain completion prototype",
            "",
            f"- Classification: `{classification}`",
            f"- Panel sequences: {len(PANEL)}",
            f"- Timeouts: {summary['timeout_run_count']}",
            f"- Leftover processes: {summary['leftover_process_count']}",
            f"- Double-run deterministic: {summary['all_double_run_deterministic']}",
            f"- CPU EGNN PASS: {summary['all_cpu_egnn_forward_pass']}",
            "- Training/GPU retrieval/formal release: not run",
            "",
        ]),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-sequence")
    parser.add_argument("--worker-seed", type=int, default=20260723)
    parser.add_argument("--worker-backbone-plan")
    parser.add_argument("--worker-output")
    parser.add_argument("--run-panel", action="store_true")
    parser.add_argument("--chemistry-audit-jsonl")
    parser.add_argument("--formal-v3-cache-jsonl")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args()
    if args.worker_sequence:
        if not args.worker_output or not args.worker_backbone_plan:
            parser.error(
                "--worker-output and --worker-backbone-plan are required"
            )
        run_worker(
            args.worker_sequence,
            args.worker_seed,
            Path(args.worker_backbone_plan).resolve(),
            Path(args.worker_output).resolve(),
        )
        return
    if args.run_panel:
        if (
            not args.chemistry_audit_jsonl
            or not args.formal_v3_cache_jsonl
            or not args.output_dir
        ):
            parser.error(
                "--chemistry-audit-jsonl, --formal-v3-cache-jsonl, "
                "and --output-dir are required"
            )
        summary = run_panel(args)
        print(json.dumps({
            "classification": summary["classification"],
            "output_dir": str(Path(args.output_dir).resolve()),
        }, sort_keys=True))
        return
    parser.error("select --run-panel or --worker-sequence")


if __name__ == "__main__":
    main()
