"""Five-sequence validation panel for FASPR fixed-backbone packing."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from phase2.pepclip.data import (
    ATOM_NAME_TO_ID,
    ELEMENT_TO_ID,
    RESIDUE_NAME_TO_ID,
    atom_tensors,
)
from phase3.drugclip.batching import PHASE2_MAX_PEPTIDE_ATOMS
from phase3.drugclip.faspr_full_atom_conformer_prototype import (
    GENERATOR_ID,
    SCHEMA_VERSION,
    BackboneOReconstructionError,
    FASPRInputContractError,
    FASPRPrototypeError,
    PackingCoverageError,
    conformer_atoms,
    generate_faspr_full_atom_conformers,
)
from phase3.drugclip.full_atom_conformer_prototype import (
    CHEMISTRY_CLASS,
    REQUIRED_HEAVY_ATOMS,
    UnsupportedPeptideChemistry,
    classify_sequence,
)
from phase3.drugclip.validate_constrained_full_atom_prototype import (
    PANEL as CONSTRAINED_PANEL,
    _formal_v3_backbone_seed_plan,
    _verify_panel_chemistry,
    cpu_egnn_forward_all,
)


PANEL = [dict(row) for row in CONSTRAINED_PANEL]
SPECIAL_CHEMISTRY_CLASSES = [
    "receptor_covalent",
    "modified_or_nonstandard",
    "chemistry_insufficient",
    "known_disulfide",
    "multiple_cys_unknown",
    "cyclic_or_crosslinked",
]
EXPECTED_FASPR_COMMIT = "0d55732fd6307f373018c6bddd842291c355c5f7"
EXPECTED_FASPR_BINARY_SHA256 = (
    "EC5A10ACBDB97E377B0A6263CC4D94192A0E3F5D8189D8726C889C1BA935EFA3"
)
EXPECTED_FASPR_REMOTE = "https://github.com/tommyhuangthu/FASPR.git"
SEQUENCE_TIMEOUT_SECONDS = 60


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def verify_faspr_tool(
    root: Path,
    *,
    expected_commit: str = EXPECTED_FASPR_COMMIT,
    expected_binary_sha256: str = EXPECTED_FASPR_BINARY_SHA256,
) -> dict[str, Any]:
    root = root.resolve()
    executable = root / "FASPR"
    library = root / "dun2010bbdep.bin"
    license_path = root / "LICENSE"
    for path in (executable, library, license_path, root / "README.md"):
        if not path.is_file():
            raise FASPRInputContractError(f"faspr_tool_file_missing:{path}")
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    remote = subprocess.run(
        ["git", "-C", str(root), "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    binary_sha = _sha256(executable)
    license_text = license_path.read_text(encoding="utf-8")
    if commit != expected_commit:
        raise FASPRInputContractError(
            f"faspr_commit_mismatch:{commit}:{expected_commit}"
        )
    if remote != EXPECTED_FASPR_REMOTE:
        raise FASPRInputContractError(
            f"faspr_remote_mismatch:{remote}:{EXPECTED_FASPR_REMOTE}"
        )
    if binary_sha != expected_binary_sha256.upper():
        raise FASPRInputContractError(
            f"faspr_binary_sha_mismatch:{binary_sha}:"
            f"{expected_binary_sha256.upper()}"
        )
    if not license_text.startswith("MIT License"):
        raise FASPRInputContractError("faspr_license_not_mit")
    return {
        "status": "PASS",
        "official_remote": remote,
        "commit_sha": commit,
        "license": "MIT",
        "license_sha256": _sha256(license_path),
        "compiler": "g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0",
        "compile_command": "g++ -O3 --fast-math -o FASPR src/*.cpp",
        "binary_path": str(executable),
        "binary_sha256": binary_sha,
        "rotamer_library_path": str(library),
        "rotamer_library_sha256": _sha256(library),
        "binary_and_library_same_directory": executable.parent == library.parent,
        "official_example": {
            "input": str(root / "example" / "1mol.pdb"),
            "run_a_sha256": _sha256(root / "example" / "example_pass_a.pdb"),
            "run_b_sha256": _sha256(root / "example" / "example_pass_b.pdb"),
            "deterministic": (
                (root / "example" / "example_pass_a.pdb").read_bytes()
                == (root / "example" / "example_pass_b.pdb").read_bytes()
            ),
            "run_a_stderr_bytes": (
                root / "example" / "example_pass_a.stderr.log"
            ).stat().st_size,
            "run_b_stderr_bytes": (
                root / "example" / "example_pass_b.stderr.log"
            ).stat().st_size,
        },
    }


def validate_payload(
    payload: dict[str, Any],
    *,
    expected_conformers: int = 10,
) -> dict[str, Any]:
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("faspr_schema_mismatch")
    if payload["generator_id"] != GENERATOR_ID:
        raise ValueError("faspr_generator_mismatch")
    if payload["conformer_count"] != expected_conformers:
        raise ValueError("faspr_conformer_count_mismatch")
    if payload["atom_count"] > PHASE2_MAX_PEPTIDE_ATOMS:
        raise ValueError(
            f"faspr_atom_cap_blocked:{payload['atom_count']}:"
            f"{PHASE2_MAX_PEPTIDE_ATOMS}"
        )
    required_by_residue = {}
    for identity in payload["atom_identity"]:
        required_by_residue.setdefault(
            (identity["residue_index"], identity["residue_name"]), set()
        ).add(identity["atom_name"])
    for (residue_index, residue_name), names in required_by_residue.items():
        required = set(REQUIRED_HEAVY_ATOMS[residue_name])
        if residue_index == len(payload["peptide_sequence"]):
            required.add("OXT")
        if not required <= names:
            raise ValueError(
                f"faspr_complete_heavy_atoms_missing:{residue_index}:"
                f"{sorted(required - names)}"
            )
    coordinate_hashes = []
    maximum_backbone_deviation = 0.0
    faspr_exit_codes = []
    faspr_seconds = []
    for conformer in payload["conformers"]:
        coordinate_hashes.append(conformer["coordinate_sha256"])
        maximum_backbone_deviation = max(
            maximum_backbone_deviation,
            conformer["maximum_backbone_deviation_angstrom"],
        )
        if (
            conformer["input_backbone_coordinate_sha256"]
            != conformer["output_backbone_coordinate_sha256"]
        ):
            raise ValueError("faspr_backbone_hash_changed")
        if conformer["oxygen_reconstruction_audit"]["status"] != "PASS":
            raise ValueError("faspr_oxygen_reconstruction_not_pass")
        for oxygen in conformer["oxygen_reconstruction_audit"]["residues"]:
            if not 1.229 <= oxygen["c_o_length_angstrom"] <= 1.233:
                raise ValueError("faspr_carbonyl_bond_invalid")
            if not 90.0 <= oxygen["ca_c_o_angle_degrees"] <= 150.0:
                raise ValueError("faspr_carbonyl_angle_invalid")
            if "peptide_c_n_length_angstrom" in oxygen and not (
                1.20 <= oxygen["peptide_c_n_length_angstrom"] <= 1.45
            ):
                raise ValueError("faspr_peptide_bond_invalid")
        if conformer["geometry_audit"]["status"] != "PASS":
            raise ValueError("faspr_geometry_not_pass")
        faspr_exit_codes.append(conformer["faspr"]["exit_code"])
        faspr_seconds.append(conformer["faspr"]["elapsed_seconds"])
    if maximum_backbone_deviation != 0.0:
        raise ValueError("faspr_backbone_not_exactly_fixed")
    if len(set(coordinate_hashes)) != expected_conformers:
        raise ValueError("faspr_conformers_not_distinct")
    if any(code != 0 for code in faspr_exit_codes):
        raise ValueError("faspr_nonzero_exit")
    unknowns = []
    for identity in payload["atom_identity"]:
        if identity["element"] not in ELEMENT_TO_ID:
            unknowns.append(f"element:{identity['element']}")
        if identity["atom_name"] not in ATOM_NAME_TO_ID:
            unknowns.append(f"atom_name:{identity['atom_name']}")
        if identity["residue_name"] not in RESIDUE_NAME_TO_ID:
            unknowns.append(f"residue_name:{identity['residue_name']}")
    unknown_count = len(unknowns)
    if unknown_count:
        raise ValueError(f"faspr_tensorization_unknown:{sorted(set(unknowns))}")
    return {
        "status": "PASS",
        "conformer_count": expected_conformers,
        "atom_count": payload["atom_count"],
        "unique_coordinate_hashes": len(set(coordinate_hashes)),
        "maximum_backbone_deviation_angstrom": maximum_backbone_deviation,
        "faspr_exit_codes": faspr_exit_codes,
        "faspr_conformer_seconds": faspr_seconds,
        "maximum_faspr_conformer_seconds": max(faspr_seconds),
        "tensorization_unk_count": unknown_count,
        "target_bound_inputs_used": False,
    }


def _special_chemistry_rejections() -> list[dict[str, Any]]:
    output = []
    for classification in SPECIAL_CHEMISTRY_CLASSES:
        sequence = "ACDC" if classification == "multiple_cys_unknown" else "SAVTTVVN"
        try:
            classify_sequence(
                sequence,
                chemistry_class=(
                    CHEMISTRY_CLASS
                    if classification == "multiple_cys_unknown"
                    else classification
                ),
            )
        except UnsupportedPeptideChemistry as error:
            output.append({
                "chemistry_classification": classification,
                "generation_rejected": True,
                "exception": f"{type(error).__name__}:{error}",
            })
        else:
            output.append({
                "chemistry_classification": classification,
                "generation_rejected": False,
                "exception": None,
            })
    return output


def _worker_classification(error: Exception) -> str:
    if isinstance(error, BackboneOReconstructionError):
        return "BACKBONE_O_RECONSTRUCTION_FAIL"
    if isinstance(error, FASPRInputContractError):
        return "FASPR_INPUT_CONTRACT_FAIL"
    if isinstance(error, PackingCoverageError):
        return "PACKING_COVERAGE_FAIL"
    if isinstance(error, TimeoutError):
        return "PERFORMANCE_BLOCKED"
    return "MODEL_INPUT_FAIL"


def run_worker(
    sequence: str,
    backbone_plan_path: Path,
    output_path: Path,
    tool_contract_path: Path,
) -> None:
    started = time.perf_counter()
    progress_path = output_path.with_suffix(".progress.json")

    def progress(row: dict[str, Any]) -> None:
        _write_json(progress_path, row)

    try:
        plan = json.loads(backbone_plan_path.read_text(encoding="utf-8"))
        tool = json.loads(tool_contract_path.read_text(encoding="utf-8"))
        matches = [
            row for row in plan["sequences"]
            if row["peptide_sequence"] == sequence
        ]
        if len(matches) != 1:
            raise FASPRInputContractError(
                f"worker_backbone_plan_not_1_to_1:{len(matches)}"
            )
        payload = generate_faspr_full_atom_conformers(
            sequence,
            backbone_seed_plan=matches[0]["conformers"],
            work_dir=output_path.parent / "faspr_conformers",
            faspr_executable=Path(tool["binary_path"]),
            faspr_commit_sha=tool["commit_sha"],
            faspr_binary_sha256=tool["binary_sha256"],
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
    except Exception as error:
        _write_json(output_path, {
            "status": "FAIL",
            "classification": _worker_classification(error),
            "sequence": sequence,
            "elapsed_seconds": time.perf_counter() - started,
            "failure_type": type(error).__name__,
            "failure_text": str(error),
            "failure_details": (
                error.details if isinstance(error, FASPRPrototypeError) else {}
            ),
            "target_bound_inputs_used": False,
        })


def _faspr_processes() -> list[str]:
    result = subprocess.run(
        ["wsl.exe", "--exec", "pgrep", "-af", "^.*/FASPR( |$)"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def launch_worker(
    sequence: str,
    backbone_plan_path: Path,
    output_path: Path,
    tool_contract_path: Path,
    *,
    timeout_seconds: int = SEQUENCE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = output_path.with_suffix(".stdout.log")
    stderr_path = output_path.with_suffix(".stderr.log")
    command = [
        sys.executable,
        "-m",
        "phase3.drugclip.validate_faspr_full_atom_prototype",
        "--worker-sequence",
        sequence,
        "--worker-backbone-plan",
        str(backbone_plan_path),
        "--worker-output",
        str(output_path),
        "--worker-tool-contract",
        str(tool_contract_path),
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
        timed_out = False
        try:
            exit_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            subprocess.run(
                ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                text=True,
                check=False,
            )
            exit_code = process.wait(timeout=10)
    leftovers = _faspr_processes()
    result: dict[str, Any] = {
        "sequence": sequence,
        "pid": process.pid,
        "command": command,
        "timeout_seconds": timeout_seconds,
        "timed_out": timed_out,
        "exit_code": exit_code,
        "elapsed_seconds": time.perf_counter() - started,
        "leftover_processes": leftovers,
        "stdout_path": str(stdout_path.resolve()),
        "stderr_path": str(stderr_path.resolve()),
        "output_path": str(output_path.resolve()),
        "status": "FAIL",
    }
    if not timed_out and exit_code == 0 and output_path.exists():
        result["worker_result"] = json.loads(
            output_path.read_text(encoding="utf-8")
        )
        result["status"] = result["worker_result"]["status"]
        result["classification"] = result["worker_result"].get("classification")
    else:
        result["classification"] = (
            "PERFORMANCE_BLOCKED" if timed_out else "PACKING_COVERAGE_FAIL"
        )
        result["stderr"] = stderr_path.read_text(encoding="utf-8")
    return result


def _final_classification(
    sequence_results: list[dict[str, Any]],
    special_rejections: list[dict[str, Any]],
) -> str:
    runs = [run for row in sequence_results for run in row["runs"]]
    if any(run["timed_out"] for run in runs):
        return "PERFORMANCE_BLOCKED"
    classifications = [
        run.get("classification") for run in runs if run["status"] != "PASS"
    ]
    for classification in (
        "BACKBONE_O_RECONSTRUCTION_FAIL",
        "FASPR_INPUT_CONTRACT_FAIL",
        "PACKING_COVERAGE_FAIL",
        "PERFORMANCE_BLOCKED",
        "DETERMINISM_FAIL",
        "MODEL_INPUT_FAIL",
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
        return "FASPR_INPUT_CONTRACT_FAIL"
    return "FASPR_FIXED_BACKBONE_PASS"


def run_panel(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"prototype_output_directory_exists:{output_dir}")
    output_dir.mkdir(parents=True)
    tool = verify_faspr_tool(Path(args.faspr_root))
    chemistry = _verify_panel_chemistry(Path(args.chemistry_audit_jsonl).resolve())
    backbone_plan = _formal_v3_backbone_seed_plan(
        Path(args.formal_v3_cache_jsonl).resolve()
    )
    backbone_plan_path = output_dir / "backbone_seed_plan.json"
    tool_contract_path = output_dir / "faspr_tool_contract.json"
    _write_json(backbone_plan_path, backbone_plan)
    _write_json(tool_contract_path, tool)
    _write_json(output_dir / "panel.json", {
        "schema_version": "phase3-v2-faspr-fixed-backbone-panel-v1",
        "panel": PANEL,
        "chemistry_eligibility": chemistry,
        "per_conformer_timeout_seconds": 30,
        "per_sequence_repeat_timeout_seconds": args.timeout_seconds,
    })
    special_rejections = _special_chemistry_rejections()
    _write_json(
        output_dir / "special_chemistry_rejections.json",
        special_rejections,
    )
    sequence_results = []
    all_runs = []
    stop = False
    for panel_index, panel_row in enumerate(PANEL):
        if stop:
            break
        runs = []
        sequence = panel_row["peptide_sequence"]
        for repeat in (1, 2):
            result = launch_worker(
                sequence,
                backbone_plan_path,
                output_dir / "workers" / f"{panel_index:02d}_{sequence}"
                / f"run{repeat}.json",
                tool_contract_path,
                timeout_seconds=args.timeout_seconds,
            )
            runs.append(result)
            all_runs.append({"panel_index": panel_index, "repeat": repeat, **result})
            if result["status"] != "PASS" or result["leftover_processes"]:
                stop = True
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
        })
    classification = _final_classification(sequence_results, special_rejections)
    _write_json(output_dir / "runs.json", all_runs)
    summary = {
        "schema_version": "phase3-v2-faspr-fixed-backbone-evidence-v1",
        "classification": classification,
        "sequence_count_completed_or_attempted": len(sequence_results),
        "formal_backbone_seed_reproduction_pass_count": backbone_plan[
            "seed_reproduction_pass_count"
        ],
        "all_double_run_deterministic": (
            len(sequence_results) == len(PANEL)
            and all(row["deterministic_double_run"] for row in sequence_results)
        ),
        "timeout_run_count": sum(run["timed_out"] for run in all_runs),
        "leftover_process_count": sum(
            bool(run["leftover_processes"]) for run in all_runs
        ),
        "all_cpu_egnn_forward_pass": (
            len(sequence_results) == len(PANEL)
            and all(
                row["cpu_egnn_forward_status"] == "PASS"
                for row in sequence_results
            )
        ),
        "all_special_chemistry_rejected": all(
            row["generation_rejected"] for row in special_rejections
        ),
        "tool_contract": tool,
        "sequence_results": sequence_results,
        "target_bound_leakage": {
            "chemistry_audit_used_only_by_orchestrator": True,
            "worker_allowed_inputs": list(
                inspect.signature(
                    generate_faspr_full_atom_conformers
                ).parameters
            ),
            "receptor_interface_contact_evidence_bound_coordinates_used_by_worker": False,
        },
        "training_run": False,
        "gpu_retrieval_run": False,
        "formal_data_version_published": False,
    }
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(
        "\n".join([
            "# FASPR fixed-backbone full-atom prototype",
            "",
            f"- Classification: `{classification}`",
            f"- Formal backbone hashes reproduced: "
            f"{summary['formal_backbone_seed_reproduction_pass_count']}/50",
            f"- Panel sequences attempted: {len(sequence_results)}/5",
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
    parser.add_argument("--worker-backbone-plan")
    parser.add_argument("--worker-output")
    parser.add_argument("--worker-tool-contract")
    parser.add_argument("--run-panel", action="store_true")
    parser.add_argument("--chemistry-audit-jsonl")
    parser.add_argument("--formal-v3-cache-jsonl")
    parser.add_argument("--faspr-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--timeout-seconds", type=int, default=SEQUENCE_TIMEOUT_SECONDS)
    args = parser.parse_args()
    if args.worker_sequence:
        if not all([
            args.worker_backbone_plan,
            args.worker_output,
            args.worker_tool_contract,
        ]):
            parser.error("worker paths are required")
        run_worker(
            args.worker_sequence,
            Path(args.worker_backbone_plan).resolve(),
            Path(args.worker_output).resolve(),
            Path(args.worker_tool_contract).resolve(),
        )
        return
    if args.run_panel:
        if not all([
            args.chemistry_audit_jsonl,
            args.formal_v3_cache_jsonl,
            args.faspr_root,
            args.output_dir,
        ]):
            parser.error("panel input paths are required")
        summary = run_panel(args)
        print(json.dumps({
            "status": (
                "PASS"
                if summary["classification"] == "FASPR_FIXED_BACKBONE_PASS"
                else "FAIL"
            ),
            "classification": summary["classification"],
            "output_dir": str(Path(args.output_dir).resolve()),
        }, sort_keys=True))
        return
    parser.error("choose --worker-sequence or --run-panel")


if __name__ == "__main__":
    main()
