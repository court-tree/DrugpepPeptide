"""Read-only validator for a selected PepCLIP Phase-3 model release."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PureWindowsPath
from typing import Any

import torch


RELEASE_SCHEMA = "pepclip-phase3-model-release-v1"
CHECKPOINT_SCHEMA = "pepclip-phase3-drugclip-training-v1"
DEFAULT_DESCRIPTOR = Path("phase3/drugclip/releases/phase3_v1_selected_model.json")
SHA256_PATTERN = re.compile(r"^[0-9A-F]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _resolve_file(repo_root: Path, relative: object, label: str, errors: list[str]) -> Path | None:
    value = str(relative or "")
    normalized = value.replace("\\", "/")
    parts = Path(normalized).parts
    if not value or Path(normalized).is_absolute() or PureWindowsPath(value).is_absolute() or ".." in parts:
        errors.append(f"{label}:path_must_be_repo_relative")
        return None
    path = (repo_root / normalized).resolve()
    try:
        path.relative_to(repo_root)
    except ValueError:
        errors.append(f"{label}:path_outside_repo")
        return None
    if not path.is_file():
        errors.append(f"{label}:missing:{normalized}")
        return None
    return path


def _check_sha_record(
    repo_root: Path,
    record: dict[str, Any],
    path_key: str,
    sha_key: str,
    label: str,
    errors: list[str],
    bytes_key: str | None = None,
) -> Path | None:
    path = _resolve_file(repo_root, record.get(path_key), label, errors)
    expected_sha = str(record.get(sha_key, ""))
    if not SHA256_PATTERN.fullmatch(expected_sha):
        errors.append(f"{label}:invalid_sha256")
    if path is None:
        return None
    if bytes_key is not None and path.stat().st_size != record.get(bytes_key):
        errors.append(f"{label}:bytes")
    if SHA256_PATTERN.fullmatch(expected_sha) and sha256_file(path) != expected_sha:
        errors.append(f"{label}:sha256")
    return path


def _path_ends_with(value: object, relative: object) -> bool:
    actual = str(value or "").replace("\\", "/").lower()
    expected = str(relative or "").replace("\\", "/").lower()
    return bool(expected) and actual.endswith(expected)


def _load_json(path: Path | None, label: str, errors: list[str]) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{label}:invalid_json:{type(exc).__name__}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{label}:json_root_not_object")
        return None
    return value


def validate_release(descriptor_path: str | Path, repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    descriptor_file = Path(descriptor_path)
    if not descriptor_file.is_absolute():
        descriptor_file = root / descriptor_file
    descriptor = json.loads(descriptor_file.read_text(encoding="utf-8"))
    errors: list[str] = []

    if descriptor.get("schema_version") != RELEASE_SCHEMA:
        errors.append("descriptor:schema_version")
    if descriptor.get("release_status") != "selected":
        errors.append("descriptor:release_status")

    checkpoint_record = descriptor.get("checkpoint", {})
    if not isinstance(checkpoint_record, dict):
        checkpoint_record = {}
        errors.append("checkpoint:record")
    checkpoint_path = _check_sha_record(
        root, checkpoint_record, "relative_path", "sha256", "checkpoint", errors, "bytes"
    )

    data_record = descriptor.get("data_contract", {})
    if not isinstance(data_record, dict):
        data_record = {}
        errors.append("data_contract:record")
    manifest_path = _check_sha_record(
        root, data_record, "manifest_relative_path", "manifest_sha256", "manifest", errors
    )
    manifest = _load_json(manifest_path, "manifest", errors)
    manifest_contract = data_record.get("manifest_contract", {})
    if not isinstance(manifest_contract, dict):
        manifest_contract = {}
        errors.append("manifest:contract")
    if manifest is not None:
        for key, expected in manifest_contract.items():
            if manifest.get(key) != expected:
                errors.append(f"manifest:{key}")

    initialization = descriptor.get("initialization", {})
    if not isinstance(initialization, dict):
        initialization = {}
        errors.append("initialization:record")
    _check_sha_record(
        root,
        initialization,
        "checkpoint_relative_path",
        "checkpoint_sha256",
        "initialization_checkpoint",
        errors,
        "checkpoint_bytes",
    )

    checkpoint: dict[str, Any] | None = None
    if checkpoint_path is not None:
        try:
            loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(loaded, dict):
                checkpoint = loaded
            else:
                errors.append("checkpoint:root_not_mapping")
        except Exception as exc:  # the error is reported; the validator never mutates the file
            errors.append(f"checkpoint:load:{type(exc).__name__}")

    if checkpoint is not None:
        expected_scalars = {
            "schema_version": checkpoint_record.get("schema_version"),
            "epoch": checkpoint_record.get("epoch"),
            "global_step": checkpoint_record.get("global_step"),
            "global_seed": checkpoint_record.get("global_seed"),
            "best_validation_loss": checkpoint_record.get("best_validation_loss"),
        }
        if expected_scalars["schema_version"] != CHECKPOINT_SCHEMA:
            errors.append("descriptor:checkpoint_schema_version")
        for key, expected in expected_scalars.items():
            if checkpoint.get(key) != expected:
                errors.append(f"checkpoint:{key}")

        checkpoint_data = checkpoint.get("data_contract")
        if not isinstance(checkpoint_data, dict):
            checkpoint_data = {}
            errors.append("checkpoint:data_contract")
        expected_data = {
            "data_version": data_record.get("data_version"),
            "dataset_version": data_record.get("dataset_version"),
            "data_manifest_sha256": data_record.get("manifest_sha256"),
            **(data_record.get("checkpoint_contract", {}) if isinstance(data_record.get("checkpoint_contract"), dict) else {}),
        }
        for key, expected in expected_data.items():
            if checkpoint_data.get(key) != expected:
                errors.append(f"checkpoint:data_contract:{key}")
        if data_record.get("data_version") != "v3":
            errors.append("descriptor:data_version")
        if data_record.get("dataset_version") != "random_conformer_v3":
            errors.append("descriptor:dataset_version")
        if not _path_ends_with(checkpoint_data.get("data_manifest_path"), data_record.get("manifest_relative_path")):
            errors.append("checkpoint:data_contract:data_manifest_path")

        run_config = checkpoint.get("run_config")
        if not isinstance(run_config, dict):
            run_config = {}
            errors.append("checkpoint:run_config")
        for key in ("data_version", "dataset_version", "data_manifest_sha256"):
            if run_config.get(key) != expected_data.get(key):
                errors.append(f"checkpoint:run_config:{key}")
        if not _path_ends_with(run_config.get("phase2_checkpoint"), initialization.get("checkpoint_relative_path")):
            errors.append("checkpoint:run_config:phase2_checkpoint")

        state_contract = checkpoint_record.get("model_state", {})
        if not isinstance(state_contract, dict):
            state_contract = {}
            errors.append("checkpoint:model_state_contract")
        state_key = state_contract.get("state_dict_key")
        model_state = checkpoint.get(state_key) if isinstance(state_key, str) else None
        if not isinstance(model_state, dict) or not model_state:
            errors.append("checkpoint:model_state")
        else:
            values = list(model_state.values())
            if not all(isinstance(value, torch.Tensor) for value in values):
                errors.append("checkpoint:model_state:non_tensor")
            else:
                if len(values) != state_contract.get("tensor_count"):
                    errors.append("checkpoint:model_state:tensor_count")
                if sum(value.numel() for value in values) != state_contract.get("parameter_and_buffer_numel"):
                    errors.append("checkpoint:model_state:numel")
                finite = all(
                    bool(torch.isfinite(value).all().item())
                    for value in values
                    if value.is_floating_point() or value.is_complex()
                )
                if state_contract.get("all_floating_and_complex_tensors_finite") is not True:
                    errors.append("descriptor:model_state:finite_requirement")
                if not finite:
                    errors.append("checkpoint:model_state:finite")

    evaluation = descriptor.get("evaluation", {})
    if not isinstance(evaluation, dict):
        evaluation = {}
        errors.append("evaluation:record")
    fixed_plan = evaluation.get("fixed_plan", {})
    if isinstance(fixed_plan, dict):
        _check_sha_record(root, fixed_plan, "relative_path", "file_sha256", "fixed_plan", errors)
    else:
        errors.append("evaluation:fixed_plan")
        fixed_plan = {}
    reports = evaluation.get("reports", {})
    report_objects: dict[str, dict[str, Any]] = {}
    if not isinstance(reports, dict):
        errors.append("evaluation:reports")
        reports = {}
    for name, record in reports.items():
        if not isinstance(record, dict):
            errors.append(f"evaluation:{name}:record")
            continue
        path = _check_sha_record(root, record, "relative_path", "sha256", f"evaluation:{name}", errors)
        loaded = _load_json(path, f"evaluation:{name}", errors)
        if loaded is not None:
            report_objects[name] = loaded
        if "schema_version" in record and loaded is not None and loaded.get("schema_version") != record["schema_version"]:
            errors.append(f"evaluation:{name}:schema_version")

    model_label = evaluation.get("model_label")
    single = report_objects.get("single_conformer")
    if single is not None:
        if single.get("requested_model_label") != model_label:
            errors.append("evaluation:single_conformer:model_label")
        if single.get("checkpoints", {}).get(model_label, {}).get("sha256") != checkpoint_record.get("sha256"):
            errors.append("evaluation:single_conformer:checkpoint_sha256")
        if single.get("validation_interface_pair_ids_sha256") != evaluation.get("validation_interface_pair_ids_sha256"):
            errors.append("evaluation:single_conformer:query_ids")
        single_plan = single.get("fixed_validation_plan", {})
        if single_plan.get("file_sha256") != fixed_plan.get("file_sha256"):
            errors.append("evaluation:single_conformer:plan_file_sha256")
        if single_plan.get("canonical_sha256") != fixed_plan.get("canonical_sha256"):
            errors.append("evaluation:single_conformer:plan_canonical_sha256")

    multi = report_objects.get("multi_conformer_config")
    if multi is not None:
        if multi.get("requested_model_label") != model_label:
            errors.append("evaluation:multi_conformer:model_label")
        if multi.get("checkpoints", {}).get(model_label, {}).get("sha256") != checkpoint_record.get("sha256"):
            errors.append("evaluation:multi_conformer:checkpoint_sha256")
        if multi.get("conformer_indices") != evaluation.get("conformer_indices"):
            errors.append("evaluation:multi_conformer:indices")
        if multi.get("conformer_zero_regression") != "passed":
            errors.append("evaluation:multi_conformer:conformer_zero_regression")
        if multi.get("fixed_validation_plan_sha256") != fixed_plan.get("canonical_sha256"):
            errors.append("evaluation:multi_conformer:plan_canonical_sha256")

    for key, value in descriptor.get("code_baseline", {}).items():
        if not re.fullmatch(r"[0-9a-f]{40}", str(value)):
            errors.append(f"code_baseline:{key}")

    return {
        "validator_id": "pepclip-phase3-model-release-validator-v1",
        "descriptor": str(descriptor_file.resolve()),
        "release_id": descriptor.get("release_id"),
        "model_version": descriptor.get("model_version"),
        "passed": not errors,
        "checkpoint_sha256": checkpoint_record.get("sha256"),
        "checkpoint_global_step": checkpoint_record.get("global_step"),
        "model_tensor_count": checkpoint_record.get("model_state", {}).get("tensor_count"),
        "evaluation_reports_checked": len(report_objects),
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", dest="descriptor_path", default=str(DEFAULT_DESCRIPTOR))
    parser.add_argument("--repo-root", dest="repo_root", default=str(Path(__file__).resolve().parents[2]))
    return parser.parse_args()


if __name__ == "__main__":
    result = validate_release(**vars(parse_args()))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
