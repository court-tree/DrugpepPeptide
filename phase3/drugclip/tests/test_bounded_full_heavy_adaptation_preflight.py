from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from phase3.drugclip import finalize_bounded_full_heavy_adaptation as finalizer
from phase3.drugclip.finalize_bounded_full_heavy_adaptation import (
    build_final_adaptation_manifest,
    manifest_bytes,
)
from phase3.drugclip.full_heavy_adaptation_contract import (
    CACHE_MANIFEST_SCHEMA,
    CANONICAL_TOPOLOGY_CONTRACT,
    CONTRACT_SCHEMA,
    FREEZE_CONTRACT_VERSION,
    canonical_json_sha256,
    sha256_file,
)
from phase3.drugclip.preflight_bounded_full_heavy_adaptation import (
    module_state_sha256,
    optimizer_parameter_group_description,
)
from phase3.drugclip.tests.test_full_heavy_adaptation_contract import _model


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _with_sha(value: dict, field: str) -> dict:
    result = copy.deepcopy(value)
    result[field] = canonical_json_sha256(result)
    return result


def _fixture(root: Path) -> dict[str, Path]:
    train_ids = [f"train:{index:04d}" for index in range(4096)]
    valid_ids = [f"valid:{index:04d}" for index in range(512)]
    plan = _with_sha(
        {
            "schema_version": "phase3-v2-bounded-full-heavy-plan-v1",
            "plans": {
                "train": {
                    "pair_count": 4096,
                    "interface_pair_ids": train_ids,
                    "interface_pair_ids_sha256": canonical_json_sha256(train_ids),
                },
                "valid": {
                    "pair_count": 512,
                    "interface_pair_ids": valid_ids,
                    "interface_pair_ids_sha256": canonical_json_sha256(valid_ids),
                },
            },
        },
        "descriptor_canonical_sha256",
    )
    plan_path = root / "plan.json"
    _write_json(plan_path, plan)
    cache_contract = _with_sha(
        {
            "schema_version": "phase3-v2-bounded-full-heavy-cache-contract-v1",
            "generator_version": "generator-v1",
            "torsion_prior_manifest_file_sha256": "A" * 64,
            "torsion_prior_manifest_canonical_sha256": "B" * 64,
            "torsion_prior_jsonl_sha256": "C" * 64,
            "faspr_source_commit": "commit",
            "faspr_binary_sha256": "D" * 64,
            "faspr_rotamer_library_sha256": "E" * 64,
            "canonical_topology_contract": CANONICAL_TOPOLOGY_CONTRACT,
            "maximum_attempts_per_logical_conformer": 25,
            "nonlocal_heavy_atom_clash_threshold_angstrom": 0.75,
            "atom_cap_exclusive": 192,
            "generation_input_contract": {
                "allowed_fields": ["peptide_sequence"],
                "forbidden_fields": ["receptor"],
                "target_bound_generation_inputs_used": False,
            },
        },
        "contract_canonical_sha256",
    )
    contract_path = root / "cache_contract.json"
    _write_json(contract_path, cache_contract)
    cache = _with_sha(
        {
            "schema_version": CACHE_MANIFEST_SCHEMA,
            "status": "MATERIALIZED",
            "purpose": "bounded_train_valid_only",
            "plan_descriptor_file_sha256": sha256_file(plan_path),
            "plan_descriptor_canonical_sha256": plan[
                "descriptor_canonical_sha256"
            ],
            "cache_contract_canonical_sha256": cache_contract[
                "contract_canonical_sha256"
            ],
            "sequence_count": 2085,
            "conformer_count": 20850,
            "conformers_per_sequence": 10,
            "atom_cap_exclusive": 192,
        },
        "manifest_canonical_sha256",
    )
    cache_path = root / "cache_manifest.json"
    _write_json(cache_path, cache)
    checkpoint = root / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    random_manifest = {
        "dataset_version": "random_conformer_v3",
        "generator_id": "formal-v3",
        "cache_schema": "cache-v3",
    }
    random_path = root / "DATA_MANIFEST.json"
    _write_json(random_path, random_manifest)
    safe = _with_sha(
        {"schema_version": "safe373-plan"}, "plan_canonical_sha256"
    )
    safe_path = root / "safe373.json"
    _write_json(safe_path, safe)
    return {
        "plan": plan_path,
        "cache": cache_path,
        "checkpoint": checkpoint,
        "random": random_path,
        "safe": safe_path,
        "output": root / "output" / "final_adaptation_manifest.json",
    }


class FinalAdaptationManifestTests(unittest.TestCase):
    def _build(self, paths: dict[str, Path]) -> dict:
        with (
            patch.object(
                finalizer, "EXPECTED_PLAN_FILE_SHA256", sha256_file(paths["plan"])
            ),
            patch.object(
                finalizer,
                "EXPECTED_PLAN_CANONICAL_SHA256",
                json.loads(paths["plan"].read_text())[
                    "descriptor_canonical_sha256"
                ],
            ),
            patch.object(
                finalizer,
                "EXPECTED_CACHE_FILE_SHA256",
                sha256_file(paths["cache"]),
            ),
            patch.object(
                finalizer,
                "EXPECTED_CACHE_CANONICAL_SHA256",
                json.loads(paths["cache"].read_text())[
                    "manifest_canonical_sha256"
                ],
            ),
            patch.object(
                finalizer,
                "PHASE2_INITIALIZATION_SHA256",
                sha256_file(paths["checkpoint"]),
            ),
            patch.object(
                finalizer,
                "SAFE373_PLAN_CANONICAL_SHA256",
                json.loads(paths["safe"].read_text())["plan_canonical_sha256"],
            ),
        ):
            return build_final_adaptation_manifest(
                output_file=paths["output"],
                plan_descriptor_file=paths["plan"],
                cache_manifest_file=paths["cache"],
                phase2_checkpoint=paths["checkpoint"],
                random_conformer_v3_manifest=paths["random"],
                safe373_evaluation_plan=paths["safe"],
            )

    def test_manifest_is_byte_deterministic_and_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _fixture(Path(temporary))
            first = self._build(paths)
            second = self._build(paths)
            self.assertEqual(manifest_bytes(first), manifest_bytes(second))
            core = {
                key: value
                for key, value in first.items()
                if key != "manifest_canonical_sha256"
            }
            self.assertEqual(
                first["manifest_canonical_sha256"], canonical_json_sha256(core)
            )

    def test_manifest_binds_all_registered_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            value = self._build(_fixture(Path(temporary)))
            self.assertEqual(value["schema_version"], CONTRACT_SCHEMA)
            self.assertEqual(value["counts"]["train_interface_pairs"], 4096)
            self.assertEqual(value["counts"]["valid_interface_pairs"], 512)
            self.assertEqual(value["counts"]["cache_sequences"], 2085)
            self.assertEqual(value["counts"]["cache_conformers"], 20850)
            self.assertEqual(
                value["freeze_contract"]["version"], FREEZE_CONTRACT_VERSION
            )
            self.assertFalse(
                value["source_policy"]["special_chemistry_in_scope"]
            )
            self.assertFalse(value["execution_state"]["optimizer_created"])

    def test_manifest_contains_only_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            value = self._build(_fixture(Path(temporary)))
            text = json.dumps(value)
            self.assertNotIn(str(Path(temporary).resolve()), text)
            for section, key in (
                ("plan", "path"),
                ("cache", "path"),
                ("initialization", "checkpoint_path"),
                ("formal_random_conformer_v3", "manifest_path"),
                ("safe373_evaluation_plan", "path"),
            ):
                self.assertFalse(Path(value[section][key]).is_absolute())

    def test_tampered_plan_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _fixture(Path(temporary))
            plan = json.loads(paths["plan"].read_text())
            plan["plans"]["train"]["pair_count"] = 4095
            plan["descriptor_canonical_sha256"] = canonical_json_sha256(
                {
                    key: value
                    for key, value in plan.items()
                    if key != "descriptor_canonical_sha256"
                }
            )
            _write_json(paths["plan"], plan)
            with self.assertRaisesRegex(
                ValueError, "cache_plan_binding|pair_count"
            ):
                self._build(paths)


class ZeroStepHelperTests(unittest.TestCase):
    def test_optimizer_groups_are_descriptions_not_optimizers(self) -> None:
        model = _model()
        from phase3.drugclip.full_heavy_adaptation_contract import (
            configure_bounded_full_heavy_trainable,
        )

        contract = configure_bounded_full_heavy_trainable(model)
        groups = optimizer_parameter_group_description(model)
        self.assertEqual(
            sum(group["tensor_count"] for group in groups),
            len(contract["trainable_parameter_names"]),
        )
        self.assertEqual(
            sum(group["parameter_count"] for group in groups),
            contract["trainable_parameter_count"],
        )
        self.assertTrue(
            all("params" not in group and "lr" not in group for group in groups)
        )

    def test_module_state_hash_detects_change(self) -> None:
        module = torch.nn.Linear(3, 2)
        before = module_state_sha256(module)
        with torch.no_grad():
            module.weight[0, 0].add_(1.0)
        self.assertNotEqual(before, module_state_sha256(module))

    def test_preflight_source_has_no_optimizer_or_backward_call(self) -> None:
        source_path = (
            Path(__file__).resolve().parents[1]
            / "preflight_bounded_full_heavy_adaptation.py"
        )
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        attributes = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        self.assertNotIn("backward", attributes)
        self.assertNotIn("step", attributes)
        self.assertNotIn("Adam", attributes)
        self.assertNotIn("AdamW", attributes)


if __name__ == "__main__":
    unittest.main()
