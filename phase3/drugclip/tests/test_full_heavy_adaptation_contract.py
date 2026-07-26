from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from phase2.pepclip.data import AA_TO_ID
from phase2.pepclip.model import PepCLIPModel
from phase2.pepclip.model_3d import PepCLIP3DModel
from phase2.pepclip.train_concat_fusion import PepCLIPConcatFusionModel
from phase3.drugclip.full_heavy_adaptation_contract import (
    ALLOWED_GENERATION_INPUTS,
    ATOM_CAP_EXCLUSIVE,
    CACHE_MANIFEST_SCHEMA,
    CACHE_SCHEMA,
    CANONICAL_TOPOLOGY_CONTRACT,
    CONTRACT_SCHEMA,
    ELIGIBILITY_REGISTRY_SCHEMA,
    FREEZE_CONTRACT_VERSION,
    PHASE2_INITIALIZATION_SHA256,
    PLAN_SCHEMA,
    TORSION_PRIOR_JSONL_SHA256,
    TORSION_PRIOR_MANIFEST_SHA256,
    assert_parameter_change_contract,
    build_bounded_optimizer_groups,
    canonical_json_sha256,
    configure_bounded_full_heavy_trainable,
    parameter_state_sha256,
    sequence_sha256,
    sha256_file,
    validate_bounded_full_heavy_contract,
)
from phase3.drugclip.tests.test_bounded_full_heavy_plan import (
    _descriptor_fixture,
)
from phase3.drugclip.training_state import (
    load_training_checkpoint,
    make_grad_scaler,
    save_training_checkpoint,
)


def _model() -> PepCLIPConcatFusionModel:
    one_d = PepCLIPModel(
        vocab_size=max(AA_TO_ID.values()) + 1,
        encoder_type="mean_pool",
        embed_dim=8,
        hidden_dim=16,
        output_dim=8,
        dropout=0.0,
    )
    three_d = PepCLIP3DModel(
        num_elements=16,
        num_atom_names=64,
        num_residue_names=32,
        encoder_type="egnn",
        element_dim=8,
        hidden_dim=16,
        output_dim=8,
        dropout=0.0,
        num_layers=2,
        num_rbf=8,
        distance_cutoff=10.0,
        num_neighbors=3,
    )
    return PepCLIPConcatFusionModel(
        one_d,
        three_d,
        concat_dim=16,
        hidden_dim=16,
        output_dim=8,
        dropout=0.0,
        temperature=1.0 / 14.0,
    )


def _hash_module(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def _assert_state_equal(
    case: unittest.TestCase,
    before: dict[str, torch.Tensor],
    module: torch.nn.Module,
) -> None:
    after = module.state_dict()
    case.assertEqual(set(before), set(after))
    case.assertTrue(all(torch.equal(before[key], after[key]) for key in before))


def _identity() -> list[dict]:
    names = [
        ("N", "N"),
        ("CA", "C"),
        ("C", "C"),
        ("O", "O"),
        ("CB", "C"),
        ("OXT", "O"),
    ]
    return [
        {
            "atom_index": index,
            "atom_name": atom_name,
            "element": element,
            "residue_index": 1,
            "residue_name": "ALA",
        }
        for index, (atom_name, element) in enumerate(names)
    ]


def _payload(sequence: str = "A") -> dict:
    identity = _identity()
    conformers = []
    for conformer_index in range(10):
        offset = conformer_index * 0.01
        conformers.append(
            {
                "conformer_index": conformer_index,
                "attempt_index": 0,
                "faspr": {"exit_code": 0, "timed_out": False},
                "geometry_audit": {
                    "status": "PASS",
                    "topology_contract": CANONICAL_TOPOLOGY_CONTRACT,
                },
                "attempt_qc": {
                    "status": "PASS",
                    "structural_qc": {
                        "status": "PASS",
                        "target_bound_inputs_used": False,
                    },
                    "cpu_egnn_forward": {
                        "status": "PASS",
                        "embedding_finite": True,
                        "tensorization_unk_count": 0,
                    },
                },
                "coordinates": [
                    [0.0, offset, 0.0],
                    [1.45, offset, 0.0],
                    [2.90, offset, 0.0],
                    [3.40, 1.1 + offset, 0.0],
                    [1.45, -1.4 + offset, 0.0],
                    [3.40, -1.1 + offset, 0.0],
                ],
            }
        )
    return {
        "peptide_sequence": sequence,
        "atom_identity": identity,
        "atom_identity_sha256": canonical_json_sha256(identity),
        "conformers": conformers,
    }


def _generator_contract() -> dict:
    return {
        "torsion_prior_manifest_sha256": TORSION_PRIOR_MANIFEST_SHA256,
        "torsion_prior_jsonl_sha256": TORSION_PRIOR_JSONL_SHA256,
        "backbone_contract": "train-only-residue-context-trans-only-v1",
        "sidechain_packer": "FASPR-fixed-backbone",
        "canonical_topology_contract": CANONICAL_TOPOLOGY_CONTRACT,
        "max_attempts_per_slot": 25,
        "nonlocal_clash_threshold_angstrom": 0.75,
        "candidate_independent": True,
        "faspr_source_commit": "COMMIT",
        "faspr_binary_sha256": "BINARY",
        "faspr_rotamer_library_sha256": "LIBRARY",
        "generator_version": "fixture-v1",
    }


def _contract_fixture(root: Path) -> tuple[Path, Path, dict]:
    train_ids = [f"train:{index:04d}" for index in range(4096)]
    valid_ids = [f"valid:{index:04d}" for index in range(512)]
    payload_path = root / "cache_A.json"
    payload_path.write_text(json.dumps(_payload("A"), sort_keys=True), encoding="utf-8")
    valid_payload_path = root / "cache_G.json"
    valid_payload_path.write_text(
        json.dumps(_payload("G"), sort_keys=True), encoding="utf-8"
    )
    index_path = root / "cache_index.jsonl"
    index_rows = [
        {
            "peptide_sequence": "A",
            "chemistry_classification": "ordinary_linear_standard",
            "split_roles": ["train"],
            "cache_path": payload_path.name,
            "cache_file_sha256": sha256_file(payload_path),
        },
        {
            "peptide_sequence": "G",
            "chemistry_classification": "ordinary_linear_standard",
            "split_roles": ["valid"],
            "cache_path": valid_payload_path.name,
            "cache_file_sha256": sha256_file(valid_payload_path),
        },
    ]
    index_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in index_rows),
        encoding="utf-8",
    )
    registry = [
        {
            "schema_version": ELIGIBILITY_REGISTRY_SCHEMA,
            "peptide_sequence": "A",
            "split": "train",
            "interface_pair_ids": train_ids,
            "structure_instance_classifications": [
                "ordinary_linear_standard"
            ],
            "chemistry_classification": "ordinary_linear_standard",
            "theoretical_heavy_atom_count": 6,
            "torsion_prior_covered": True,
            "eligible": True,
        },
        {
            "schema_version": ELIGIBILITY_REGISTRY_SCHEMA,
            "peptide_sequence": "G",
            "split": "valid",
            "interface_pair_ids": valid_ids,
            "structure_instance_classifications": [
                "ordinary_linear_standard"
            ],
            "chemistry_classification": "ordinary_linear_standard",
            "theoretical_heavy_atom_count": 5,
            "torsion_prior_covered": True,
            "eligible": True,
        },
    ]
    registry_path = root / "registry.jsonl"
    registry_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in registry),
        encoding="utf-8",
    )
    safe_plan = {
        "plan_canonical_sha256": "SAFE_PLAN",
        "safe_query_interface_pair_ids": ["valid:safe"],
        "safe_peptide_candidate_ids": ["SAFESEQ"],
    }
    safe_plan_path = root / "safe373_plan.json"
    safe_plan_path.write_text(json.dumps(safe_plan), encoding="utf-8")
    core = {
        "schema_version": CONTRACT_SCHEMA,
        "initialization": {
            "role": "phase2_learned_concat_baseline",
            "checkpoint_sha256": PHASE2_INITIALIZATION_SHA256,
        },
        "source_policy": {
            "training_source_split": "formal_train_only",
            "chemistry_classification": "ordinary_linear_standard",
            "evaluation_cache_used_for_training": False,
            "target_bound_generation_inputs_used": False,
            "generation_seed_inputs": ALLOWED_GENERATION_INPUTS,
        },
        "generator": _generator_contract(),
        "eligibility_registry": {
            "schema_version": ELIGIBILITY_REGISTRY_SCHEMA,
            "path": registry_path.name,
            "file_sha256": sha256_file(registry_path),
            "canonical_sha256": canonical_json_sha256(registry),
        },
        "plans": {
            "schema_version": PLAN_SCHEMA,
            "target_train_pair_count": 4096,
            "target_valid_pair_count": 512,
            "train_interface_pair_ids": train_ids,
            "train_interface_pair_ids_sha256": sequence_sha256(train_ids),
            "valid_interface_pair_ids": valid_ids,
            "valid_interface_pair_ids_sha256": sequence_sha256(valid_ids),
            "train_unique_peptide_sequences": ["A"],
            "train_unique_peptide_sequences_sha256": sequence_sha256(["A"]),
            "valid_unique_peptide_sequences": ["G"],
            "valid_unique_peptide_sequences_sha256": sequence_sha256(["G"]),
        },
        "evaluation_exclusion": {
            "safe373_plan_path": safe_plan_path.name,
            "safe373_plan_file_sha256": sha256_file(safe_plan_path),
            "safe373_plan_canonical_sha256": "SAFE_PLAN",
        },
        "cache": {
            "status": "materialized",
            "schema_version": CACHE_SCHEMA,
            "purpose": "bounded_train_valid_only",
            "index_path": index_path.name,
            "index_sha256": sha256_file(index_path),
            "conformers_per_sequence": 10,
            "atom_cap_exclusive": ATOM_CAP_EXCLUSIVE,
            "required_peptide_sequences": ["A", "G"],
            "required_peptide_sequences_sha256": sequence_sha256(["A", "G"]),
        },
    }
    manifest = {**core, "manifest_canonical_sha256": canonical_json_sha256(core)}
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    checkpoint = root / "phase2.pt"
    checkpoint.write_bytes(b"fixture")
    inputs = {
        "train_interface_pair_ids": train_ids,
        "valid_interface_pair_ids": valid_ids,
        "train_sequence_by_pair": dict.fromkeys(train_ids, "A"),
        "valid_sequence_by_pair": dict.fromkeys(valid_ids, "G"),
        "train_relation_by_pair": {
            pair_id: f"train_relation:{index}"
            for index, pair_id in enumerate(train_ids)
        },
        "valid_relation_by_pair": {
            pair_id: f"valid_relation:{index}"
            for index, pair_id in enumerate(valid_ids)
        },
    }
    return manifest_path, checkpoint, inputs


def _runtime_contract_fixture(
    root: Path,
) -> tuple[Path, Path, Path, Path, dict, dict]:
    descriptor_path, inputs = _descriptor_fixture(root)
    payload_path = root / "cache_A.json"
    payload_path.write_text(
        json.dumps(_payload("A"), sort_keys=True), encoding="utf-8"
    )
    valid_payload_path = root / "cache_G.json"
    valid_payload_path.write_text(
        json.dumps(_payload("G"), sort_keys=True), encoding="utf-8"
    )
    index_path = root / "cache_index.jsonl"
    rows = [
        {
            "peptide_sequence": "A",
            "chemistry_classification": "ordinary_linear_standard",
            "split_roles": ["train"],
            "cache_path": payload_path.name,
            "cache_file_sha256": sha256_file(payload_path),
        },
        {
            "peptide_sequence": "G",
            "chemistry_classification": "ordinary_linear_standard",
            "split_roles": ["valid"],
            "cache_path": valid_payload_path.name,
            "cache_file_sha256": sha256_file(valid_payload_path),
        },
    ]
    index_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    descriptor = json.loads(descriptor_path.read_text())
    required_sequences = ["A", "G"]
    cache_core = {
        "schema_version": CACHE_MANIFEST_SCHEMA,
        "status": "MATERIALIZED",
        "purpose": "bounded_train_valid_only",
        "plan_descriptor_file_sha256": sha256_file(descriptor_path),
        "plan_descriptor_canonical_sha256": descriptor[
            "descriptor_canonical_sha256"
        ],
        "conformers_per_sequence": 10,
        "atom_cap_exclusive": ATOM_CAP_EXCLUSIVE,
        "required_peptide_sequences": required_sequences,
        "required_peptide_sequences_sha256": sequence_sha256(
            required_sequences
        ),
        "index_path": index_path.name,
        "index_sha256": sha256_file(index_path),
    }
    cache_manifest = {
        **cache_core,
        "manifest_canonical_sha256": canonical_json_sha256(cache_core),
    }
    cache_manifest_path = root / "cache_manifest.json"
    cache_manifest_path.write_text(
        json.dumps(cache_manifest, sort_keys=True), encoding="utf-8"
    )
    freeze_contract = configure_bounded_full_heavy_trainable(_model())
    adaptation_core = {
        "schema_version": CONTRACT_SCHEMA,
        "phase2_checkpoint_sha256": PHASE2_INITIALIZATION_SHA256,
        "plan_descriptor_file_sha256": sha256_file(descriptor_path),
        "plan_descriptor_canonical_sha256": descriptor[
            "descriptor_canonical_sha256"
        ],
        "cache_manifest_file_sha256": sha256_file(cache_manifest_path),
        "cache_manifest_canonical_sha256": cache_manifest[
            "manifest_canonical_sha256"
        ],
        "freeze_contract_version": FREEZE_CONTRACT_VERSION,
        "trainable_parameter_names_sha256": freeze_contract[
            "trainable_parameter_names_sha256"
        ],
    }
    adaptation = {
        **adaptation_core,
        "manifest_canonical_sha256": canonical_json_sha256(adaptation_core),
    }
    adaptation_path = root / "adaptation_manifest.json"
    adaptation_path.write_text(
        json.dumps(adaptation, sort_keys=True), encoding="utf-8"
    )
    checkpoint = root / "phase2.pt"
    checkpoint.write_bytes(b"fixture")
    return (
        adaptation_path,
        descriptor_path,
        cache_manifest_path,
        checkpoint,
        inputs,
        freeze_contract,
    )


class FullHeavyAdaptationContractTests(unittest.TestCase):
    def test_trainable_parameter_set_is_exact(self):
        model = _model()
        contract = configure_bounded_full_heavy_trainable(model)
        names = set(contract["trainable_parameter_names"])
        self.assertTrue(names)
        self.assertTrue(all(
            name.startswith((
                "model_3d.peptide_encoder.layers.1.edge_mlp.",
                "model_3d.peptide_encoder.layers.1.node_mlp.",
                "model_3d.peptide_encoder.layers.1.norm.",
                "model_3d.peptide_encoder.final_norm.",
                "model_3d.peptide_encoder.project.",
                "peptide_fusion.",
            ))
            for name in names
        ))
        self.assertFalse(any(parameter.requires_grad for parameter in model.model_1d.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.model_3d.receptor_encoder.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.receptor_fusion.parameters()))
        last = model.model_3d.peptide_encoder.layers[-1]
        self.assertTrue(all(parameter.requires_grad for parameter in last.edge_mlp.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in last.node_mlp.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in last.norm.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in last.coord_mlp.parameters()))
        self.assertFalse(any(".coord_mlp." in name for name in names))

    def test_optimizer_contains_exact_allowed_parameters(self):
        model = _model()
        configure_bounded_full_heavy_trainable(model)
        groups = build_bounded_optimizer_groups(model, fusion_lr=1e-6, tower_lr=2e-7)
        actual = {id(parameter) for group in groups for parameter in group["params"]}
        expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
        self.assertEqual(actual, expected)
        self.assertEqual([group["group_name"] for group in groups], [
            "peptide_fusion", "peptide_3d_last1"
        ])

    def test_one_step_preserves_all_frozen_directional_modules(self):
        torch.manual_seed(7)
        model = _model()
        contract = configure_bounded_full_heavy_trainable(model)
        frozen_before = {
            "receptor_1d": _hash_module(model.model_1d.receptor_encoder),
            "receptor_3d": _hash_module(model.model_3d.receptor_encoder),
            "receptor_fusion": _hash_module(model.receptor_fusion),
            "peptide_1d": _hash_module(model.model_1d.peptide_encoder),
        }
        before = parameter_state_sha256(model)
        optimizer = torch.optim.SGD(
            build_bounded_optimizer_groups(model, fusion_lr=0.05, tower_lr=0.05)
        )
        coords = torch.tensor([[
            [0.0, 0.0, 0.0], [1.4, 0.0, 0.0],
            [2.8, 0.2, 0.0], [3.8, 0.3, 0.1],
        ]])
        values = torch.tensor([[1, 2, 2, 3]])
        mask = torch.ones(1, 4, dtype=torch.bool)
        receptor_tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            receptor_1d_before = model.model_1d.encode_receptor(
                receptor_tokens
            ).clone()
            receptor_3d_before = model.model_3d.encode_receptor(
                coords, values, mask, values, values
            ).clone()
            receptor_fused_before = model.receptor_fusion(
                torch.cat([receptor_1d_before, receptor_3d_before], dim=-1)
            ).clone()
            peptide_1d_before = model.model_1d.encode_peptide(
                receptor_tokens
            ).clone()
        peptide_3d = model.model_3d.encode_peptide(
            coords, values, mask, values, values
        )
        peptide_1d = model.model_1d.encode_peptide(
            torch.tensor([[1, 2, 3, 4]])
        )
        loss = model.peptide_fusion(torch.cat([peptide_1d, peptide_3d], dim=-1)).square().sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        audit = assert_parameter_change_contract(
            before, parameter_state_sha256(model), contract
        )
        self.assertTrue(audit["changed_allowed_parameters"])
        _assert_state_equal(self, frozen_before["receptor_1d"], model.model_1d.receptor_encoder)
        _assert_state_equal(self, frozen_before["receptor_3d"], model.model_3d.receptor_encoder)
        _assert_state_equal(self, frozen_before["receptor_fusion"], model.receptor_fusion)
        _assert_state_equal(self, frozen_before["peptide_1d"], model.model_1d.peptide_encoder)
        with torch.no_grad():
            receptor_1d_after = model.model_1d.encode_receptor(receptor_tokens)
            receptor_3d_after = model.model_3d.encode_receptor(
                coords, values, mask, values, values
            )
            receptor_fused_after = model.receptor_fusion(
                torch.cat([receptor_1d_after, receptor_3d_after], dim=-1)
            )
            peptide_1d_after = model.model_1d.encode_peptide(receptor_tokens)
        self.assertTrue(torch.equal(receptor_1d_before, receptor_1d_after))
        self.assertTrue(torch.equal(receptor_3d_before, receptor_3d_after))
        self.assertTrue(torch.equal(receptor_fused_before, receptor_fused_after))
        self.assertTrue(torch.equal(peptide_1d_before, peptide_1d_after))

    def test_forbidden_parameter_change_is_detected(self):
        model = _model()
        contract = configure_bounded_full_heavy_trainable(model)
        before = parameter_state_sha256(model)
        with torch.no_grad():
            next(model.receptor_fusion.parameters()).add_(1.0)
        with self.assertRaisesRegex(AssertionError, "forbidden_parameter_changed"):
            assert_parameter_change_contract(
                before, parameter_state_sha256(model), contract
            )

    def test_manifest_validates_train_valid_cache_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            (
                manifest,
                descriptor,
                cache_manifest,
                checkpoint,
                inputs,
                freeze_contract,
            ) = _runtime_contract_fixture(Path(temporary))
            original = sha256_file
            def fake_hash(path):
                return (
                    PHASE2_INITIALIZATION_SHA256
                    if Path(path).resolve() == checkpoint.resolve()
                    else original(path)
                )
            with patch(
                "phase3.drugclip.full_heavy_adaptation_contract.sha256_file",
                side_effect=fake_hash,
            ):
                result = validate_bounded_full_heavy_contract(
                    manifest,
                    plan_descriptor_file=descriptor,
                    cache_manifest_file=cache_manifest,
                    phase2_checkpoint=checkpoint,
                    freeze_contract=freeze_contract,
                    **inputs,
                )
            self.assertEqual(result["cache_sequence_count"], 2)
            self.assertEqual(len(result["train_interface_pair_ids"]), 4096)
            self.assertEqual(len(result["valid_interface_pair_ids"]), 512)

    def test_old_26_tensor_freeze_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            (
                manifest,
                descriptor,
                cache_manifest,
                checkpoint,
                inputs,
                freeze_contract,
            ) = _runtime_contract_fixture(Path(temporary))
            freeze_contract["contract_version"] = (
                "phase3-v2-peptide-3d-last1-plus-peptide-fusion-v1"
            )
            original = sha256_file

            def fake_hash(path):
                return (
                    PHASE2_INITIALIZATION_SHA256
                    if Path(path).resolve() == checkpoint.resolve()
                    else original(path)
                )

            with patch(
                "phase3.drugclip.full_heavy_adaptation_contract.sha256_file",
                side_effect=fake_hash,
            ):
                with self.assertRaisesRegex(
                    ValueError, "full_heavy_adaptation_binding_mismatch"
                ):
                    validate_bounded_full_heavy_contract(
                        manifest,
                        plan_descriptor_file=descriptor,
                        cache_manifest_file=cache_manifest,
                        phase2_checkpoint=checkpoint,
                        freeze_contract=freeze_contract,
                        **inputs,
                    )

    def test_old_adaptation_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            (
                manifest,
                descriptor,
                cache_manifest,
                checkpoint,
                inputs,
                freeze_contract,
            ) = _runtime_contract_fixture(Path(temporary))
            value = json.loads(manifest.read_text())
            value["freeze_contract_version"] = (
                "phase3-v2-peptide-3d-last1-plus-peptide-fusion-v1"
            )
            core = {
                key: item
                for key, item in value.items()
                if key != "manifest_canonical_sha256"
            }
            value["manifest_canonical_sha256"] = canonical_json_sha256(core)
            manifest.write_text(json.dumps(value), encoding="utf-8")
            original = sha256_file

            def fake_hash(path):
                return (
                    PHASE2_INITIALIZATION_SHA256
                    if Path(path).resolve() == checkpoint.resolve()
                    else original(path)
                )

            with patch(
                "phase3.drugclip.full_heavy_adaptation_contract.sha256_file",
                side_effect=fake_hash,
            ):
                with self.assertRaisesRegex(
                    ValueError, "full_heavy_adaptation_binding_mismatch"
                ):
                    validate_bounded_full_heavy_contract(
                        manifest,
                        plan_descriptor_file=descriptor,
                        cache_manifest_file=cache_manifest,
                        phase2_checkpoint=checkpoint,
                        freeze_contract=freeze_contract,
                        **inputs,
                    )

    def test_manifest_rejects_evaluation_cache_as_training_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                manifest,
                descriptor,
                cache_manifest,
                checkpoint,
                inputs,
                freeze_contract,
            ) = _runtime_contract_fixture(root)
            value = json.loads(cache_manifest.read_text())
            value["index_path"] = "safe373_cache_index.jsonl"
            core = {
                key: item for key, item in value.items()
                if key != "manifest_canonical_sha256"
            }
            value["manifest_canonical_sha256"] = canonical_json_sha256(core)
            cache_manifest.write_text(
                json.dumps(value), encoding="utf-8"
            )
            adaptation = json.loads(manifest.read_text())
            adaptation["cache_manifest_file_sha256"] = sha256_file(
                cache_manifest
            )
            adaptation["cache_manifest_canonical_sha256"] = value[
                "manifest_canonical_sha256"
            ]
            adaptation_core = {
                key: item for key, item in adaptation.items()
                if key != "manifest_canonical_sha256"
            }
            adaptation["manifest_canonical_sha256"] = canonical_json_sha256(
                adaptation_core
            )
            manifest.write_text(
                json.dumps(adaptation), encoding="utf-8"
            )
            original = sha256_file
            def fake_hash(path):
                return (
                    PHASE2_INITIALIZATION_SHA256
                    if Path(path).resolve() == checkpoint.resolve()
                    else original(path)
                )
            with patch(
                "phase3.drugclip.full_heavy_adaptation_contract.sha256_file",
                side_effect=fake_hash,
            ):
                with self.assertRaisesRegex(
                    ValueError, "evaluation_cache_path_forbidden"
                ):
                    validate_bounded_full_heavy_contract(
                        manifest,
                        plan_descriptor_file=descriptor,
                        cache_manifest_file=cache_manifest,
                        phase2_checkpoint=checkpoint,
                        freeze_contract=freeze_contract,
                        **inputs,
                    )

    def test_manifest_rejects_plan_or_split_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            (
                manifest,
                descriptor,
                cache_manifest,
                checkpoint,
                inputs,
                freeze_contract,
            ) = _runtime_contract_fixture(Path(temporary))
            inputs["train_interface_pair_ids"] = [
                "other",
                *inputs["train_interface_pair_ids"][1:],
            ]
            inputs["train_sequence_by_pair"] = {
                pair_id: "A" for pair_id in inputs["train_interface_pair_ids"]
            }
            inputs["train_relation_by_pair"] = {
                pair_id: f"relation:{index}"
                for index, pair_id in enumerate(inputs["train_interface_pair_ids"])
            }
            original = sha256_file
            def fake_hash(path):
                return (
                    PHASE2_INITIALIZATION_SHA256
                    if Path(path).resolve() == checkpoint.resolve()
                    else original(path)
                )
            with patch(
                "phase3.drugclip.full_heavy_adaptation_contract.sha256_file",
                side_effect=fake_hash,
            ):
                with self.assertRaisesRegex(ValueError, "registry_pair_coverage"):
                    validate_bounded_full_heavy_contract(
                        manifest,
                        plan_descriptor_file=descriptor,
                        cache_manifest_file=cache_manifest,
                        phase2_checkpoint=checkpoint,
                        freeze_contract=freeze_contract,
                        **inputs,
                    )

    def test_adaptation_manifest_rejects_changed_plan_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            (
                manifest,
                descriptor,
                cache_manifest,
                checkpoint,
                inputs,
                freeze_contract,
            ) = _runtime_contract_fixture(Path(temporary))
            value = json.loads(manifest.read_text())
            value["plan_descriptor_file_sha256"] = "F" * 64
            core = {
                key: item for key, item in value.items()
                if key != "manifest_canonical_sha256"
            }
            value["manifest_canonical_sha256"] = canonical_json_sha256(core)
            manifest.write_text(json.dumps(value), encoding="utf-8")
            original = sha256_file

            def fake_hash(path):
                return (
                    PHASE2_INITIALIZATION_SHA256
                    if Path(path).resolve() == checkpoint.resolve()
                    else original(path)
                )

            with patch(
                "phase3.drugclip.full_heavy_adaptation_contract.sha256_file",
                side_effect=fake_hash,
            ):
                with self.assertRaisesRegex(
                    ValueError, "adaptation_binding_mismatch"
                ):
                    validate_bounded_full_heavy_contract(
                        manifest,
                        plan_descriptor_file=descriptor,
                        cache_manifest_file=cache_manifest,
                        phase2_checkpoint=checkpoint,
                        freeze_contract=freeze_contract,
                        **inputs,
                    )

    def test_manifest_rejects_atom_cap_and_unk(self):
        payload = _payload()
        payload["atom_identity"][0]["element"] = "UNK"
        payload["atom_identity_sha256"] = canonical_json_sha256(payload["atom_identity"])
        from phase3.drugclip.full_heavy_adaptation_contract import _validate_full_heavy_payload
        with self.assertRaisesRegex(ValueError, "element_unk"):
            _validate_full_heavy_payload(payload, "A")

    def test_checkpoint_binds_full_heavy_and_freeze_contracts(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            scaler = make_grad_scaler(torch.device("cpu"), False)
            config = {
                "phase2_checkpoint": "phase2.pt",
                "relation_schema": "schema",
                "random_pairs_jsonl": "pairs.jsonl",
                "valid_random_pairs_jsonl": "valid.jsonl",
                "random_conformer_cache_jsonl": "v3-cache.jsonl",
                "biological_pairs_jsonl": "bio.jsonl",
                "biological_pairs_sha256": "A" * 64,
                "pair_splits_jsonl": "splits.jsonl",
                "freeze_configuration": {"contract_version": "freeze-v1"},
                "full_heavy_data_contract": {"manifest_canonical_sha256": "B" * 64},
                "global_seed": 1,
                "sampling_unit": "interface_pair",
                "train_interface_pair_ids": ["train:1"],
                "valid_interface_pair_ids": ["valid:1"],
                "train_interface_pair_ids_sha256": "C" * 64,
                "valid_interface_pair_ids_sha256": "D" * 64,
                "fixed_validation_plan_sha256": "E" * 64,
                "total_train_steps": 1,
                "warmup_fraction": 0.0,
                "warmup_steps": 0,
                "scheduler_kind": "linear_warmup_constant",
            }
            path = Path(temporary) / "checkpoint.pt"
            save_training_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=0,
                global_step=0,
                global_seed=1,
                best_validation_loss=1.0,
                run_config=config,
                sampler_state={},
                history=[],
            )
            raw = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(
                raw["data_contract"]["full_heavy_data_contract"],
                config["full_heavy_data_contract"],
            )
            mismatch = copy.deepcopy(config)
            mismatch["full_heavy_data_contract"]["manifest_canonical_sha256"] = "F" * 64
            with self.assertRaisesRegex(ValueError, "full_heavy_data_contract"):
                load_training_checkpoint(
                    path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    expected_run_config=mismatch,
                    device=torch.device("cpu"),
                )


if __name__ == "__main__":
    unittest.main()
