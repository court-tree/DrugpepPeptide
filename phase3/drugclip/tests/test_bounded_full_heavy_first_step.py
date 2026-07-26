from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import torch

from phase3.drugclip.audit_bounded_full_heavy_first_step import (
    ExactlyOneStepAdamW,
    analyze_parameter_changes,
    named_buffer_hashes,
    named_parameter_hashes,
    nested_state_sha256,
    tensor_sha256,
    validate_optimizer_state,
    validate_trainable_gradients,
)


class BoundedFullHeavyFirstStepTests(unittest.TestCase):
    def test_optimizer_hard_stops_before_second_step(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = ExactlyOneStepAdamW([parameter], lr=0.1)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        self.assertEqual(optimizer.successful_step_calls, 1)
        parameter.grad = torch.ones_like(parameter)
        with self.assertRaisesRegex(RuntimeError, "second_optimizer_step_forbidden"):
            optimizer.step()
        self.assertEqual(optimizer.successful_step_calls, 1)

    def test_optimizer_state_is_exactly_step_one(self) -> None:
        model = torch.nn.Linear(2, 1)
        optimizer = ExactlyOneStepAdamW(model.parameters(), lr=0.01)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        audit = validate_optimizer_state(
            optimizer, model, [name for name, _ in model.named_parameters()]
        )
        self.assertEqual(audit["state_tensor_count"], 2)
        self.assertTrue(audit["all_steps_equal_one"])

    def test_optimizer_state_must_cover_every_allowed_parameter(self) -> None:
        model = torch.nn.Linear(2, 1)
        optimizer = ExactlyOneStepAdamW(model.parameters(), lr=0.01)
        with self.assertRaisesRegex(
            AssertionError, "optimizer_state_parameter_scope_mismatch"
        ):
            validate_optimizer_state(
                optimizer, model, [name for name, _ in model.named_parameters()]
            )

    def test_any_trainable_gradient_none_is_rejected(self) -> None:
        model = torch.nn.Linear(2, 1)
        model.weight.grad = torch.ones_like(model.weight)
        with self.assertRaisesRegex(
            AssertionError, "trainable_parameter_missing_gradient.*bias"
        ):
            validate_trainable_gradients(model)

    def test_all_trainable_gradients_must_be_finite(self) -> None:
        model = torch.nn.Linear(2, 1)
        model.weight.grad = torch.ones_like(model.weight)
        model.bias.grad = torch.ones_like(model.bias)
        audit = validate_trainable_gradients(model)
        self.assertEqual(set(audit), {"weight", "bias"})

    def test_forbidden_parameter_change_is_rejected(self) -> None:
        before = {"allowed": "A", "frozen": "B"}
        after = {"allowed": "C", "frozen": "D"}
        with self.assertRaisesRegex(AssertionError, "forbidden_parameter_changed"):
            analyze_parameter_changes(before, after, ["allowed"])

    def test_allowed_parameter_change_is_reported(self) -> None:
        result = analyze_parameter_changes(
            {"allowed": "A", "frozen": "B"},
            {"allowed": "C", "frozen": "B"},
            ["allowed"],
        )
        self.assertEqual(result["changed_allowed"], ["allowed"])
        self.assertEqual(result["changed_forbidden"], [])

    def test_named_parameter_and_buffer_hashes_are_sensitive(self) -> None:
        module = torch.nn.BatchNorm1d(2)
        parameter_before = named_parameter_hashes(module)
        buffer_before = named_buffer_hashes(module)
        with torch.no_grad():
            module.weight.add_(1.0)
            module.running_mean.add_(1.0)
        self.assertNotEqual(parameter_before, named_parameter_hashes(module))
        self.assertNotEqual(buffer_before, named_buffer_hashes(module))

    def test_tensor_and_nested_hashes_are_deterministic(self) -> None:
        tensor = torch.arange(4, dtype=torch.float32)
        self.assertEqual(tensor_sha256(tensor), tensor_sha256(tensor.clone()))
        state = {"tensor": tensor, "value": [1, "x"]}
        self.assertEqual(nested_state_sha256(state), nested_state_sha256(state))

    def test_checkpoint_filename_scope_is_single_step(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "step_001.pt").write_bytes(b"fixture")
            self.assertEqual(
                sorted(path.name for path in root.glob("*.pt")), ["step_001.pt"]
            )


if __name__ == "__main__":
    unittest.main()
