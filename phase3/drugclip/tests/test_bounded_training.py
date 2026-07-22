from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from phase3.drugclip.train import run_epoch
from phase3.drugclip.training_state import make_grad_scaler


class _CountingSGD(torch.optim.SGD):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


class BoundedTrainingTests(unittest.TestCase):
    def test_max_steps_executes_exactly_32_optimizer_steps(self) -> None:
        model = torch.nn.Linear(1, 1)
        optimizer = _CountingSGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        scaler = make_grad_scaler(torch.device("cpu"), False)
        loader = [{"one_d": {"sample_id": [str(index)]}, "value": torch.tensor([[1.0]])} for index in range(40)]

        def fake_forward(current_model, batch, device):
            loss = current_model(batch["value"]).square().mean()
            return {
                "loss_total": loss,
                "loss_receptor_to_peptide": loss,
                "loss_peptide_to_receptor": loss,
            }

        observed_steps: list[tuple[int, int]] = []
        with patch("phase3.drugclip.train.forward_and_known_positive_loss", side_effect=fake_forward):
            metrics, increment = run_epoch(
                model, loader, optimizer, scaler, torch.device("cpu"), True, False, 1.0,
                scheduler=scheduler, max_steps=32,
                on_step=lambda step, batch: observed_steps.append((step, batch)),
            )

        self.assertEqual(optimizer.step_calls, 32)
        self.assertEqual(increment, 32)
        self.assertEqual(scheduler.last_epoch, 32)
        self.assertEqual(metrics["batches"], 32)
        self.assertEqual(observed_steps[-1], (32, 32))


if __name__ == "__main__":
    unittest.main()
