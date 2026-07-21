"""Configuration contract for the isolated Phase-3 DrugCLIP implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE2_CHECKPOINT = (
    REPO_ROOT
    / "phase2"
    / "runs"
    / "v9_concat_fusion_partial_unfreeze_1d3d_last1_from3d_e40_v1"
    / "checkpoint_best.pt"
)
DEFAULT_RUN_ROOT = REPO_ROOT / "phase3" / "runs" / "drugclip"


@dataclass(frozen=True)
class DrugCLIPPhase3Config:
    """Algorithm-level defaults, separate from any legacy Phase-3 config."""

    schema_version: str = "pepclip-phase3-drugclip-v1"
    phase2_checkpoint: Path = DEFAULT_PHASE2_CHECKPOINT
    output_root: Path = DEFAULT_RUN_ROOT
    conformers_per_peptide: int = 10
    conformer_sampling: str = "uniform_per_pair_access"
    train_seed: int = 20260710
    valid_seed: int = 20260711
    test_seed: int = 20260712
    allowed_training_losses: tuple[str, ...] = field(
        default=("receptor_to_peptide_contrastive", "peptide_to_receptor_contrastive")
    )

    def validate(self) -> None:
        if self.schema_version != "pepclip-phase3-drugclip-v1":
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        if self.conformers_per_peptide < 1:
            raise ValueError("conformers_per_peptide must be >= 1")
        if self.conformer_sampling != "uniform_per_pair_access":
            raise ValueError("only uniform per-pair-access sampling is allowed")
        expected_root = (REPO_ROOT / "phase3" / "runs" / "drugclip").resolve()
        actual_root = self.output_root.resolve()
        if actual_root != expected_root and expected_root not in actual_root.parents:
            raise ValueError(
                "DrugCLIP outputs must stay under phase3/runs/drugclip; "
                f"received {actual_root}"
            )
