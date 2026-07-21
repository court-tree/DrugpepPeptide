"""Active DrugCLIP-style Phase-3 fine-tuning namespace.

This package must stay independent of ``phase3.active_algorithm``. Public
imports are intentionally minimal while the new implementation is built.
"""

from .config import DrugCLIPPhase3Config

__all__ = ["DrugCLIPPhase3Config"]
__version__ = "0.1.0"
