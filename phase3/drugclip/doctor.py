"""Read-only environment and isolation checks for Phase-3 DrugCLIP."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from .config import DrugCLIPPhase3Config


def run_doctor() -> tuple[bool, dict[str, object]]:
    config = DrugCLIPPhase3Config()
    checks: dict[str, object] = {
        "python": sys.version.split()[0],
        "active_namespace": "phase3.drugclip",
        "legacy_namespace_imported": "phase3.active_algorithm" in sys.modules,
        "phase2_checkpoint": str(config.phase2_checkpoint),
        "phase2_checkpoint_exists": config.phase2_checkpoint.is_file(),
        "output_root": str(config.output_root),
        "dependencies": {
            name: importlib.util.find_spec(name) is not None
            for name in ("torch", "numpy", "scipy", "gemmi", "lmdb", "transformers")
        },
    }
    try:
        config.validate()
        checks["config_valid"] = True
    except ValueError as exc:
        checks["config_valid"] = False
        checks["config_error"] = str(exc)

    package_root = Path(__file__).resolve().parent
    forbidden = "phase3" + ".active_algorithm"
    offenders: list[str] = []
    for path in package_root.rglob("*.py"):
        if forbidden in path.read_text(encoding="utf-8"):
            # Package documentation may name the forbidden namespace. Only
            # executable import statements violate isolation.
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith((f"from {forbidden}", f"import {forbidden}")):
                    offenders.append(str(path.relative_to(package_root)))
                    break
    checks["legacy_import_offenders"] = offenders

    deps_ok = all(checks["dependencies"].values())  # type: ignore[union-attr]
    ok = bool(
        deps_ok
        and checks["config_valid"]
        and checks["phase2_checkpoint_exists"]
        and not checks["legacy_namespace_imported"]
        and not offenders
    )
    checks["ok"] = ok
    return ok, checks


def main() -> int:
    ok, checks = run_doctor()
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
