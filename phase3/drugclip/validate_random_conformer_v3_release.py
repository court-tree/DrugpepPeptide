"""Read-only release-integrity validator for random_conformer_v3."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def validate(v2_dir: str | Path, v3_dir: str | Path) -> dict[str, object]:
    v2, v3 = Path(v2_dir).resolve(), Path(v3_dir).resolve()
    manifest = json.loads((v3 / "DATA_MANIFEST.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    if manifest.get("dataset_version") != "random_conformer_v3": errors.append("dataset_version")
    if manifest.get("parent_dataset") != "random_conformer_v2": errors.append("parent_dataset")
    if manifest.get("absolute_paths_recorded") is not False: errors.append("absolute_paths_recorded")
    checked = 0
    for entry in manifest.get("formal_files", []):
        relative = str(entry["relative_path"])
        if Path(relative).is_absolute(): errors.append(f"absolute_manifest_path:{relative}")
        path = v3 / relative
        if not path.is_file(): errors.append(f"missing:{relative}"); continue
        if path.stat().st_size != int(entry["bytes"]): errors.append(f"size:{relative}")
        if sha256_file(path) != str(entry["sha256"]): errors.append(f"sha256:{relative}")
        checked += 1
    parent_checked = 0
    for relative, expected in manifest.get("parent_core_sha256", {}).items():
        path = v2 / relative
        if not path.is_file(): errors.append(f"missing_parent:{relative}"); continue
        if sha256_file(path) != expected: errors.append(f"parent_changed:{relative}")
        parent_checked += 1
    return {
        "validator_id": "drugclip-random-conformer-v3-release-validator-v1",
        "passed": not errors,
        "formal_files_checked": checked,
        "parent_v2_files_checked": parent_checked,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2_dir", required=True); parser.add_argument("--v3_dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = validate(**vars(parse_args()))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
