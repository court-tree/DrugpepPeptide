from __future__ import annotations

import argparse
import re
import sys
from datetime import date, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "agent"

REQUIRED_HEADINGS = {
    "PROJECT_STATE.md": (
        "# PepCLIP Current Project State",
        "## Current Phase",
        "## Current Problem",
        "## Single Next Action",
        "## Workspace Safety",
    ),
    "DECISIONS.md": ("# PepCLIP Decisions And Resolved Questions",),
    "ACTIVE_WORK.md": ("# PepCLIP Active Work Claims",),
    "HANDOFF_TEMPLATE.md": ("# PepCLIP Session Handoff Template",),
    "README.md": ("# PepCLIP Session Bridge",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the repository-owned Codex session bridge."
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=7,
        help="maximum allowed age of PROJECT_STATE.md (default: 7)",
    )
    return parser.parse_args()


def read_required(path: Path, errors: list[str]) -> str:
    if not path.is_file():
        errors.append(f"missing required file: {path.relative_to(ROOT)}")
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        errors.append(f"not valid UTF-8: {path.relative_to(ROOT)}")
        return ""


def validate_date_marker(
    text: str,
    marker: str,
    filename: str,
    max_age_days: int,
    errors: list[str],
) -> None:
    match = re.search(rf"^{re.escape(marker)}:\s*(\d{{4}}-\d{{2}}-\d{{2}})\s*$", text, re.M)
    if not match:
        errors.append(f"{filename} lacks '{marker}: YYYY-MM-DD'")
        return
    marked_date = datetime.strptime(match.group(1), "%Y-%m-%d").date()
    age = (date.today() - marked_date).days
    if age < 0:
        errors.append(f"{filename} {marker} date is in the future")
    elif age > max_age_days:
        errors.append(
            f"{filename} is stale: {age} days old (limit {max_age_days})"
        )


def validate_state(text: str, max_age_days: int, errors: list[str]) -> None:
    validate_date_marker(text, "Last verified", "PROJECT_STATE.md", max_age_days, errors)

    next_action = re.search(
        r"^## Single Next Action\s*$\n(?P<body>.*?)(?=^## |\Z)",
        text,
        re.M | re.S,
    )
    if not next_action or not next_action.group("body").strip():
        errors.append("PROJECT_STATE.md has no bounded next action")


def validate_active_work(text: str, max_age_days: int, errors: list[str]) -> None:
    validate_date_marker(text, "Last checked", "ACTIVE_WORK.md", max_age_days, errors)
    rows = [line for line in text.splitlines() if line.startswith("|")]
    data_rows = [
        row
        for row in rows
        if "Session / owner" not in row and not re.fullmatch(r"[|:\- ]+", row)
    ]
    if not data_rows:
        errors.append("ACTIVE_WORK.md has no claim table data row")
        return

    idle_rows = [row for row in data_rows if row.split("|")[1].strip() == "None"]
    if idle_rows and len(data_rows) != 1:
        errors.append("ACTIVE_WORK.md idle row cannot coexist with active claims")

    owners: set[str] = set()
    for row in data_rows:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        if len(cells) != 6:
            errors.append(f"ACTIVE_WORK.md malformed row: {row}")
            continue
        owner, _, source_scope, output_scope, _, status = cells
        if owner == "None":
            if cells != ["None", "No active claimed work", "-", "-", "-", "idle"]:
                errors.append("ACTIVE_WORK.md idle row must use the canonical placeholder")
            continue
        if status == "idle":
            errors.append(f"ACTIVE_WORK.md active owner cannot have idle status: {owner}")
        if owner in owners:
            errors.append(f"ACTIVE_WORK.md duplicate owner/session: {owner}")
        owners.add(owner)
        invalid_source = source_scope in {
            "",
            "phase1/",
            "phase2/",
            "phase3/",
            "project/",
        }
        invalid_output = output_scope in {
            "",
            "phase1/runs/",
            "phase2/runs/",
            "phase3/runs/",
        }
        if invalid_source:
            errors.append(f"ACTIVE_WORK.md has broad/empty source scope for {owner}")
        if invalid_output:
            errors.append(f"ACTIVE_WORK.md has broad/empty output scope for {owner}")
        if source_scope == "-" and output_scope == "-":
            errors.append(f"ACTIVE_WORK.md has no concrete scope for {owner}")


def main() -> int:
    args = parse_args()
    errors: list[str] = []

    agents = read_required(ROOT / "AGENTS.md", errors)
    for required_ref in (
        "docs/agent/PROJECT_STATE.md",
        "docs/agent/DECISIONS.md",
        "docs/agent/ACTIVE_WORK.md",
    ):
        if required_ref not in agents:
            errors.append(f"AGENTS.md does not reference {required_ref}")

    contents: dict[str, str] = {}
    for name, headings in REQUIRED_HEADINGS.items():
        text = read_required(DOCS / name, errors)
        contents[name] = text
        for heading in headings:
            if heading not in text:
                errors.append(f"{name} lacks required heading: {heading}")

    validate_state(contents["PROJECT_STATE.md"], args.max_age_days, errors)
    validate_active_work(contents["ACTIVE_WORK.md"], args.max_age_days, errors)

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("PASS: PepCLIP session bridge is present, current, and structurally valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
