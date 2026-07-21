"""Command entry point for the active Phase-3 package."""

from __future__ import annotations

import argparse

from .doctor import main as doctor_main


def main() -> int:
    parser = argparse.ArgumentParser(description="PepCLIP Phase-3 DrugCLIP tools")
    parser.add_argument("command", choices=("doctor",))
    args = parser.parse_args()
    if args.command == "doctor":
        return doctor_main()
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
