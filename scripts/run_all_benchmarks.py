from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


DEFAULT_COMMANDS = [
    ["scripts/run_materials_signal.py", "--config", "configs/materials_signal.json"],
    ["scripts/run_external_tool_wear.py", "--config", "configs/external_tool_wear_vicomtech.json"],
    ["scripts/run_external_uniwear.py", "--config", "configs/external_uniwear_tool_wear.json"],
    ["scripts/run_external_concrete.py", "--config", "configs/external_concrete_strength.json"],
    ["scripts/run_external_commons.py", "--config", "configs/external_commons_microscopy.json"],
    ["scripts/build_materials_ai_leaderboard.py"],
]


def run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run([PYTHON, *command], cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the separate materials-ML project benchmark suite.")
    parser.add_argument("--include-sem", action="store_true", help="Also run the slower SEM quick benchmark.")
    args = parser.parse_args()
    commands = DEFAULT_COMMANDS.copy()
    if args.include_sem:
        commands.insert(0, ["scripts/run_sem_suite.py", "--config", "configs/sem_suite.json"])
    for command in commands:
        run_command(command)


if __name__ == "__main__":
    main()
