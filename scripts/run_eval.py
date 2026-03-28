"""CLI entrypoint for a single LIBERO eval run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from libero_experiments.config import load_config, parse_overrides
from libero_experiments.eval_libero import eval_libero


def _default_interventions_path() -> str:
    return str(REPO_ROOT / "configs" / "interventions" / "dictionaries.yaml")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to run config YAML")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values (dot notation), e.g. env.num_trials_per_task=5",
    )
    parser.add_argument(
        "--interventions",
        default=_default_interventions_path(),
        help="Path to intervention dictionaries YAML",
    )
    args = parser.parse_args()

    overrides = parse_overrides(args.override)
    cfg = load_config(args.config, overrides)
    eval_libero(cfg, args.interventions)


if __name__ == "__main__":
    main()
