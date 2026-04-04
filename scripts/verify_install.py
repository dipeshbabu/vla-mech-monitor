"""Smoke-test the environment and core repo imports after installation."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


DEPENDENCY_IMPORTS = [
    "numpy",
    "PIL",
    "yaml",
    "imageio",
    "sklearn",
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "accelerate",
    "huggingface_hub",
    "libero",
    "robosuite",
    "bddl",
    "easydict",
    "cloudpickle",
    "gym",
    "mujoco",
]

REPO_IMPORTS = [
    "libero_experiments.config",
    "libero_experiments.libero_utils",
    "libero_experiments.model",
    "libero_experiments.eval_libero",
]


def _check_imports(module_names: list[str]) -> list[str]:
    failures: list[str] = []
    for name in module_names:
        try:
            importlib.import_module(name)
        except Exception as exc:
            failures.append(f"{name}: {exc}")
    return failures


def _run_local_smoke_checks() -> list[str]:
    failures: list[str] = []

    import numpy as np
    from PIL import Image

    from libero_experiments.config import load_config, parse_overrides
    from libero_experiments.libero_utils import resize_image
    from libero_experiments.model import crop_and_resize

    cfg = load_config(REPO_ROOT / "configs" / "warning_noop.yaml")
    if cfg.model.family != "openvla":
        failures.append(f"Unexpected default model family: {cfg.model.family}")

    overrides = parse_overrides(
        [
            "env.num_trials_per_task=5",
            "monitor.nearmiss.visual.enabled=true",
            "monitor.nearmiss.visual.kinds=[occlusion]",
        ]
    )
    if overrides["env"]["num_trials_per_task"] != 5:
        failures.append("parse_overrides did not parse integer override correctly")
    if overrides["monitor"]["nearmiss"]["visual"]["kinds"] != ["occlusion"]:
        failures.append("parse_overrides did not parse list override correctly")

    image = Image.fromarray(np.zeros((32, 48, 3), dtype=np.uint8))
    cropped = crop_and_resize(image, crop_scale=0.9)
    if cropped.size != (224, 224):
        failures.append(f"crop_and_resize returned wrong size: {cropped.size}")

    resized = resize_image(np.zeros((32, 48, 3), dtype=np.uint8), (16, 20))
    if resized.shape != (16, 20, 3):
        failures.append(f"resize_image returned wrong shape: {resized.shape}")

    return failures


def main() -> int:
    failures: list[str] = []
    failures.extend(_check_imports(DEPENDENCY_IMPORTS))
    failures.extend(_check_imports(REPO_IMPORTS))

    if not failures:
        failures.extend(_run_local_smoke_checks())

    if failures:
        print("Install verification failed:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print("Install verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
