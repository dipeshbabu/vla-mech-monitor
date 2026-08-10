from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from libero_experiments.monitoring import ClosedLoopController, WarningController


def test_closed_loop_controller_reports_trigger() -> None:
    controller = ClosedLoopController(tau=0.5, alpha=2.0, patience=2, duration=1)
    assert controller.step(0.6) == (0.0, False)
    assert controller.step(0.7) == (-2.0, True)


def test_warning_controller_reports_trigger() -> None:
    controller = WarningController(tau=0.5, patience=2, duration=1)
    assert controller.step(0.6) == (False, False)
    assert controller.step(0.7) == (True, True)
