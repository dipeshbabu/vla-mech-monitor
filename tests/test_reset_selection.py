from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from libero_experiments.reset_selection import resolve_initial_state_indices


def test_default_selection_starts_at_zero() -> None:
    assert resolve_initial_state_indices(available_count=50, num_trials=3) == [0, 1, 2]


def test_offset_selects_disjoint_block() -> None:
    assert resolve_initial_state_indices(available_count=50, num_trials=3, offset=20) == [20, 21, 22]


def test_explicit_manifest_preserves_order() -> None:
    assert resolve_initial_state_indices(
        available_count=50,
        num_trials=3,
        explicit_indices=[12, 4, 31],
    ) == [12, 4, 31]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"offset": 20, "explicit_indices": [1, 2]}, "mutually exclusive"),
        ({"explicit_indices": [1, 2]}, "length must equal"),
        ({"explicit_indices": [1, 1, 2]}, "duplicates"),
        ({"offset": 49}, "out of range"),
    ],
)
def test_invalid_selection_fails_loudly(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        resolve_initial_state_indices(available_count=50, num_trials=3, **kwargs)
