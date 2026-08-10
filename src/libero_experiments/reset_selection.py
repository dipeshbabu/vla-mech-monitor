"""Validated selection of LIBERO initial states."""

from __future__ import annotations

from collections.abc import Sequence


def resolve_initial_state_indices(
    *,
    available_count: int,
    num_trials: int,
    offset: int = 0,
    explicit_indices: Sequence[int] | None = None,
) -> list[int]:
    """Return the ordered initial-state indices for one task.

    An explicit list is mutually exclusive with a nonzero offset. Requiring the
    list length to match ``num_trials`` prevents a run from silently evaluating
    a different number of resets than its configuration reports.
    """

    if available_count <= 0:
        raise ValueError(f"available_count must be positive, got {available_count}")
    if num_trials <= 0:
        raise ValueError(f"num_trials must be positive, got {num_trials}")
    if offset < 0:
        raise ValueError(f"initial_state_offset must be nonnegative, got {offset}")

    requested = [int(index) for index in (explicit_indices or [])]
    if requested and offset != 0:
        raise ValueError("initial_state_indices and a nonzero initial_state_offset are mutually exclusive")
    if requested and len(requested) != num_trials:
        raise ValueError(
            "initial_state_indices length must equal num_trials_per_task: "
            f"{len(requested)} != {num_trials}"
        )

    selected = requested or list(range(offset, offset + num_trials))
    if len(set(selected)) != len(selected):
        raise ValueError(f"initial_state_indices contains duplicates: {selected}")

    bad = [index for index in selected if index < 0 or index >= available_count]
    if bad:
        raise ValueError(
            f"initial-state indices out of range for {available_count} available states: {bad}"
        )
    return selected
