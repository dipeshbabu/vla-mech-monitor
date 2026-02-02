from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class TriggerState:
    consec_count: int = 0
    cooldown_left: int = 0
    last_mode: Optional[str] = None


def update_trigger(
    risk_by_mode: Dict[str, float],
    tau: float,
    consec: int,
    cooldown_steps: int,
    state: TriggerState,
) -> Tuple[bool, Optional[str], TriggerState]:
    """
    Returns (trigger_now, mode, new_state).
    """
    if state.cooldown_left > 0:
        state.cooldown_left -= 1
        state.consec_count = 0
        return False, None, state

    if not risk_by_mode:
        state.consec_count = 0
        return False, None, state

    mode = max(risk_by_mode, key=lambda k: risk_by_mode[k])
    val = risk_by_mode[mode]

    if val > tau:
        state.consec_count += 1
        if state.consec_count >= consec:
            state.cooldown_left = cooldown_steps
            state.consec_count = 0
            state.last_mode = mode
            return True, mode, state
        return False, None, state

    state.consec_count = 0
    return False, None, state
