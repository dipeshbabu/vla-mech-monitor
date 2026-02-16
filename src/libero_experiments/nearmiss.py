"""Near-miss rollout generation utilities.

We treat a *near-miss* as a rollout that is close to the nominal task but more
error-prone. The minimal, repo-local way to generate this is to perturb the
natural-language task description passed to the policy.

This intentionally stays lightweight: it does not require extra labels,
simulation changes, or external datasets.
"""

from __future__ import annotations

import random
import re
from typing import Literal, Tuple


def perturb_task_description(
    task_description: str,
    mode: Literal["none", "lr_swap", "synonym", "token_dropout"],
    *,
    token_dropout_p: float = 0.08,
    lr_swap_prob: float = 0.5,
    seed: int = 0,
) -> Tuple[str, str]:
    """Return (effective_description, near_miss_kind).

    This is the main entry point used by eval_libero.py.
    """

    if mode == "none":
        return task_description, "none"

    if mode == "lr_swap":
        out = maybe_perturb_instruction(
            task_description,
            enabled=True,
            mode="lr_swap",
            p_apply=lr_swap_prob,
            seed=seed,
        )
        return out, "lr_swap" if out != task_description else "none"

    if mode == "synonym":
        out = maybe_perturb_instruction(
            task_description,
            enabled=True,
            mode="synonym",
            p_apply=1.0,
            seed=seed,
        )
        return out, "synonym" if out != task_description else "none"

    if mode == "token_dropout":
        out = _token_dropout(task_description, p=token_dropout_p, seed=seed)
        return out, "token_dropout" if out != task_description else "none"

    return task_description, "none"


def maybe_perturb_instruction(
    instruction: str,
    *,
    enabled: bool,
    mode: Literal["lr_swap", "synonym"],
    p_apply: float,
    seed: int,
) -> str:
    """Return a (possibly) perturbed instruction.

    Args:
        instruction: Original task description.
        enabled: Whether perturbations are enabled.
        mode: Perturbation type.
        p_apply: Probability of applying perturbation (0..1).
        seed: Seed for deterministic perturbations.
    """

    if not enabled:
        return instruction

    rng = random.Random(seed)
    if rng.random() > max(0.0, min(1.0, p_apply)):
        return instruction

    if mode == "lr_swap":
        return _swap_left_right(instruction)
    if mode == "synonym":
        return _light_synonym_swap(instruction, rng=rng)
    return instruction


def _swap_left_right(text: str) -> str:
    # Use placeholders to avoid double replacement.
    text2 = re.sub(r"\bleft\b", "__TMP_LEFT__", text, flags=re.IGNORECASE)
    text2 = re.sub(r"\bright\b", "left", text2, flags=re.IGNORECASE)
    text2 = re.sub(r"__TMP_LEFT__", "right", text2)
    return text2


def _light_synonym_swap(text: str, *, rng: random.Random) -> str:
    # Very small set of safe, meaning-preserving swaps.
    swaps = [
        (r"\bgrasp\b", "grab"),
        (r"\bgrab\b", "grasp"),
        (r"\bplace\b", "put"),
        (r"\bput\b", "place"),
        (r"\bmove\b", "shift"),
        (r"\bshift\b", "move"),
    ]

    # Apply at most one swap to keep perturbations mild.
    pattern, repl = rng.choice(swaps)
    return re.sub(pattern, repl, text, flags=re.IGNORECASE)


def _token_dropout(text: str, *, p: float, seed: int) -> str:
    """Drop a small fraction of tokens (very mild, deterministic)."""

    p = max(0.0, min(0.5, p))
    tokens = text.split()
    if len(tokens) <= 3 or p <= 0:
        return text

    rng = random.Random(seed)

    # Keep at least 3 tokens to avoid empty instructions.
    kept = [tok for tok in tokens if rng.random() > p]
    if len(kept) < 3:
        kept = tokens[:3]
    return " ".join(kept)
