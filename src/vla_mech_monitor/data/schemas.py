from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

FailureType = Literal["none", "wrong_object",
                      "wrong_location", "drop", "goal_drift", "other"]


class PerturbationSpec(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class EpisodeSpec(BaseModel):
    episode_id: str
    task_id: str
    seed: int
    instruction: str
    perturbation: PerturbationSpec


class StepLog(BaseModel):
    t: int
    action: List[float]
    reward: Optional[float] = None
    info: Dict[str, Any] = Field(default_factory=dict)


class EpisodeLog(BaseModel):
    episode_id: str
    task_id: str
    seed: int
    instruction: str
    perturbation: Optional[PerturbationSpec] = None

    success: bool
    failure_type: FailureType = "none"
    fail_step: Optional[int] = None

    steps: List[StepLog] = Field(default_factory=list)

    # Optional pointers to activation files, rather than storing huge arrays in JSON
    activations_ref: Optional[str] = None
