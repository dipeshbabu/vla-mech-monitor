from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class ProbeModel:
    clf: LogisticRegression
    modes: list[str]


def fit_logistic(X: np.ndarray, y: np.ndarray, modes: list[str]) -> ProbeModel:
    # y: integer class index
    clf = LogisticRegression(max_iter=2000, n_jobs=-1, multi_class="auto")
    clf.fit(X, y)
    return ProbeModel(clf=clf, modes=modes)


def predict_proba(model: ProbeModel, X: np.ndarray) -> np.ndarray:
    return model.clf.predict_proba(X)
