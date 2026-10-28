"""Pairwise failure-event metric matrices used by the AMOS failure comparison."""
from __future__ import annotations
import numpy as np

DIRECTIONS = {
    "kappa": "higher means more chance-adjusted dependence",
    "q_statistic": "higher means more dependence; -1 is inverse and +1 concordant",
    "disagreement_rate": "higher means more diversity and therefore less dependence",
    "double_fault_rate": "higher means more joint failure; not chance-adjusted",
}


def _metric_matrices(success: np.ndarray) -> dict[str, np.ndarray]:
    data = np.asarray(success, dtype=float)
    total = float(data.shape[1])
    n11 = data @ data.T
    success_count = np.sum(data, axis=1)
    n10 = success_count[:, None] - n11
    n01 = success_count[None, :] - n11
    n00 = total - n11 - n01 - n10
    observed = (n11 + n00) / total
    p = data.mean(axis=1)
    expected = np.outer(p, p) + np.outer(1.0 - p, 1.0 - p)
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = (observed - expected) / (1.0 - expected)
        q_value = (n11 * n00 - n01 * n10) / (n11 * n00 + n01 * n10)
    return {
        "kappa": kappa,
        "q_statistic": q_value,
        "disagreement_rate": (n01 + n10) / total,
        "double_fault_rate": n00 / total,
    }
