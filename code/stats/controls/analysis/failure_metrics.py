"""Binary failure-event metrics (kappa, Q, disagreement, double fault) for the control pools."""
from __future__ import annotations
import numpy as np
from analysis.trace_validation import DINOV2_PAIR

DIRECTIONS = {
    "kappa": "higher means more chance-adjusted dependence",
    "q_statistic": "higher means more dependence; -1 is inverse and +1 concordant",
    "disagreement_rate": "higher means more diversity and therefore less dependence",
    "double_fault_rate": "higher means more joint failure; not chance-adjusted",
}
KAPPA_NEVER_FAIL = ("official", "exclude")


def _check_kappa_never_fail(kappa_never_fail: str) -> None:
    if kappa_never_fail not in KAPPA_NEVER_FAIL:
        raise ValueError(f"kappa_never_fail must be one of {KAPPA_NEVER_FAIL}, got {kappa_never_fail!r}")


def _binary_inputs(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_arr = np.asarray(left, dtype=bool)
    right_arr = np.asarray(right, dtype=bool)
    if left_arr.shape != right_arr.shape or left_arr.ndim != 1 or left_arr.size == 0:
        raise ValueError("binary metric inputs must be non-empty matched vectors")
    return left_arr, right_arr


def _counts(left: np.ndarray, right: np.ndarray) -> tuple[int, int, int, int]:
    left_arr, right_arr = _binary_inputs(left, right)
    n11 = int(np.sum(left_arr & right_arr))
    n00 = int(np.sum(~left_arr & ~right_arr))
    n01 = int(np.sum(~left_arr & right_arr))
    n10 = int(np.sum(left_arr & ~right_arr))
    return n11, n00, n01, n10


def cohen_kappa(
    left: np.ndarray, right: np.ndarray, kappa_never_fail: str = "official"
) -> float | None:
    _check_kappa_never_fail(kappa_never_fail)
    left_arr, right_arr = _binary_inputs(left, right)
    observed = float(np.mean(left_arr == right_arr))
    p_left = float(np.mean(left_arr))
    p_right = float(np.mean(right_arr))
    expected = p_left * p_right + (1.0 - p_left) * (1.0 - p_right)
    denominator = 1.0 - expected
    if kappa_never_fail == "official" and observed == 1.0:
        return 1.0
    return None if denominator == 0.0 else (observed - expected) / denominator


def q_statistic(left: np.ndarray, right: np.ndarray) -> float | None:
    n11, n00, n01, n10 = _counts(left, right)
    numerator = n11 * n00 - n01 * n10
    denominator = n11 * n00 + n01 * n10
    return None if denominator == 0 else float(numerator / denominator)


def disagreement_rate(left: np.ndarray, right: np.ndarray) -> float:
    n11, n00, n01, n10 = _counts(left, right)
    return float((n01 + n10) / (n11 + n00 + n01 + n10))


def double_fault_rate(left: np.ndarray, right: np.ndarray) -> float:
    n11, n00, n01, n10 = _counts(left, right)
    return float(n00 / (n11 + n00 + n01 + n10))


def _same_mask(keys: list[tuple[str, int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pair_i, pair_j = np.triu_indices(len(keys), k=1)
    same = np.asarray(
        [
            keys[i][0] == keys[j][0]
            or frozenset({keys[i][0], keys[j][0]}) == DINOV2_PAIR
            for i, j in zip(pair_i, pair_j, strict=True)
        ],
        dtype=bool,
    )
    return pair_i, pair_j, same


def _metric_matrices(success: np.ndarray, kappa_never_fail: str = "official") -> dict[str, np.ndarray]:
    _check_kappa_never_fail(kappa_never_fail)
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
    if kappa_never_fail == "official":
        kappa = np.where(observed == 1.0, 1.0, kappa)
    return {
        "kappa": kappa,
        "q_statistic": q_value,
        "disagreement_rate": (n01 + n10) / total,
        "double_fault_rate": n00 / total,
    }
