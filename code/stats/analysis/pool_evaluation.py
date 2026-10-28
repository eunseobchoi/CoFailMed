"""Point metrics for selected pools; repeated inference lives in pool_repeated."""
from __future__ import annotations
import numpy as np
from analysis.pool_core import EvaluationVectors



def point(vectors: EvaluationVectors) -> dict[str, float | None]:
    recovery = (
        float(np.mean(vectors.recovered[vectors.anchor_fail]))
        if np.any(vectors.anchor_fail)
        else None
    )
    return {
        "all_fail_probability": float(np.mean(vectors.all_fail)),
        "conditional_recovery": recovery,
        "marginal_dice": float(np.mean(vectors.marginal_dice)),
    }
