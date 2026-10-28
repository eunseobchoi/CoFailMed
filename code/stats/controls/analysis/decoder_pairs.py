"""Same-bundle/release and cross-bundle pair indices for the UNet-skip control."""
from __future__ import annotations
import numpy as np
from analysis.trace_validation import DINOV2_PAIR



def _pair_masks(
    keys: list[tuple[str, int]],
) -> tuple[np.ndarray, np.ndarray]:
    within: list[tuple[int, int]] = []
    cross: list[tuple[int, int]] = []
    for left_index, left in enumerate(keys):
        for right_index in range(left_index + 1, len(keys)):
            right = keys[right_index]
            same = (
                left[0] == right[0]
                or frozenset({left[0], right[0]}) == DINOV2_PAIR
            )
            (within if same else cross).append((left_index, right_index))
    return np.asarray(within, dtype=int), np.asarray(cross, dtype=int)
