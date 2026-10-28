"""Trace loading, validation and the bundle-level Dice floor for the control scorers."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np

DINOV2_PAIR = frozenset({"dinov2_vitb14", "dinov2_vits14"})


@dataclass(frozen=True)
class Trace:
    task: str
    bundle: str
    seed: int
    test_ids: tuple[str, ...]
    dice: np.ndarray
    source: str


def _validate_trace(trace: Trace) -> None:
    if trace.dice.ndim != 1:
        raise ValueError(f"{trace.source}: per_case_dice must be one-dimensional")
    if len(trace.test_ids) != len(trace.dice):
        raise ValueError(f"{trace.source}: test_ids and per_case_dice length mismatch")
    if len(set(trace.test_ids)) != len(trace.test_ids):
        raise ValueError(f"{trace.source}: duplicate test_ids")
    if not np.all(np.isfinite(trace.dice)):
        raise ValueError(f"{trace.source}: per_case_dice contains NaN or infinity")


def load_trace(path: Path, root: Path) -> Trace:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    trace = Trace(
        task=str(payload["task"]),
        bundle=str(payload.get("fm") or path.parent.name),
        seed=int(payload["seed"]),
        test_ids=tuple(str(case_id) for case_id in payload["test_ids"]),
        dice=np.asarray(payload["per_case_dice"], dtype=float),
        source=str(path.relative_to(root)),
    )
    _validate_trace(trace)
    return trace


def apply_bundle_floor(traces: list[Trace], floor: float) -> list[Trace]:
    grouped: dict[str, list[Trace]] = {}
    for trace in traces:
        grouped.setdefault(trace.bundle, []).append(trace)
    means = {
        bundle: float(np.mean([float(np.mean(trace.dice)) for trace in members]))
        for bundle, members in grouped.items()
    }
    return [trace for trace in traces if means[trace.bundle] >= floor]
