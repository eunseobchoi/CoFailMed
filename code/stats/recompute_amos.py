
"""Recompute the AMOS22 statistics from a finalized per-case CSV (not included).

With --reference-metrics, every numeric field is also compared with an earlier metrics.json.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np


SCORER_ROOT = Path(__file__).resolve().parent
SCORER = SCORER_ROOT / "amos/aggregate.py"
HELPER_SOURCES = {
    "effective_size": SCORER_ROOT / "amos/analysis/effective_size.py",
    "failure_comparison": SCORER_ROOT / "amos/analysis/failure_comparison.py",
}
SYSTEMS = ("mrseg_fold0", "mrseg_fold1", "totalseg")
TOLERANCE = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_scorer() -> Any:
    sys.path.insert(0, str(SCORER.parent))
    spec = importlib.util.spec_from_file_location("amos_original_aggregate", SCORER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load scorer: {SCORER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def numeric_leaves(value: Any, prefix: str = "") -> dict[str, float]:
    if isinstance(value, dict):
        output: dict[str, float] = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            output.update(numeric_leaves(item, path))
        return output
    if isinstance(value, list):
        output = {}
        for index, item in enumerate(value):
            output.update(numeric_leaves(item, f"{prefix}[{index}]"))
        return output
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {prefix: float(value)}
    return {}


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"case", "modality", *SYSTEMS}
    if set(frame.columns) != required:
        raise ValueError(
            f"unexpected CSV columns: {list(frame.columns)}; expected {sorted(required)}"
        )
    if frame["case"].isna().any() or (frame["case"].astype(str).str.len() == 0).any():
        raise ValueError("case IDs must be non-empty")
    duplicates = frame.loc[frame["case"].duplicated(keep=False), "case"].astype(str)
    if not duplicates.empty:
        raise ValueError(f"duplicate case IDs: {sorted(duplicates.unique())}")
    try:
        scores = frame[list(SYSTEMS)].to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Dice columns must be numeric") from error
    if not np.isfinite(scores).all():
        raise ValueError("Dice columns must contain only finite values")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("Dice values must be within [0, 1]")
    ordered = frame.sort_values(["modality", "case"]).reset_index(drop=True)
    counts = ordered.groupby("modality").size().to_dict()
    if counts != {"ct": 60, "mri": 60}:
        raise ValueError(f"expected 60 CT and 60 MRI rows, got {counts}")
    return ordered


def write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True,
                        help="finalized comparison_per_case.csv (60 CT + 60 MRI rows)")
    parser.add_argument("--reference-metrics", type=Path,
                        help="optional earlier metrics.json; its by_modality fields are compared")
    parser.add_argument("--out", type=Path, required=True, help="output JSON path")
    args = parser.parse_args()

    frame = validate_frame(pd.read_csv(args.input_csv))

    scorer = load_scorer()
    recomputed = scorer.aggregate_metrics(frame)
    inputs = {
        "comparison_per_case_sha256": sha256(args.input_csv),
        "scorer_source_sha256": sha256(SCORER),
        "helper_source_sha256": {
            name: sha256(path) for name, path in HELPER_SOURCES.items()
        },
    }
    if args.reference_metrics is None:
        write_payload(args.out, {"status": "RECOMPUTED", "inputs": inputs,
                                 "recomputed_by_modality": recomputed})
        print(f"RECOMPUTED: no reference given; output={args.out}")
        return 0
    reference = json.loads(args.reference_metrics.read_text(encoding="utf-8"))[
        "by_modality"
    ]
    actual_numbers = numeric_leaves(recomputed)
    expected_numbers = numeric_leaves(reference)
    missing = sorted(expected_numbers.keys() - actual_numbers.keys())
    extra = sorted(actual_numbers.keys() - expected_numbers.keys())
    # Numerical-eligibility diagnostics are additive; every reference field is still required.
    allowed_extra = {
        key: actual_numbers[key] for key in extra
        if key.endswith(('.support.n_pools_numerically_ineligible', '.support.n_pools_finite'))
        and '.M_eff.' in key
    }
    if missing or set(extra) != set(allowed_extra) or any(value != 0 for value in allowed_extra.values()):
        raise RuntimeError(f"numeric field mismatch: missing={missing}, extra={extra}")
    differences = {
        key: abs(actual_numbers[key] - expected_numbers[key])
        for key in expected_numbers
    }
    failures = {
        key: difference
        for key, difference in differences.items()
        if not math.isfinite(difference) or difference > TOLERANCE
    }
    payload = {
        "status": "PASS" if not failures else "FAIL",
        "comparison": {
            "scope": "all numeric fields under metrics.json/by_modality",
            "tolerance": TOLERANCE,
            "fields_compared": len(differences),
            "max_absolute_difference": max(differences.values(), default=0.0),
            "failures": failures,
            "added_zero_eligibility_diagnostics": allowed_extra,
        },
        "inputs": {**inputs, "reference_metrics_sha256": sha256(args.reference_metrics)},
        "recomputed_by_modality": recomputed,
    }
    write_payload(args.out, payload)
    print(
        f"{payload['status']}: {len(differences)} numeric fields; "
        f"max_abs_diff={payload['comparison']['max_absolute_difference']:.3g}; "
        f"output={args.out}"
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
