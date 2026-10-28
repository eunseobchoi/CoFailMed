#!/usr/bin/env python3
"""Low-M_eff frequency of the recorded architecture-diversity pools of a pool-selection output.

``analyze_task`` (``analysis.pool_meff``) reproduces every recorded split (anchor, P33, candidates, quality gate,
pool, evaluation) before scoring the recorded pool with ``split_meff_record`` (support rule >=5 failures and >=5
passes per member; M_eff < 2.0 is low; unsupported splits are counted as not-low). Taskwise frequency
intervals use two-sided 95% Clopper-Pearson limits for random splits conditional on the fixed cohort,
not uncertainty over new patients or datasets. M_eff values are unchanged. The supplied
artifact is checked itself (status COMPLETE, split seeds 0..n-1 in every task); the task list is the
artifact's own (a restricted ``scope`` when present), recorded in the output.
Usage: python3 -m analysis.pool_low_meff --selection pool_selection.json --results-root ROOT --output OUT.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

from analysis.common import write_json
from analysis import pool_meff


def selection_tasks(selection: dict[str, Any]) -> tuple[str, ...]:
    if selection.get("status") != "COMPLETE":
        raise ValueError("pool-selection artifact is not COMPLETE")
    tasks = tuple(selection.get("scope", {}).get("tasks") or selection["protocol"]["tasks"])
    if not tasks or set(tasks) - set(selection["tasks"]):
        raise ValueError(f"artifact lacks task payloads for {sorted(set(tasks) - set(selection['tasks']))}")
    n = len(selection["tasks"][tasks[0]]["primary_split_records"])
    for task in tasks:
        seeds = [r.get("split_seed") for r in selection["tasks"][task]["primary_split_records"]]
        if seeds != list(range(n)):
            raise ValueError(f"{task}: recorded split seeds are not exactly 0..{n - 1}")
    return tasks


def exact_binomial_interval(successes: int, trials: int) -> list[float] | None:
    """Conditional split-frequency interval; no interval exists for zero trials."""
    if trials == 0 and successes == 0:
        return None
    ci = binomtest(successes, trials).proportion_ci(confidence_level=0.95, method="exact")
    return [float(ci.low), float(ci.high)]


def run(selection_path: Path, results_root: Path) -> dict[str, Any]:
    raw = selection_path.read_bytes()
    selection = json.loads(raw)
    tasks = selection_tasks(selection)
    n = len(selection["tasks"][tasks[0]]["primary_split_records"])
    bootstrap = np.random.default_rng(pool_meff.RANDOM_SEED).integers(0, n, size=(pool_meff.BOOTSTRAP_B, n))
    per_task = {t: pool_meff.analyze_task(task=t, task_payload=selection["tasks"][t], results_root=results_root,
                                   bootstrap_indices=bootstrap) for t in tasks}
    for result in per_task.values():
        for cell in result["summary"].values():
            if isinstance(cell, dict) and {"ci95", "numerator", "denominator"} <= cell.keys():
                cell["ci95"] = exact_binomial_interval(cell["numerator"], cell["denominator"])
    records = [r for t in tasks for r in per_task[t]["split_records"]]
    low, valid = sum(r["low_M_eff"] for r in records), sum(r["support_valid"] for r in records)
    return {
        "schema_version": "l2ps_low_meff_v2",
        "status": "PASS",
        "method": "recorded selection replay and M_eff scoring; exact binomial frequency intervals",
        "artifact_checks": "status COMPLETE; split seeds 0..n-1 in every task",
        "selection_artifact": {"path": str(selection_path), "sha256": hashlib.sha256(raw).hexdigest(),
                               "generated_at": selection.get("generated_at"), "scope": selection.get("scope")},
        "results_root": str(results_root),
        "tasks": list(tasks),
        "splits_per_task": n,
        "low_M_eff_cutoff": pool_meff.LOW_MEFF_CUTOFF,
        "frequency_interval": {"method": "Clopper-Pearson", "confidence_level": 0.95,
                               "scope": "random-split Monte Carlo uncertainty conditional on the fixed cohort",
                               "new_patient_uncertainty": False},
        "per_task": {t: per_task[t]["summary"] for t in tasks},
        "all_task_split_counts": {
            "n_task_splits": len(records), "detected_low_M_eff": int(low), "support_valid": int(valid),
            "unsupported": int(len(records) - valid), "low_M_eff_fraction_all_task_splits": low / len(records),
            "low_M_eff_fraction_support_valid": low / valid if valid else None,
        },
        "split_records": {t: per_task[t]["split_records"] for t in tasks},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    payload = run(args.selection, args.results_root)
    write_json(args.output, payload)
    c = payload["all_task_split_counts"]
    print(f"PASS {args.output}: low M_eff {c['detected_low_M_eff']}/{c['n_task_splits']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
