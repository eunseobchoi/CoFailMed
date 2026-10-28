#!/usr/bin/env python3
"""Paired gap difference between two results roots scored on identical test cases.

For each task: the 11-bundle primary pool of each root (compute_all collection, per-bundle 0.30 floor applied to
each root separately), gap = within r-bar - cross r-bar (DINOv2-S/B counted within, as in compute_all), and the
difference gap_A - gap_B with a 95% percentile interval from ONE set of bootstrap draws applied to both roots
(B=2000, seed 0; whole patients for tasks with a cluster unit, i.e. ACDC-LV and BraTS, cases otherwise; for
clustered tasks a case-level interval is added as sensitivity). Also per-bundle mean Dice (mean over seeds of the
per-case mean) for both roots.

Usage: compute_paired_gap_difference.py --root-a A --root-b B --tasks T [T ...] --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_all as ca  # noqa: E402  (collection, alignment, case units and constants of the primary scorer)
from fmpool.estimators import _bootstrap_case_indices, functional_floor_filter, within_cross_rbar  # noqa: E402

B, SEED = 2000, 0


def pool(root: Path, task: str):
    fm_jsons, _ = ca._collect_task_jsons(root, task)
    ca._validate_alignment(fm_jsons)
    per_case = ca._as_per_case_map(fm_jsons)
    kept = functional_floor_filter(per_case, min_mean_dice=ca.FUNCTIONAL_FLOOR, scope="per_fm")
    means = {fm: float(np.mean([np.mean(v) for (f, _), v in per_case.items() if f == fm]))
             for fm in sorted({f for f, _ in per_case})}
    return fm_jsons[0].test_ids, kept, means


def gap(per_case: dict, idx=None) -> float:
    sub = per_case if idx is None else {k: np.asarray(v)[idx] for k, v in per_case.items()}
    return float(within_cross_rbar(sub, family_pairs_within=ca.DINOV2_FAMILY_PAIRS, stat="pearson")["gap"])


def paired(ka: dict, kb: dict, n: int, subjects) -> dict:
    diffs = []
    for idx in _bootstrap_case_indices(n, B, SEED, subjects):
        x, y = gap(ka, idx), gap(kb, idx)
        if math.isfinite(x) and math.isfinite(y):
            diffs.append(x - y)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"ci95": [float(lo), float(hi)], "n_valid_boot": len(diffs)}


def task_result(root_a: Path, root_b: Path, task: str) -> dict:
    ids_a, ka, ma = pool(root_a, task)
    ids_b, kb, mb = pool(root_b, task)
    if ids_a != ids_b:
        raise ValueError(f"{task}: test_ids differ between roots")
    meta = ca._case_metadata(task, ids_a)
    subjects = meta["unit_ids"] if meta["cluster_unit"] is not None else None
    ga, gb = gap(ka), gap(kb)
    out = {"n_cases": len(ids_a), "unit": meta["cluster_unit"] or meta["case_unit"], "n_units": meta["n_units"],
           "gap_a": ga, "gap_b": gb, "gap_diff_a_minus_b": ga - gb,
           "gap_diff_ci95_primary": paired(ka, kb, len(ids_a), subjects),
           "K_a": len({f for f, _ in ka}), "K_b": len({f for f, _ in kb}),
           "kept_a": sorted({f for f, _ in ka}), "kept_b": sorted({f for f, _ in kb}),
           "bundle_mean_dice_a": ma, "bundle_mean_dice_b": mb,
           "bundle_mean_dice_diff_a_minus_b": {fm: ma[fm] - mb[fm] for fm in ma if fm in mb}}
    if subjects is not None:
        out["gap_diff_ci95_case_sensitivity"] = paired(ka, kb, len(ids_a), None)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root-a", type=Path, required=True)
    ap.add_argument("--root-b", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+", required=True, choices=ca.TASKS)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    d = {"script": "code/scripts/compute_paired_gap_difference.py", "root_a": str(a.root_a), "root_b": str(a.root_b),
         "bootstrap": {"B": B, "seed": SEED, "draws": "shared by both roots; patients for clustered tasks, else cases"},
         "tasks": {t: task_result(a.root_a, a.root_b, t) for t in a.tasks}}
    a.out.write_text(json.dumps(d, indent=1, sort_keys=True) + "\n")
    for t, r in d["tasks"].items():
        c = r["gap_diff_ci95_primary"]["ci95"]
        print(f"{t}: gap A {r['gap_a']:.4f} B {r['gap_b']:.4f} diff {r['gap_diff_a_minus_b']:+.4f} [{c[0]:+.4f}, {c[1]:+.4f}] "
              f"K {r['K_a']}/{r['K_b']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
