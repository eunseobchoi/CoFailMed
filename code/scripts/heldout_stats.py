#!/usr/bin/env python3
"""compute_all within/cross for one held-out task (compute_all --task only accepts primary tasks). If compute_all
finds no functional bundles, write stats/no_pool/<task>.json with the verbatim error and the per-bundle mean Dice vs the 0.30 floor,
computed from the same valid traces. Unexpected failures are propagated with a nonzero exit status.
usage: heldout_stats.py RESULTS_ROOT TASK"""
import argparse, json, sys, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_all as c  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("results_root", type=Path)
ap.add_argument("task")
a = ap.parse_args()
root, task = a.results_root, a.task
try:
    c.run_pipeline(root, root / "stats", tasks=(task,), only_within_cross=True)
    print(f"stats {task} ok")
except StopIteration as exc:
    fm_jsons, _ = c._collect_task_jsons(root, task)
    c._validate_alignment(fm_jsons)
    means = {}
    for j in fm_jsons:
        means.setdefault(j.group, []).append(float(j.per_case_dice.mean()))
    bundle = {fm: sum(v) / len(v) for fm, v in sorted(means.items())}
    kept = sorted(fm for fm, m in bundle.items() if m >= c.FUNCTIONAL_FLOOR)
    if not fm_jsons or kept:
        raise  # A nonempty functional pool or invalid inputs are not a no-pool result.
    out = root / "stats/no_pool" / f"{task}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"task": task, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                               "functional_floor": c.FUNCTIONAL_FLOOR, "bundle_mean_dice": bundle,
                               "fms_available_before_floor": sorted(bundle), "fms_kept_after_floor": kept,
                               "n_test": len(fm_jsons[0].per_case_dice) if fm_jsons else None}, indent=1) + "\n")
    print(f"stats {task}: no functional pool (K=0); record {out}")
