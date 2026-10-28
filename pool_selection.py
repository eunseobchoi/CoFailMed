"""Family-selected pools: 200 calibration/evaluation half-splits per task.

Calibration alone determines the bundle floor, ranking, pool and anchor P33.
ACDC/BraTS means weight patients equally; P33 and failures use slice scores.
Frequency intervals are 95% Clopper-Pearson intervals conditional on the
fixed cohort, not uncertainty over new patients. Unsupported splits remain
in the total denominator and count as not low (M_eff < 2).
"""

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binomtest

from effective_size import effective_size
from scores import PRIMARY_TASKS, load_panel

TASKS = (*PRIMARY_TASKS, "isic2018", "msd_hippocampus")


def select_pool(panel, calibration_columns):
    """Return anchor, four family representatives and calibration-slice P33."""
    calibration = panel["dice"][:, calibration_columns]
    clusters = panel["cluster_ids"][calibration_columns]
    units = np.unique(clusters)
    if len(units) < 2:
        raise ValueError("calibration requires at least two units")
    unit_scores = np.stack([calibration[:, clusters == unit].mean(axis=1)
                            for unit in units], axis=1)
    means = unit_scores.mean(axis=1)
    bundles = panel["bundle"]
    functional = {bundle for bundle in np.unique(bundles)
                  if unit_scores[bundles == bundle].mean(axis=0).mean() >= 0.30}
    names = [f"{bundle}/seed_{seed}" for bundle, seed
             in zip(bundles, panel["seed"], strict=True)]
    ranked = sorted((i for i, bundle in enumerate(bundles) if bundle in functional),
                    key=lambda i: (-means[i], names[i]))
    winners = {}
    for i in ranked:
        winners.setdefault(panel["selection_family"][i], i)
    if len(winners) < 4:
        raise ValueError("calibration has fewer than four functional families")
    pool = list(winners.values())[:4]
    anchor = ranked[0]
    return anchor, pool, float(np.percentile(calibration[anchor], 33))


def selection_records(panel):
    """Replay study seeds 0..199 over sorted cluster IDs with NumPy PCG64."""
    units = np.unique(panel["cluster_ids"])
    if len(units) < 4:
        raise ValueError("half-splits require at least four units")
    names = [f"{bundle}/seed_{seed}" for bundle, seed
             in zip(panel["bundle"], panel["seed"], strict=True)]
    for seed in range(200):
        shuffled = units[np.random.default_rng(seed).permutation(len(units))]
        calibration = np.isin(panel["cluster_ids"], shuffled[:len(units) // 2])
        anchor, pool, threshold = select_pool(panel, calibration)
        selected = panel["dice"][pool][:, ~calibration]
        value = effective_size(selected, threshold)
        yield {"split_seed": seed, "anchor": names[anchor],
               "pool": [names[i] for i in pool], "p33": threshold,
               "M_eff": value, "support_valid": bool(np.isfinite(value)),
               "low_M_eff": bool(np.isfinite(value) and value < 2),
               "n_calibration_units": len(units) // 2,
               "n_evaluation_units": len(units) - len(units) // 2,
               "n_evaluation_cases": int((~calibration).sum())}


def exact_interval(successes, trials):
    if trials == 0:
        return np.nan, np.nan
    ci = binomtest(successes, trials).proportion_ci(0.95, method="exact")
    return float(ci.low), float(ci.high)


def summarize(task, records):
    total = len(records)
    supported = sum(r["support_valid"] for r in records)
    low = sum(r["low_M_eff"] for r in records)
    # Pooled task counts are descriptive: tasks do not share a binomial rate.
    ci = exact_interval(low, total) if task != "all_tasks" else (np.nan, np.nan)
    conditional_ci = (exact_interval(low, supported) if task != "all_tasks"
                      else (np.nan, np.nan))
    return {"task": task, "n_splits": total, "low_M_eff": low,
            "supported": supported, "unsupported": total - supported,
            "low_fraction_all": low / total,
            "ci95_low": ci[0], "ci95_high": ci[1],
            "low_fraction_supported": low / supported if supported else np.nan,
            "supported_ci95_low": conditional_ci[0],
            "supported_ci95_high": conditional_ci[1]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", help="directory containing the released score packs (default: data/ in this repository)")
    parser.add_argument("--output", help="CSV destination (default: stdout)")
    args = parser.parse_args(argv)
    rows, all_records = [], []
    for task in TASKS:
        group = "primary" if task in PRIMARY_TASKS else "heldout"
        panel = load_panel(f"{group}/{task}", args.data_dir)
        records = list(selection_records(panel))
        rows.append(summarize(task, records))
        all_records.extend(records)
    rows.append(summarize("all_tasks", all_records))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output = open(args.output, "w", newline="") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.output:
            output.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
