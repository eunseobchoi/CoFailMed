"""Four-distinct-bundle P33 effective sizes from released Dice traces.

Outputs point estimates for five frozen panels and four fine-tuning panels.
Each recipe uses its own pooled P33 after the bundle floor. These are means
of eligible pool-specific ratios, not a ratio computed from the mean rho.

Each row also gives the exact-covariance size M*tr(S)/(1'S1) (paper Eq. 3),
averaged over jointly eligible pools.
--bootstrap 2000 adds 95% percentile CIs. Floors and P33 thresholds stay fixed;
member support is rechecked per draw. Fig. 6a/Table E.1 use task-offset
seeds; Table G.4 uses shared seed-0 draws, including its frozen marginals.
ACDC/BraTS resample whole patients with slice multiplicity; other tasks
resample images. Intervals condition on the observed model panel.

--analysis fixed reproduces Table E.1(a): five frozen primary panels at
tau=0.85, with means of 4/(1+3*rho) (Eq. 2) and of the exact-covariance size
(Eq. 3), each over its own eligible pools.
Fixed results are point estimates only; --bootstrap is not accepted.
"""

import argparse
import csv
import itertools
from pathlib import Path
import sys

import numpy as np

from scores import PRIMARY_TASKS, correlation_matrix, floor_panel, load_panel, validate_panel


def _pool_summary(dice, threshold, pools, *, covariance=False, separate=False):
    """Optional Eq. 3 mean uses pools with both Eq. 2 and Eq. 3 defined.

    This joint denominator gate can exclude a covariance-defined pool when
    Eq. 2 is undefined; n_covariance_eligible counts the joint set, not all
    independently covariance-defined pools.
    Fixed analysis uses separate=True for the canonical independent gates.
    """
    dice = np.asarray(dice, dtype=float)
    if (dice.ndim != 2 or dice.shape[0] < 2 or dice.shape[1] < 2
            or not np.isfinite(dice).all() or np.any((dice < 0) | (dice > 1))):
        raise ValueError("Dice must be a finite [member, case] matrix in [0, 1]")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    failures = dice < threshold  # Ties pass; support is checked per member.
    counts = failures.sum(axis=1)
    supported = (counts >= 5) & (dice.shape[1] - counts >= 5)
    supported_pools = pools[np.all(supported[pools], axis=1)]
    m = pools.shape[1]
    left, right = np.triu_indices(m, 1)
    correlations = correlation_matrix(failures.astype(float))
    rhos = correlations[supported_pools[:, left], supported_pools[:, right]].mean(axis=1)
    denominator = 1 + (m - 1) * rhos
    eligible = np.isfinite(rhos) & np.isfinite(denominator) & (denominator > 1e-12)
    values = np.divide(m, denominator, out=np.full(rhos.shape, np.nan), where=eligible)
    eligible &= np.isfinite(values)
    result = {
        "n_pools": len(pools),
        "n_supported": len(supported_pools),
        "n_eligible": int(eligible.sum()),
        "rho_fail": float(rhos[eligible].mean()) if eligible.any() else np.nan,
        "M_eff": float(values[eligible].mean()) if eligible.any() else np.nan,
    }
    if covariance or separate:
        # Eq. 3: M trace(S)/sum(S); P33 retains its joint eligibility gate.
        centered = failures.astype(float) - failures.mean(axis=1, keepdims=True)
        scatter = centered @ centered.T
        trace = np.diag(scatter)[supported_pools].sum(axis=1)
        total = trace + 2 * scatter[supported_pools[:, left],
                                    supported_pools[:, right]].sum(axis=1)
        if separate:
            # Canonical compute_m_eff uses sums of squares to avoid cancellation.
            for start in range(0, len(supported_pools), 256):
                batch = supported_pools[start:start + 256]
                total[start:start + len(batch)] = np.square(
                    centered[batch].sum(axis=1)).sum(axis=1)
        vif = np.divide(total, trace, out=np.full(trace.shape, np.nan), where=trace > 0)
        covariance_eligible = np.isfinite(rhos) & np.isfinite(vif) & (vif > 1e-12)
        joint = eligible & covariance_eligible
        selected = covariance_eligible if separate else joint
        result.update(n_covariance_eligible=int(selected.sum()),
                      M_eff_covariance_exact=float((m / vif[selected]).mean())
                      if selected.any() else np.nan,
                      M_eff_same_pools=float(values[joint].mean())
                      if joint.any() else np.nan)
        if separate:
            rho_eligible = np.isfinite(rhos)
            result.update(n_joint_eligible=int(joint.sum()),
                          n_rho_eligible=int(rho_eligible.sum()),
                          rho_fail=float(rhos[rho_eligible].mean())
                          if rho_eligible.any() else np.nan)
    return result


def effective_size(dice, threshold):
    """Size of one pool; NaN if any member lacks support or the ratio is undefined.

    Requires >=5 failures AND >=5 passes per member and denominator >1e-12.
    Valid negative correlations are retained, so size may exceed member count.
    """
    dice = np.asarray(dice, dtype=float)
    if dice.ndim != 2:
        raise ValueError("Dice must be a [member, case] matrix")
    return _pool_summary(dice, threshold, np.arange(len(dice))[None, :])["M_eff"]


def enumerate_pools(panel):
    """Enumerate one seed head from each combination of four distinct bundles."""
    groups = [np.flatnonzero(panel["bundle"] == name)
              for name in np.unique(panel["bundle"])]
    return np.asarray([pool for group in itertools.combinations(groups, 4)
                       for pool in itertools.product(*group)], dtype=int).reshape(-1, 4)


def panel_summary(panel, *, covariance=False):
    """Panel must already have its recipe-specific bundle floor applied."""
    validate_panel(panel)
    threshold = float(np.percentile(panel["dice"], 33))
    return {
        "n_bundles": len(np.unique(panel["bundle"])),
        "n_heads": len(panel["dice"]),
        "n_cases": panel["dice"].shape[1],
        "pooled_P33": threshold,
        **_pool_summary(panel["dice"], threshold, enumerate_pools(panel),
                        covariance=covariance),
    }


def fixed_panel_summary(panel):
    """Table E.1(a), after the bundle floor; each ratio has its own eligibility.

    Uses the study's compute_m_eff_pools and compute_m_eff rules;
    source links and result mapping are in REPRODUCING.md.
    """
    validate_panel(panel)
    return {
        "n_bundles": len(np.unique(panel["bundle"])),
        "n_heads": len(panel["dice"]),
        "n_cases": panel["dice"].shape[1],
        "threshold": 0.85,
        **_pool_summary(panel["dice"], 0.85, enumerate_pools(panel), separate=True),
    }


def bootstrap_draws(panel, threshold, n_boot, seed, *, clustered=False):
    """Return aligned [rho, Eq. 2] draws; undefined draws remain NaN.

    Floor the panel once before calling; never recompute its threshold here.
    """
    if (isinstance(n_boot, (bool, np.bool_))
            or not isinstance(n_boot, (int, np.integer)) or n_boot < 0):
        raise ValueError("n_boot must be a nonnegative integer")
    pools = enumerate_pools(panel)
    _pool_summary(panel["dice"], threshold, pools)
    groups = None
    if clustered:
        labels = np.asarray(panel["cluster_ids"])
        if labels.shape != (panel["dice"].shape[1],) or len(np.unique(labels)) < 2:
            raise ValueError("cluster_ids must align with cases and include at least two patients")
        groups = [np.flatnonzero(labels == unit) for unit in np.unique(labels)]
    n = panel["dice"].shape[1]
    rng = np.random.default_rng(seed)
    draws = np.full((n_boot, 2), np.nan)
    for draw in range(n_boot):
        indices = (rng.integers(0, n, size=n) if groups is None else
                   np.concatenate([groups[i] for i in
                                   rng.integers(0, len(groups), size=len(groups))]))
        summary = _pool_summary(panel["dice"][:, indices], threshold, pools)
        draws[draw] = summary["rho_fail"], summary["M_eff"]
    return draws


def _interval(values, name):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    low, high = np.percentile(finite, [2.5, 97.5]) if len(finite) else (np.nan, np.nan)
    return {f"{name}_n_valid": len(finite),
            f"{name}_ci95_low": float(low), f"{name}_ci95_high": float(high)}


def _bootstrap_summary(draws, seed, clustered):
    return {"bootstrap": len(draws), "bootstrap_seed": seed,
            "bootstrap_unit": "patient" if clustered else "image",
            **_interval(draws[:, 0], "rho_fail"),
            **_interval(draws[:, 1], "M_eff")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", help="directory containing the released score packs (default: data/ in this repository)")
    parser.add_argument("--output", help="CSV destination (default: stdout)")
    parser.add_argument("--analysis", choices=("p33", "fixed"), default="p33",
                        help="p33: pooled-P33 effective sizes (default); fixed: tau=0.85, Table E.1(a)")
    parser.add_argument("--bootstrap", type=int, default=None, metavar="B",
                        help="bootstrap draws using the paper's fixed seeds (default: 0)")
    args = parser.parse_args(argv)
    if args.analysis == "fixed" and args.bootstrap is not None:
        parser.error("--bootstrap is not supported with --analysis fixed (point estimates only)")
    args.bootstrap = 0 if args.bootstrap is None else args.bootstrap
    if args.bootstrap < 0:
        parser.error("--bootstrap must be nonnegative")
    rows, frozen, frozen_draws = [], {}, {}
    for task_index, task in enumerate(PRIMARY_TASKS):
        panel = floor_panel(load_panel(f"primary/{task}", args.data_dir))
        if args.analysis == "fixed":
            rows.append({"task": task, "recipe": "frozen", **fixed_panel_summary(panel)})
            continue
        row = {"task": task, "recipe": "frozen",
               **panel_summary(panel, covariance=True),
               "M_eff_minus_frozen": 0.0}
        if args.bootstrap:
            seed = task_index * 100_000
            clustered = task in ("acdc_lv", "brats_wt")
            draws = bootstrap_draws(panel, row["pooled_P33"], args.bootstrap,
                                    seed, clustered=clustered)
            row.update(_bootstrap_summary(draws, seed, clustered))
            if task in ("kvasir", "riga_cup"):
                frozen_draws[task] = (draws if seed == 0 else bootstrap_draws(
                    panel, row["pooled_P33"], args.bootstrap, 0))
        frozen[task] = panel, row
        rows.append(row)
    for task in (("kvasir", "riga_cup") if args.analysis == "p33" else ()):
        reference, baseline = frozen[task]
        for rate in ("1e-4", "1e-5"):
            panel = floor_panel(load_panel(f"ft/lr{rate}/{task}", args.data_dir))
            if any(not np.array_equal(panel[key], reference[key])
                   for key in ("case_ids", "cluster_ids", "bundle", "seed")):
                raise ValueError(f"{task}/lr{rate}: frozen/FT cases or members differ")
            result = panel_summary(panel, covariance=True)
            row = {"task": task, "recipe": f"ft_lr{rate}", **result,
                   "M_eff_minus_frozen": result["M_eff"] - baseline["M_eff"]}
            if args.bootstrap:
                draws = bootstrap_draws(panel, row["pooled_P33"], args.bootstrap, 0)
                row.update(_bootstrap_summary(draws, 0, False))
                row.update(_interval(frozen_draws[task][:, 1], "paired_frozen_M_eff"))
                row.update(_interval(draws[:, 1] - frozen_draws[task][:, 1],
                                     "M_eff_minus_frozen"))
            rows.append(row)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output = open(args.output, "w", newline="") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(output, fieldnames=list(dict.fromkeys(
            key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.output:
            output.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
