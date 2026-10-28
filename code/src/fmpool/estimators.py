"""Correlation, filter, M_eff, and bootstrap estimators for FM-pool analysis.

Shared estimator definitions:

* ``pearson_rbar`` returns NaN (not 0.0) when either input has std < eps;
  such pairs are excluded from the within/cross averages.
* ``hierarchical_bootstrap`` uses ICC when ``stat="icc"`` is requested.
* ``compute_m_eff`` requires support for every member and every pair and
  separates the correlation approximation from exact covariance diagnostics.
* Hierarchical resampling is an occurrence-local finite-panel sensitivity,
  not a confidence interval for a population of model families.

References
----------
- McGraw & Wong (1996) "Forming inferences about some intraclass correlation
  coefficients." Psychological Methods 1(1):30-46.
- Kuncheva & Whitaker (2003) "Measures of diversity in classifier ensembles."
  Machine Learning 51(2):181-207.
- Davison & Hinkley (1997) "Bootstrap Methods and Their Application."
- Field & Welsh (2007) "Bootstrapping clustered data." JRSS-B 69(3):369-390.
"""
from __future__ import annotations

import logging
from typing import Callable, Iterable

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

FamilyKey = tuple[str, int]
PerCaseDice = dict[FamilyKey, np.ndarray]


# ---------------------------------------------------------------------------
# Per-case Dice
# ---------------------------------------------------------------------------


def dice_per_case(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-case Dice with the ``Dice(∅, ∅) = 1`` convention.

    ``pred`` and ``target`` are expected in shape ``[N, ...]`` where the leading
    axis indexes cases. Values must be binary (0/1) or boolean. Returns a 1-D
    array of length ``N`` with per-case Dice coefficients.

    The ``Dice(∅, ∅) = 1`` convention is applied *before* any empty-empty
    filter.
    """
    pred, target = np.asarray(pred), np.asarray(target)
    if any(not np.isin(a, (0, 1)).all() for a in (pred, target)):
        raise ValueError("Dice inputs must be finite binary masks")
    pred, target = pred.astype(bool), target.astype(bool)
    if pred.shape != target.shape:
        raise ValueError(
            f"pred/target shape mismatch: {pred.shape} vs {target.shape}"
        )
    if pred.ndim < 1:
        raise ValueError("expected leading case axis; got scalar input")

    n_cases = pred.shape[0]
    out = np.empty(n_cases, dtype=np.float64)
    flat_pred = pred.reshape(n_cases, -1)
    flat_target = target.reshape(n_cases, -1)
    for i in range(n_cases):
        p_sum = float(flat_pred[i].sum())
        t_sum = float(flat_target[i].sum())
        if p_sum == 0.0 and t_sum == 0.0:
            out[i] = 1.0
            continue
        inter = float(np.logical_and(flat_pred[i], flat_target[i]).sum())
        out[i] = (2.0 * inter) / (p_sum + t_sum)
    return out


# ---------------------------------------------------------------------------
# Correlation estimators
# ---------------------------------------------------------------------------


def pearson_rbar(x: np.ndarray | Iterable[float], y: np.ndarray | Iterable[float],
                 eps: float = 1e-8) -> float:
    """Pearson correlation with NaN on low-std inputs.

    Returns ``float('nan')`` when either input's standard deviation is below
    ``eps``. Callers filter out NaNs before averaging (NaN-exclude).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape != y_arr.shape:
        raise ValueError(f"shape mismatch: {x_arr.shape} vs {y_arr.shape}")
    if x_arr.size < 2:
        return float("nan")
    if np.std(x_arr) < eps or np.std(y_arr) < eps:
        return float("nan")
    r, _ = stats.pearsonr(x_arr, y_arr)
    return float(r)


def icc_a1(x: np.ndarray | Iterable[float], y: np.ndarray | Iterable[float]) -> float:
    """ICC(A,1) two-way absolute-agreement single-measurement.

    McGraw-Wong (1996) Case 2A, k=2 raters × n subjects. Returns NaN when the
    denominator is non-positive (very low between-subject variance).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    n = len(x_arr)
    if n < 2:
        return float("nan")
    subj_mean = (x_arr + y_arr) / 2.0
    grand = float(subj_mean.mean())
    rater_mean = np.array([x_arr.mean(), y_arr.mean()])
    ss_subject = 2.0 * float(np.sum((subj_mean - grand) ** 2))
    ss_rater = n * float(np.sum((rater_mean - grand) ** 2))
    ss_total = float(np.sum((x_arr - grand) ** 2) + np.sum((y_arr - grand) ** 2))
    ss_error = ss_total - ss_subject - ss_rater
    bms = ss_subject / max(n - 1, 1)
    rms = ss_rater  # df = k - 1 = 1
    ems = ss_error / max(n - 1, 1)
    denom = bms + ems + 2 * (rms - ems) / n
    if denom <= 0:
        return float("nan")
    return float((bms - ems) / denom)


# ---------------------------------------------------------------------------
# Pair-matrix and within/cross r-bar
# ---------------------------------------------------------------------------


def _stat_fn(stat: str) -> Callable[[np.ndarray, np.ndarray], float]:
    if stat == "pearson":
        return pearson_rbar
    if stat == "icc":
        return icc_a1
    raise ValueError(f"stat must be 'pearson' or 'icc', got {stat!r}")


def _pearson_matrix(
    per_case_dice: PerCaseDice,
    keys: list[FamilyKey],
    eps: float = 1e-8,
) -> np.ndarray:
    """Vectorized Pearson matrix with the same NaN convention as pearson_rbar."""
    n = len(keys)
    matrix = np.full((n, n), np.nan, dtype=np.float64)
    if n == 0:
        return matrix
    data = np.stack([np.asarray(per_case_dice[k], dtype=np.float64) for k in keys])
    if data.shape[1] < 2:
        return matrix
    valid = np.std(data, axis=1) >= eps
    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        return matrix
    corr = np.corrcoef(data[valid_idx])
    if corr.ndim == 0:
        corr = np.asarray([[float(corr)]], dtype=np.float64)
    matrix[np.ix_(valid_idx, valid_idx)] = corr
    return matrix


def family_pair_matrix(per_case_dice: PerCaseDice,
                       stat: str = "pearson") -> tuple[list[FamilyKey], np.ndarray]:
    """Pairwise correlation matrix over every (FM, seed) combination.

    Returns ``(keys, matrix)`` where ``keys`` is the sorted list of dict keys
    and ``matrix[i, j]`` is the pairwise statistic between ``keys[i]`` and
    ``keys[j]``.
    """
    keys = sorted(per_case_dice.keys())
    n = len(keys)
    if stat == "pearson":
        return keys, _pearson_matrix(per_case_dice, keys)
    fn = _stat_fn(stat)
    matrix = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            a = np.asarray(per_case_dice[keys[i]])
            b = np.asarray(per_case_dice[keys[j]])
            matrix[i, j] = matrix[j, i] = fn(a, b)
    return keys, matrix


def within_cross_rbar(per_case_dice: PerCaseDice,
                      family_pairs_within: set[frozenset[str]] | None = None,
                      stat: str = "pearson") -> dict[str, float | int | str]:
    """Symmetric multi-seed within- and cross-family r̄.

    Parameters
    ----------
    per_case_dice
        Mapping ``{(family, seed): 1-D array of per-case Dice}``.
    family_pairs_within
        Optional set of frozensets of family labels to count as within-family
        even if they appear as different family labels (for example,
        DINOv2-B × DINOv2-S counts as within-family).
    stat
        ``"pearson"`` or ``"icc"``.

    Returns
    -------
    dict
        Keys ``within``, ``cross``, ``gap``, ``n_within_pairs``,
        ``n_cross_pairs``, ``stat``. NaN-valued pairs are excluded from the
        mean (NaN-exclude convention).
    """
    if family_pairs_within is None:
        family_pairs_within = set()

    families = sorted({k[0] for k in per_case_dice})
    seeds_by_fam = {
        f: sorted({k[1] for k in per_case_dice if k[0] == f}) for f in families
    }

    within: list[float] = []
    cross: list[float] = []

    if stat == "pearson":
        keys, matrix = family_pair_matrix(per_case_dice, stat="pearson")
        for i, (fa, _sa) in enumerate(keys):
            for j in range(i + 1, len(keys)):
                fb, _sb = keys[j]
                value = float(matrix[i, j])
                is_within_pair = (
                    fa == fb or frozenset({fa, fb}) in family_pairs_within
                )
                (within if is_within_pair else cross).append(value)
    else:
        fn = _stat_fn(stat)

        # same-family, different-seed pairs
        for fam in families:
            seeds = seeds_by_fam[fam]
            for i, s1 in enumerate(seeds):
                for s2 in seeds[i + 1:]:
                    a = per_case_dice[(fam, s1)]
                    b = per_case_dice[(fam, s2)]
                    within.append(fn(np.asarray(a), np.asarray(b)))

        # different-family pairs (all seed × seed combinations)
        for i, fa in enumerate(families):
            for fb in families[i + 1:]:
                is_within_pair = frozenset({fa, fb}) in family_pairs_within
                bucket = within if is_within_pair else cross
                for sa in seeds_by_fam[fa]:
                    for sb in seeds_by_fam[fb]:
                        a = per_case_dice[(fa, sa)]
                        b = per_case_dice[(fb, sb)]
                        bucket.append(fn(np.asarray(a), np.asarray(b)))

    within_clean = [v for v in within if not np.isnan(v)]
    cross_clean = [v for v in cross if not np.isnan(v)]

    w_mean = float(np.mean(within_clean)) if within_clean else float("nan")
    c_mean = float(np.mean(cross_clean)) if cross_clean else float("nan")
    gap = w_mean - c_mean if (within_clean and cross_clean) else float("nan")

    return {
        "within": w_mean,
        "cross": c_mean,
        "gap": gap,
        "n_within_pairs": len(within_clean),
        "n_cross_pairs": len(cross_clean),
        "stat": stat,
    }


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def functional_floor_filter(per_case_dice: PerCaseDice,
                            min_mean_dice: float = 0.30,
                            scope: str = "per_fm") -> PerCaseDice:
    """Drop pool members whose mean Dice < ``min_mean_dice``.

    Parameters
    ----------
    per_case_dice
        Mapping ``{(fm, seed): 1-D array of per-case Dice}``.
    min_mean_dice
        Functional-floor threshold. The paper uses 0.30.
    scope
        ``"per_fm"`` (paper convention): drop an entire FM across all seeds
        when its cross-seed mean falls below the floor. ``"per_fm_seed"``:
        drop only the individual (FM, seed) pairs that fail.
    """
    typed = {k: np.asarray(v, dtype=np.float64) for k, v in per_case_dice.items()}

    if scope == "per_fm_seed":
        return {
            k: v for k, v in typed.items()
            if float(np.mean(v)) >= float(min_mean_dice)
        }
    if scope != "per_fm":
        raise ValueError(f"scope must be 'per_fm' or 'per_fm_seed', got {scope!r}")

    by_fm: dict[str, list[np.ndarray]] = {}
    for (fm, _seed), arr in typed.items():
        by_fm.setdefault(fm, []).append(arr)

    functional_fms: set[str] = set()
    for fm, seed_arrs in by_fm.items():
        fm_mean = float(np.mean([float(np.mean(a)) for a in seed_arrs]))
        if fm_mean >= float(min_mean_dice):
            functional_fms.add(fm)

    return {k: v for k, v in typed.items() if k[0] in functional_fms}


# ---------------------------------------------------------------------------
# M_eff
# ---------------------------------------------------------------------------


def compute_m_eff(per_case_dice: PerCaseDice, tau: float = 0.85) -> dict:
    """Finite-panel second-moment diagnostics, conditional on pool and tau.

    Every member needs >=5 failures AND >=5 passes; all M(M-1)/2 phi
    correlations must be defined. ``M_eff`` retains the historical correlation
    formula as an explicit approximation for unequal marginal variances (equal
    pair correlations are not required if marginal variances match). For S,
    the exact VIF is
    sum(S)/trace(S); M/VIF compares the pool mean with independent members of
    the observed average marginal variance. Neither is independent coverage.
    Raw and standardized mean variances have separate eligibility: zero in
    either gives NaN for its own size ratio with an explicit reason.
    """
    if not np.isfinite(tau):
        raise ValueError("tau must be finite")
    keys = sorted(per_case_dice)
    m = len(keys)
    data = np.stack([np.asarray(per_case_dice[k], dtype=float) for k in keys]) if keys else np.empty((0, 0))
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError("expected aligned finite one-dimensional score traces")
    fail = (data < tau).astype(float)
    counts = fail.sum(axis=1).astype(int)
    n = data.shape[1]
    unsupported = [f"{k[0]}:{k[1]}" for k, count in zip(keys, counts)
                   if count < 5 or n - count < 5]
    matrix = _pearson_matrix(dict(zip(keys, fail)), keys)
    pairs = matrix[np.triu_indices(m, k=1)]
    n_valid = int(np.isfinite(pairs).sum())
    support_ok = m >= 2 and not unsupported and n_valid == len(pairs)
    out = {
        "rho_fail": float("nan"),
        "M_eff": float("nan"),
        "M_eff_equicorrelation_approx": float("nan"),
        "M_eff_covariance_exact": float("nan"),
        "variance_inflation_exact": float("nan"),
        "n_fm_seed": m,
        "n_cases": n,
        "n_expected_pairs": m * (m - 1) // 2,
        "n_valid_pairs": n_valid,
        "min_failures_and_passes": 5,
        "member_event_counts": {f"{k[0]}:{k[1]}": {"failures": int(c), "passes": int(n - c)}
                                for k, c in zip(keys, counts)},
        "unsupported_members": unsupported,
        "support_ok": bool(support_ok),
        "eligible": False,
        "covariance_eligible": False,
        "ineligibility_reason": "insufficient_member_or_pair_support",
        "covariance_ineligibility_reason": "insufficient_member_or_pair_support",
        "diagnostic_target": "finite-panel variance of the mean failure indicator",
        "M_eff_alias": "M_eff_equicorrelation_approx",
        "conditioning": "observed pool and fixed tau; no independent-coverage interpretation",
    }
    if not support_ok:
        return out
    rho = float(np.mean(pairs))
    denom = 1.0 + (m - 1) * rho
    centered = fail - fail.mean(axis=1, keepdims=True)
    # Sum of squares avoids cancellation in sum(S) for complementary members.
    vif = float(np.sum(centered.sum(axis=0) ** 2) / np.sum(centered ** 2))
    out.update(rho_fail=rho, variance_inflation_exact=vif)
    if denom > 1e-12:
        out.update(M_eff=m / denom, M_eff_equicorrelation_approx=m / denom,
                   eligible=True, ineligibility_reason=None)
    else:
        out["ineligibility_reason"] = "zero_or_numerically_singular_standardized_mean_variance"
    if vif > 1e-12:
        out.update(M_eff_covariance_exact=m / vif, covariance_eligible=True,
                   covariance_ineligibility_reason=None)
    else:
        out["covariance_ineligibility_reason"] = "zero_or_numerically_singular_mean_variance"
    return out


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------


def _ci95(arr: list[float]) -> tuple[float, float]:
    if not arr:
        return (float("nan"), float("nan"))
    a = np.asarray(arr)
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))


def _bootstrap_case_indices(n_cases: int, n_boot: int, seed: int | np.random.SeedSequence = 0,
                            case_to_subject: Iterable[str] | None = None) -> Iterable[np.ndarray]:
    """Shared aligned draws: cases, or whole subjects with multiplicity."""
    if n_cases < 2 or n_boot < 0:
        raise ValueError("need at least two cases and nonnegative n_boot")
    groups = None
    if case_to_subject is not None:
        labels = np.asarray(list(case_to_subject))
        if labels.ndim != 1 or len(labels) != n_cases:
            raise ValueError("case_to_subject length mismatch")
        groups = [np.flatnonzero(labels == s) for s in sorted(set(labels.tolist()))]
        if len(groups) < 2:
            raise ValueError("need at least two independent subjects")
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        if groups is None:
            yield rng.integers(0, n_cases, n_cases)
        else:
            yield np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])


def case_bootstrap(per_case_dice: PerCaseDice,
                   family_pairs_within: set[frozenset[str]] | None = None,
                   n_boot: int = 2000,
                   seed: int = 0,
                   stat: str = "pearson") -> dict:
    """Case-level bootstrap — resample test-case indices with replacement.

    Returns the point estimate from :func:`within_cross_rbar` and 95% percentile
    CIs for ``within``, ``cross``, ``gap``.
    """
    keys = list(per_case_dice.keys())
    n = len(next(iter(per_case_dice.values())))
    boots: dict[str, list[float]] = {"within": [], "cross": [], "gap": []}

    for idx in _bootstrap_case_indices(n, n_boot, seed):
        sampled = {k: np.asarray(per_case_dice[k])[idx] for k in keys}
        r = within_cross_rbar(sampled, family_pairs_within=family_pairs_within, stat=stat)
        for key in boots:
            val = r[key]
            if isinstance(val, (int, float)) and not np.isnan(val):
                boots[key].append(float(val))

    point = within_cross_rbar(per_case_dice,
                              family_pairs_within=family_pairs_within,
                              stat=stat)
    return {
        "point": point,
        "ci95": {k: _ci95(v) for k, v in boots.items()},
        "n_boot": int(n_boot),
        "n_valid_boot": {k: len(v) for k, v in boots.items()},
        "stat": stat,
    }


def subject_cluster_bootstrap(per_case_dice: PerCaseDice,
                              case_to_subject: Iterable[str],
                              family_pairs_within: set[frozenset[str]] | None = None,
                              n_boot: int = 2000,
                              seed: int = 0,
                              stat: str = "pearson") -> dict:
    """Cluster bootstrap over higher-level case units.

    Resamples subjects/patients with replacement, then takes all 2-D cases
    belonging to each sampled subject. Use this when the released test unit is a
    slice but inference should respect patient/volume clustering (ACDC, BraTS).
    """
    case_to_subject_arr = np.asarray(list(case_to_subject))
    n_cases = len(next(iter(per_case_dice.values())))
    if len(case_to_subject_arr) != n_cases:
        raise ValueError(
            "case_to_subject length mismatch: "
            f"{len(case_to_subject_arr)} vs {n_cases}"
        )

    unique_subjects = np.array(sorted(set(case_to_subject_arr.tolist())))
    boots: dict[str, list[float]] = {"within": [], "cross": [], "gap": []}

    for case_idx in _bootstrap_case_indices(n_cases, n_boot, seed, case_to_subject_arr):
        sampled = {k: np.asarray(v)[case_idx] for k, v in per_case_dice.items()}
        r = within_cross_rbar(
            sampled, family_pairs_within=family_pairs_within, stat=stat
        )
        for key in boots:
            val = r[key]
            if isinstance(val, (int, float)) and not np.isnan(val):
                boots[key].append(float(val))

    point = within_cross_rbar(
        per_case_dice, family_pairs_within=family_pairs_within, stat=stat
    )
    return {
        "point": point,
        "ci95": {k: _ci95(v) for k, v in boots.items()},
        "n_boot": int(n_boot),
        "stat": stat,
        "n_subjects": int(len(unique_subjects)),
        "n_valid_boot": {k: len(v) for k, v in boots.items()},
    }


def _within_cross_occurrences(occurrences: list[tuple[str, list[np.ndarray]]],
                              stat: str = "pearson") -> dict:
    """Within pairs belong to one draw; cross pairs need different families.

    Distinct outer draws of the same family are neither within nor cross pairs.
    Duplicate inner seed draws remain separate observations in their occurrence.
    """
    traces = {(str(i), j): arr for i, (_, arrays) in enumerate(occurrences)
              for j, arr in enumerate(arrays)}
    keys, matrix = family_pair_matrix(traces, stat=stat)
    within, cross = [], []
    excluded = 0
    for i, (a, _) in enumerate(keys):
        for j in range(i + 1, len(keys)):
            b = keys[j][0]
            if a == b:
                within.append(float(matrix[i, j]))
            elif occurrences[int(a)][0] != occurrences[int(b)][0]:
                cross.append(float(matrix[i, j]))
            else:
                excluded += 1
    w = [v for v in within if np.isfinite(v)]
    c = [v for v in cross if np.isfinite(v)]
    wm = float(np.mean(w)) if w else float("nan")
    cm = float(np.mean(c)) if c else float("nan")
    return {"within": wm, "cross": cm, "gap": wm - cm, "stat": stat,
            "n_within_pairs": len(w), "n_cross_pairs": len(c),
            "n_undefined_within_pairs": len(within) - len(w),
            "n_undefined_cross_pairs": len(cross) - len(c),
            "n_excluded_same_family_between_occurrence_pairs": excluded,
            "target": "occurrence-local finite-panel sensitivity"}


def hierarchical_bootstrap(per_case_dice: PerCaseDice,
                           families: list[str] | None = None,
                           n_boot: int = 2000,
                           seed: int = 0,
                           stat: str = "pearson",
                           case_to_subject: Iterable[str] | None = None) -> dict:
    """Occurrence-local sensitivity over the observed families and seeds.

    Case/subject draws are shared across all occurrences. These percentiles
    describe empirical panel resampling, not representative family-population
    inference or a conservative confidence interval. Invalid gaps are counted.
    """
    case_seed, panel_seed = np.random.SeedSequence(seed).spawn(2)
    rng = np.random.default_rng(panel_seed)
    if families is None:
        families = sorted({k[0] for k in per_case_dice})
    else:
        families = list(families)
    if (not families or len(set(families)) != len(families)
            or not set(families) <= {k[0] for k in per_case_dice}):
        raise ValueError("families must be a nonempty unique subset of observed families")

    seeds_by_fam = {
        f: sorted({k[1] for k in per_case_dice if k[0] == f}) for f in families
    }
    n = len(next(iter(per_case_dice.values())))

    boots: dict[str, list[float]] = {"within": [], "cross": [], "gap": []}
    for case_idx in _bootstrap_case_indices(n, n_boot, case_seed, case_to_subject):
        sampled_fams = rng.choice(families, size=len(families), replace=True)
        occurrences = []
        for fam in sampled_fams:
            fam_seeds = seeds_by_fam[fam]
            sampled_seeds = rng.choice(fam_seeds, size=len(fam_seeds), replace=True)
            occurrences.append((str(fam), [np.asarray(per_case_dice[(fam, int(s))])[case_idx]
                                           for s in sampled_seeds]))

        r = _within_cross_occurrences(occurrences, stat=stat)
        if not np.isnan(r["within"]) and not np.isnan(r["cross"]):
            w = float(r["within"])
            c = float(r["cross"])
            boots["within"].append(w)
            boots["cross"].append(c)
            boots["gap"].append(w - c)

    point = _within_cross_occurrences(
        [(f, [np.asarray(per_case_dice[(f, s)]) for s in seeds_by_fam[f]]) for f in families], stat=stat)
    return {
        "point": point,
        "ci95": {k: _ci95(v) for k, v in boots.items()},
        "n_boot": int(n_boot),
        "stat": stat,
        "n_valid_boot": {k: len(v) for k, v in boots.items()},
        "target": "occurrence-local finite-panel sensitivity",
        "interval_kind": "descriptive resampling percentiles",
        "case_resampling_unit": "subject" if case_to_subject is not None else "case",
        "seed_resampling": "with replacement; duplicate seed positions retain self-correlations",
        "conditioning": "observed family/seed panel and draws with defined within and cross means",
    }
