#!/usr/bin/env python3
"""Quality-controlled pair regression with patient-cluster confidence intervals.
Resample patients, retaining all slices and multiplicity; refit pair-quality
OLS each draw. This is NOT patient-mean regression or the case-bootstrap CI.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import compute_quality_controlled_pairs as qc
import compute_cross_checkpoint_family as ccf
from fmpool.estimators import _bootstrap_case_indices, functional_floor_filter

def corr(X, eps=1e-8):
    Xc = X - X.mean(axis=1, keepdims=True)
    ss = np.sqrt((Xc * Xc).sum(axis=1))
    sd = np.sqrt((Xc * Xc).mean(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        C = (Xc @ Xc.T) / np.outer(ss, ss)
    bad = sd < eps
    C[bad, :] = np.nan
    C[:, bad] = np.nan
    return C

def pair_design(X, labels, pairs):
    iu, ju, same = pairs
    mu, var = X.mean(1), X.var(1)
    sd = np.sqrt(var)
    C = corr(X, eps=0.0)
    r = C[iu, ju]
    ok = (sd[iu] >= 1e-8) & (sd[ju] >= 1e-8) & np.isfinite(r)
    D = np.column_stack([np.ones(len(iu)), same.astype(float), (mu[iu] + mu[ju]) / 2, np.abs(mu[iu] - mu[ju]),
                         (var[iu] + var[ju]) / 2, np.abs(var[iu] - var[ju])])[ok]
    return r[ok], D, same[ok]

def ols(y, D):
    beta = np.linalg.lstsq(D, y, rcond=None)[0]
    res = y - D @ beta
    sst = np.sum((y - y.mean()) ** 2)
    return beta, 1 - np.sum(res ** 2) / sst

def summarize(results_root, n_boot=1000):
    out = {}
    for ti, task in enumerate(qc.ca.TASKS):
        if task not in ("acdc_lv", "brats_wt"):
            continue
        traces, _ = qc.ca._collect_task_jsons(results_root, task)
        qc.ca._validate_alignment(traces)
        per_case = functional_floor_filter(qc.ca._as_per_case_map(traces), qc.ca.FUNCTIONAL_FLOOR)
        keys, matrix = ccf._keys_matrix(per_case)
        labels = [key[0] for key in keys]
        i, j = np.triu_indices(len(keys), 1)
        pairs = i, j, np.array([qc._is_same_group(labels[a], labels[b]) for a, b in zip(i, j)])
        _, units = ccf._case_units(task, traces[0].test_ids)
        draws = [ols(*pair_design(matrix[:, idx], labels, pairs)[:2])[0][1]
                 for idx in _bootstrap_case_indices(matrix.shape[1], n_boot, ti * 1009, units)]
        out[task] = {"beta_same_patient_cluster_ci95": qc._ci95(draws), "n_patients": len(set(units))}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    qc._write_json(args.out, summarize(args.results_root))
    print(args.out)


if __name__ == '__main__':
    main()
