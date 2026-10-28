"""Distinct-checkpoint estimates and paired patient-cluster intervals."""
from __future__ import annotations
import itertools, json
from pathlib import Path
import numpy as np
import compute_cross_checkpoint_family as ccf
from fmpool.estimators import _bootstrap_case_indices, functional_floor_filter

CLUSTER_TASKS = ("acdc_lv", "brats_wt")
TASKS = ("kvasir", "acdc_lv", "brats_wt", "riga_cup", "riga_disc")


def pair_rbars(root: Path, task: str, families: dict[str, list[str]]) -> dict[str, dict]:
    out = {}
    for fms in families.values():
        for a, b in itertools.combinations(sorted(fms), 2):
            ra = [json.loads(p.read_text()) for p in sorted((root / task / a).glob("seed_*.json"))]
            rb = [json.loads(p.read_text()) for p in sorted((root / task / b).glob("seed_*.json"))]
            assert all(x["test_ids"] == ra[0]["test_ids"] for x in ra + rb), (task, a, b)
            rs = [np.corrcoef(x["per_case_dice"], y["per_case_dice"])[0, 1] for x in ra for y in rb]
            out[f"{a}--{b}"] = {"rbar": float(np.mean(rs)), "n_member_pairs": len(rs)}
    return out


def bundle_means(root: Path, task: str) -> dict[str, float]:
    return {d.name: float(np.mean([np.mean(json.loads(p.read_text())["per_case_dice"])
                                   for p in sorted(d.glob("seed_*.json"))]))
            for d in sorted((root / task).iterdir())}


def cluster_gap(root: Path, task: str, expected_point: float) -> dict:
    """Patient-cluster bootstrap CI of the xcheckpoint-minus-cross gap (diagnostic strata and floor)."""
    traces = ccf._collect(root, task)
    per_case = functional_floor_filter({(t.fm, t.seed): t.per_case_dice for t in traces},
                                       min_mean_dice=ccf.FUNCTIONAL_FLOOR, scope="per_fm")
    keys, matrix = ccf._keys_matrix(per_case)
    pairs, ex = ccf._bucket_pairs(keys)
    unit, ids = ccf._case_units(task, traces[0].test_ids)
    gap = lambda m: ccf._stat_from_matrix(keys, m, pairs, ex)["gaps"]["xcheckpoint_minus_cross"]
    assert abs(gap(matrix) - expected_point) < 1e-12, (root, task)
    draws = np.asarray([gap(matrix[:, idx]) for idx in
                        _bootstrap_case_indices(matrix.shape[1], ccf.BOOTSTRAP_B, ccf.BOOTSTRAP_SEED, ids)])
    return {"ci95": ccf._ci(draws), "unit": unit, "n_units": len(set(ids)),
            "n_valid_boot": int(np.isfinite(draws).sum())}


def summarize_row(pool_root: Path, primary_root: Path, t: str, r: dict, primary_diag: dict, gate=None) -> dict:
    """Summarize checkpoint correlations with explicit trace roots and diagnostics."""
    traces = pool_root / "per_case_dice"
    p, ci = r["point"], r["case_bootstrap_ci95"]
    means = bundle_means(traces, t)
    fams = {k: v for k, v in r["family_groups_after_floor"].items() if len(v) > 1}
    sub = r["subject_aggregated"]
    pairs = pair_rbars(traces, t, fams)
    # self-check: equal-weight pair mean must reproduce the diagnostic stratum
    assert abs(np.mean([v["rbar"] for v in pairs.values()]) - p["same_family_cross_checkpoint"]["rbar"]) < 1e-9, t
    gap, gap11 = p["gaps"]["xcheckpoint_minus_cross"], primary_diag["point"]["gaps"]["xcheckpoint_minus_cross"]
    cl = cluster_gap(pool_root, t, gap) if t in CLUSTER_TASKS else None
    cl11 = cluster_gap(primary_root, t, gap11) if t in CLUSTER_TASKS else None
    case_ci, case_ci11 = ci["xcheckpoint_minus_cross"], primary_diag["case_bootstrap_ci95"]["xcheckpoint_minus_cross"]
    return {
        "n_bundles_before_floor": len(means), "K_after_floor": len(r["fms_kept_after_floor"]),
        "dropped_by_floor": {fm: m for fm, m in means.items() if fm not in r["fms_kept_after_floor"]},
        "same_bundle_rbar": p["same_fm_seed_floor"]["rbar"], "same_bundle_n_pairs": p["same_fm_seed_floor"]["n_pairs"],
        "same_family_ckpt_rbar": p["same_family_cross_checkpoint"]["rbar"],
        "same_family_ckpt_n_pairs": p["same_family_cross_checkpoint"]["n_pairs"],
        "same_family_ckpt_pairs": pairs,
        "cross_family_rbar": p["cross_family"]["rbar"], "cross_family_n_pairs": p["cross_family"]["n_pairs"],
        "ckpt_gap": gap, "ckpt_gap_ci95": cl["ci95"] if cl else case_ci,
        "ckpt_gap_ci95_kind": f"cluster ({cl['unit']}, n={cl['n_units']})" if cl else "case",
        "gap_ci_case": case_ci, "gap_ci_cluster": cl["ci95"] if cl else None, "gap_cluster_detail": cl,
        "subject_level_ckpt_gap": None if sub is None else {
            "unit": sub["cluster_unit"], "n_units": sub["n_units"],
            "gap": sub["point"]["gaps"]["xcheckpoint_minus_cross"],
            "ci95": sub["case_bootstrap_ci95"]["xcheckpoint_minus_cross"]},
        "primary11_ckpt_gap": gap11, "primary11_ckpt_gap_ci95": cl11["ci95"] if cl11 else case_ci11,
        "primary11_gap_ci_case": case_ci11, "primary11_gap_ci_cluster": cl11["ci95"] if cl11 else None,
        "l1b_gate": gate,
    }
