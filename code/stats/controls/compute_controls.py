
"""Scalar control statistics; invoke through code/stats/recompute_controls.py."""
from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import sys
import time

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path[:0] = [str(OUT), str(ROOT / "code/src")]
import numpy as np
from analysis import effective_size as e1, failure_comparison as e4
from analysis import failure_metrics as l1, decoder_pairs as unet, trace_validation

B = 2000
FLOOR = 0.30
TOL = 1e-12
FROZEN = ROOT / "results/primary"
FT = ROOT / "results/finetuning"
SOURCE_PATHS = {}
INPUTS = {}
META = {}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    path = Path(path)
    INPUTS[str(path)] = digest(path)
    return json.loads(path.read_text())


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def progress(stage):
    print(time.strftime("%Y-%m-%dT%H:%M:%S%z"), stage, flush=True)
    write("runtime.json", {"pid": os.getpid(), "status": "running", "stage": stage,
                           "started_unix": START, "elapsed_s": time.time() - START,
                           "command": " ".join([sys.executable, *sys.argv])})


def same(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (a.keys(), b.keys())
        for key in a:
            same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for aa, bb in zip(a, b):
            same(aa, bb)
    elif isinstance(a, (float, int)) and not isinstance(a, bool):
        np.testing.assert_allclose(a, b, rtol=0, atol=TOL)
    else:
        assert a == b, (a, b)


def load_collection(paths, task, label, *, nnunet=False):
    """Validate explicit IDs before using E1's actual trace loader."""
    means, summaries, nnrows = {}, [], []
    paths = sorted(paths)
    assert paths, label
    for path in paths:
        d = read(path)
        ids = d.get("test_ids")
        values, eids = e1._trace_values(d, path)
        assert isinstance(ids, list) and len(ids) == len(set(ids)) == len(values), path
        assert d["task"] == task and np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        assert int(d["n_test"]) == len(values)
        assert abs(float(values.mean()) - d["mean_dice"]) < 1e-6, path
        identity = d.get("run_identity_v2", {})
        member = f"fold{d['fold']}_seed{d['seed']}" if nnunet else f"{d['fm']}/seed_{d['seed']}"
        bundle = f"fold{d['fold']}" if nnunet else d["fm"]
        means.setdefault(bundle, []).append(float(values.mean()))
        summaries.append({"path": str(path), "member": member, "bundle": bundle, "mean_dice": float(values.mean()),
                          "seed": d["seed"], "recipe": identity.get("recipe"),
                          "precision": identity.get("config", {}).get("precision"), "tf32": identity.get("tf32"),
                          "input_identity_keys": sorted((identity.get("input_identity") or {}).keys()),
                          "head_or_checkpoint_sha256": d.get("head_checkpoint_sha256", d.get("checkpoint_sha256")),
                          "dice_convention": d.get("dice_convention"), "epochs": d.get("epochs"),
                          "encoder_lr": d.get("encoder_lr"), "all_zero": bool(np.all(values == 0)),
                          "constant": bool(np.ptp(values) == 0), "nonzero_cases": int((values != 0).sum())})
        if nnunet:
            nnrows.append((member, bundle, eids, values))
    bundle_means = {k: float(np.mean(v)) for k, v in means.items()}
    kept = {k for k, v in bundle_means.items() if v >= FLOOR}
    if nnunet:
        assert kept == set(means), "fixed nnU-Net members fail floor"
        ids = nnrows[0][2]
        matrix = []
        for _, _, ids_i, vals in nnrows:
            assert set(ids_i) == set(ids)
            lookup = dict(zip(ids_i, vals, strict=True))
            matrix.append([lookup[i] for i in ids])
        c = e1.TraceCollection(label, task, tuple(r[0] for r in nnrows), tuple(r[1] for r in nnrows),
                               np.asarray(matrix), ids, None, tuple(map(str, paths)), "saved case-identical 100-epoch scores")
    else:
        c = e1._load_trace_paths([p for p in paths if p.parent.name in kept], config=label, task=task,
                                 evidence_scope="complete saved-score comparison")
        # Independently check the floor's existing Trace-based implementation.
        traces = [trace_validation.load_trace(p, ROOT) for p in paths]
        assert {t.bundle for t in trace_validation.apply_bundle_floor(traces, FLOOR)} == kept
    assert len(c.member_ids) == len(set(c.member_ids))
    META[label] = {"n_cases": len(c.case_ids), "n_members": len(c.member_ids), "K_bundles": len(kept),
                   "members": list(c.member_ids), "case_order": list(c.case_ids),
                   "floor": FLOOR, "bundle_means_before_floor": bundle_means,
                   "kept_bundles": sorted(kept), "excluded_bundles": sorted(set(means) - kept),
                   "trace_metadata": summaries, "cluster_ids": None}
    return c


def align_to(c, ids):
    assert set(c.case_ids) == set(ids), (c.config, "case-set mismatch")
    lookup = {v: i for i, v in enumerate(c.case_ids)}
    return e1.TraceCollection(**{**c.__dict__, "case_ids": ids, "matrix": c.matrix[:, [lookup[i] for i in ids]]})


# The failure-comparison module defaults to excluding undefined kappa cells; this scorer
# swaps in the official convention for its matrix function only. Formulas, stratum
# handling and the bootstrap loop are unchanged.
def official_matrices(success, kappa_never_fail="official"):
    return l1._metric_matrices(success, kappa_never_fail="official")


def checks():
    values = np.asarray([[1, 1, 1, 1], [0, 0, 0, 0], [1, 0, 1, 0], [1, 1, 0, 0]], bool)
    matrices = official_matrices(values)
    scalar = {"kappa": l1.cohen_kappa, "q_statistic": l1.q_statistic,
              "disagreement_rate": l1.disagreement_rate, "double_fault_rate": l1.double_fault_rate}
    for metric, fn in scalar.items():
        for i, j in itertools.product(range(4), repeat=2):
            ref = fn(values[i], values[j])
            if ref is None:
                assert np.isnan(matrices[metric][i, j])
            else:
                np.testing.assert_allclose(matrices[metric][i, j], ref, rtol=0, atol=TOL)
    assert matrices["kappa"][0, 0] == 1 and np.isnan(l1._metric_matrices(values, "exclude")["kappa"][0, 0])
    keys = [("dinov2_vitb14", 42), ("dinov2_vits14", 42), ("other", 42), ("other", 43)]
    wp, cp = unet._pair_masks(keys)
    assert set(map(tuple, wp)) == {(0, 1), (2, 3)} and len(cp) == 4
    nw, nc = e4.nnunet_pair_masks(("fold0_seed13", "fold0_seed37", "fold1_seed13", "fold1_seed37"))
    assert set(map(tuple, nw)) == {(0, 1), (2, 3)} and len(nc) == 4
    toy = np.random.default_rng(9).uniform(size=(5, 40))
    spec = e1._pool_spec_diverse4(("a", "a", "b", "c", "d"))
    assert len(spec.pool_members) == 2
    for pool in spec.pool_members:
        assert len({("a", "a", "b", "c", "d")[i] for i in pool}) == 4
    t = 0.4
    point = e1.pool_summary(toy, t, spec)
    rr = [np.corrcoef((toy[pool] < t).astype(float))[np.triu_indices(4, 1)].mean() for pool in spec.pool_members]
    np.testing.assert_allclose(point["M_eff"], np.mean([4 / (1 + 3 * r) for r in rr]), rtol=0, atol=TOL)
    draws, _, _ = e1_draws(toy, t, spec, seed=0, b=31, label="toy", announce=False)
    reference = e1.bootstrap_pool_rows(toy, [t], spec, b=31, seed=0, cluster_ids=None)[t]
    same(intervals(draws, 31), reference)
    assert e4._support(np.ones((2, 10)), 0.5)["status"] != "support-valid"
    write("adapter_checks.json", {"status": "PASS", "scalar_metric_cells": 64, "grouping": "UNet same-bundle plus B/S; nnU-Net same-fold",
                                  "E1": "two toy pools and 31 bootstrap draws match existing bootstrap_pool_rows exactly",
                                  "official_kappa_never_fail": 1, "source_files_changed": False})


def e1_draws(matrix, threshold, spec, *, seed, b=B, label, announce=True):
    rng = np.random.default_rng(seed)
    draws = np.full((b, 2), np.nan)
    supported = np.zeros(b, dtype=int)
    finite = np.zeros(b, dtype=int)
    for i in range(b):
        sample = matrix[:, e1._resample_indices(rng, matrix.shape[1], None)]
        row = e1.pool_summary(sample, threshold, spec)
        supported[i] = row["n_pools_supported"]
        finite[i] = row.get("n_pools_finite", 0)
        if row["status"] == "support-valid":
            draws[i] = row["rho_fail"], row["M_eff"]
        if announce and (i + 1) % 500 == 0:
            progress(f"{label}: {i + 1}/{b} paired case draws")
    return draws, supported, finite


def intervals(draws, b=B):
    rho, meff = (d[np.isfinite(d)] for d in draws.T)
    return {"B_requested": b, "B_finite": min(len(rho), len(meff)),
            "rho_fail_ci95": np.percentile(rho, [2.5, 97.5]).tolist() if len(rho) else None,
            "M_eff_ci95": np.percentile(meff, [2.5, 97.5]).tolist() if len(meff) else None}


def effective(c, seed, label):
    progress(label)
    threshold = next(t for t in e1.threshold_union(c.matrix) if "P33" in t["sources"])
    spec = e1._pool_spec_diverse4(c.bundle_ids)
    point = e1.pool_summary(c.matrix, threshold["value"], spec)
    draws, supported, finite = e1_draws(c.matrix, threshold["value"], spec, seed=seed, label=label)
    boot = intervals(draws)
    record = e1._threshold_record(threshold, point, boot)
    result = {"label": label, "collection": c.config, "task": c.task, "nominal_M": 4,
              "bootstrap_seed": seed, "n_cases": len(c.case_ids), "n_members": len(c.member_ids),
              "K_bundles": len(set(c.bundle_ids)), "primary_P33": record,
              "support_over_draws": {"minimum_supported_pools": int(supported.min()), "maximum_supported_pools": int(supported.max()),
                                     "draws_with_any_unsupported_pool": int((supported < len(spec.pool_members)).sum()),
                                     "draws_with_zero_supported_pools": int((supported == 0).sum()),
                                     "draws_with_nonfinite_supported_pool": int((finite < supported).sum())},
              "observed_threshold_ties": int((c.matrix == threshold["value"]).sum()),
              "draw_file": label + "_draws.npz"}
    np.savez(OUT / result["draw_file"], rho_fail=draws[:, 0], M_eff=draws[:, 1], supported_pools=supported, finite_pools=finite)
    write(label + ".json", result)
    return result, draws


def alternatives(c, sanity, nnunet=False):
    progress(c.config)
    if nnunet:
        wp, cp = e4.nnunet_pair_masks(c.member_ids)
    else:
        keys = [(bundle, int(member.rsplit("_", 1)[1])) for bundle, member in zip(c.bundle_ids, c.member_ids)]
        wp, cp = unet._pair_masks(keys)
        ii, jj, mask = l1._same_mask(keys)
        assert set(map(tuple, wp)) == set(zip(ii[mask], jj[mask]))
    floor = e4._floor_gate(c)
    threshold = next(t for t in e1.threshold_union(c.matrix) if "P33" in t["sources"])
    support = e4._support(c.matrix, threshold["value"])
    assert support["status"] == "support-valid", support
    pearson = e4._pearson_point(c.matrix, wp, cp)
    same(pearson["gap_same_minus_cross"], sanity["diagnostic_points_even_if_unsupported"]["pearson"]["gap_same_minus_cross"])
    same(threshold["value"], sanity["pooled_p33"])
    assert len(wp) == sanity["n_same_pairs"] and len(cp) == sanity["n_cross_pairs"]
    original = e4.metric_points(c.matrix >= threshold["value"], wp, cp)
    e4._metric_matrices = official_matrices
    point = e4.metric_points(c.matrix >= threshold["value"], wp, cp)
    same(original, point)  # observed support-valid points must not drift
    bootstrap = e4.bootstrap_metrics(c.matrix, threshold["value"], wp, cp, b=B, seed=0)
    rows = {}
    for metric, p in point.items():
        coverage = not (p["n_undefined_same_pairs"] or p["n_undefined_cross_pairs"])
        rows[metric] = {"point": p, "case_bootstrap_95ci": bootstrap[metric],
                        "interpretation": e4.classify_order(metric, p["gap_same_minus_cross"], pearson["gap_same_minus_cross"],
                                                               bootstrap[metric]["gap_same_minus_cross"]["ci95"], complete_pair_coverage=coverage)}
    result = {"label": c.config, "n_cases": len(c.case_ids), "n_members": len(c.member_ids), "threshold": threshold,
              "functional_floor": floor, "support": support, "pearson_sanity": pearson,
              "pairing": {"same": wp.tolist(), "cross": cp.tolist()}, "bootstrap_B": B, "bootstrap_seed": 0,
              "kappa_never_fail": "official", "estimators": rows,
              "threshold_ties": int((c.matrix == threshold["value"]).sum())}
    write(c.config + ".json", result)
    return result


def main():
    for module in (e1, e4, l1, unet, trace_validation):
        INPUTS[module.__file__] = digest(Path(module.__file__))
    INPUTS[str(Path(__file__).resolve())] = digest(Path(__file__).resolve())
    progress("validate small numerical adapters")
    checks()
    points = read(ROOT / "results/controls/metric_points.json")
    e4_rows = {}
    for label, task in (("UNet-skip Cup", "riga_cup"), ("UNet-skip Disc", "riga_disc"), ("nnU-Net Cup", "riga_cup")):
        source = points[label]
        paths = [ROOT / SOURCE_PATHS.get(f["path"], f["path"]) for f in source["sources"]]
        for path, f in zip(paths, source["sources"], strict=True):
            assert digest(path) == f["sha256"], f["path"]
        c = load_collection(paths, task,
                            {"UNet-skip Cup": "unet_cup", "UNet-skip Disc": "unet_disc", "nnU-Net Cup": "nnunet_cup100"}[label],
                            nnunet=label == "nnU-Net Cup")
        e4_rows[c.config] = alternatives(c, source, nnunet=label == "nnU-Net Cup")
    references = read(FROZEN / "postproc/e1_meff_configs/out/metrics.json")
    e1_rows, collections, draws_map, gates = {}, {}, {}, {}
    for task in ("kvasir", "riga_cup"):
        label = "frozen_" + task
        c = load_collection(list((FROZEN / "per_case_dice" / task).glob("*/seed_*.json")), task, label)
        refc = next(r for r in references["configurations"] if r["config"] == "released_1x1" and r["task"] == task)
        ref = next(r for r in refc["pools"] if r["pool"] == "diverse4_enumeration")
        assert list(c.member_ids) == refc["members"]
        assert set(c.bundle_ids) == {r["fm"] for r in read(FROZEN / "stats/m_eff" / (task + ".json"))["mono_pools"]}
        result, draws = effective(c, ref["bootstrap"]["seed"], label + "_reference_gate")
        same(result["primary_P33"], ref["selection"]["primary_P33"])
        gates[task] = {"status": "PASS", "seed": ref["bootstrap"]["seed"], "tolerance": TOL,
                       "reference_file": str(FROZEN / "postproc/e1_meff_configs/out/metrics.json"),
                       "reference_config": "released_1x1", "reference_pool": "diverse4_enumeration",
                       "recomputed": result["primary_P33"]}
        collections[task] = c
        if ref["bootstrap"]["seed"] == 0:
            e1_rows[label], draws_map[label] = result, draws
        else:
            e1_rows[label], draws_map[label] = effective(c, 0, label)
    write("frozen_reference_gates.json", gates)
    # Both frozen reference gates pass before any FT comparison is calculated.
    comparisons = {}
    ft_reference = read(FT / "FT_TABLE.json")
    for task in ("kvasir", "riga_cup"):
        for arm in ("1e-4", "1e-5"):
            label = f"ft_{task}_lr{arm}"
            root = FT / "kvasir_official" if task == "kvasir" else FT
            c = load_collection(list((root / f"per_case_dice_lr{arm}" / task).glob("*/seed_*.json")), task, label)
            c = align_to(c, collections[task].case_ids)
            table = ft_reference["rows"]["kvasir_official" if task == "kvasir" else task]["lr" + arm]
            assert META[label]["kept_bundles"] == table["ft"]["fms_kept_after_floor"]
            assert c.member_ids == collections[task].member_ids, "report mismatched pools before comparing"
            result, draws = effective(c, 0, label)
            e1_rows[label], draws_map[label] = result, draws
            baseline = e1_rows["frozen_" + task]
            delta = draws[:, 1] - draws_map["frozen_" + task][:, 1]
            valid = np.isfinite(delta)
            comparisons[label + "_minus_frozen"] = {
                "difference": result["primary_P33"]["M_eff"]["value"] - baseline["primary_P33"]["M_eff"]["value"],
                "paired_ci95": np.percentile(delta[valid], [2.5, 97.5]).tolist() if valid.any() else None,
                "B_requested": B, "B_paired_finite": int(valid.sum()), "bootstrap_seed": 0,
                "same_case_set_order_and_members": True,
                "thresholds": {"frozen": baseline["primary_P33"]["threshold"]["value"], "FT": result["primary_P33"]["threshold"]["value"]},
                "estimand": "FT minus frozen: same pooled-P33 rule; each recipe holds its own observed threshold fixed",
                "frozen_root_in_existing_FT_TABLE": table["frozen_root"],
                "frozen_root_in_this_comparison": str(FROZEN), "original_M_eff_definition_unchanged": True}
            write("paired_comparisons.json", comparisons)
    assert all(digest(Path(p)) == value for p, value in INPUTS.items()), "input changed during analysis"
    write("inputs.json", {"sha256": INPUTS, "collections": META})
    payload = {"status": "complete", "B": B, "seed": 0, "floor": FLOOR, "E4": e4_rows, "E1": e1_rows,
               "frozen_reference_gates": gates, "paired_comparisons": comparisons,
               "environment": {"python": platform.python_version(), "numpy": np.__version__},
               "original_M_eff_definition_unchanged": True, "no_training_or_raw_prediction_replay": True}
    write("results.json", payload)
    write("runtime.json", {"pid": os.getpid(), "status": "complete", "started_unix": START,
                           "elapsed_s": time.time() - START, "input_files_hash_verified": len(INPUTS)})
    print("COMPLETE: failure comparison in three settings; effective size for frozen and both FT arms on Kvasir and Cup; all CIs retained", flush=True)


START = time.time()
if __name__ == "__main__":
    # Run directly, this module would write into the source folder; the wrapper sets ROOT and OUT.
    raise SystemExit("run this scorer through code/stats/recompute_controls.py --root ROOT --out OUT")
