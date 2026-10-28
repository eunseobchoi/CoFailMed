#!/usr/bin/env python3
"""ViT same-architecture / different-pretraining diagnostic on released traces.

Computes the same-architecture diagnostic and same-pair share
from ``<root>/per_case_dice/{task}/{fm}/seed_{42..45}.json``.

Definitions:
* ViT subset = {dinov2_vitb14, biomedclip, mae_vitb16, deit_vitb16, clip_vitb16}.
* Functional filter = 0.30 floor on the bundle's cross-seed mean Dice on that
  task (mean over seeds of the per-seed case-mean; same rule as
  ``fmpool.estimators.functional_floor_filter(scope="per_fm")``).
* Seed floor = mean Pearson over same-bundle seed pairs of functional ViTs
  (6 pairs per bundle).
* ViT cross-pretrain = mean Pearson over all trace pairs from two different
  functional ViT bundles (16 pairs per bundle pair).
* Cross ref. = ``cross_rbar`` of ``<root>/within_cross/<task>.json`` (``--within-cross`` overrides the folder).
* Floor - cross-pretrain = seed floor - ViT cross-pretrain.
Pearson and NaN handling mirror ``fmpool.estimators._pearson_matrix`` (NaN
for std < 1e-8, NaN pairs excluded from means); means are pair-weighted.
Uncertainty: case bootstrap (B=2000, ``np.random.default_rng(0)``, resample
cases, recompute both means). For ACDC/BraTS a patient-cluster bootstrap is
reported as a supplementary interval because slices are not independent.

Same-pair share: 6K same-bundle seed pairs over 6K+16 same pairs
when both DINOv2-B and DINOv2-S are retained (else 6K/6K), K = number of
bundles in ``fms_kept_after_floor``; checked against ``n_within_pairs``.

stdlib + numpy only (no scipy / fmpool import). ``--selftest`` checks the
pair-count, floor and share logic on synthetic traces.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "results"
TASKS = ("riga_cup", "acdc_lv", "kvasir", "brats_wt", "riga_disc")
VIT_SUBSET = ("dinov2_vitb14", "biomedclip", "mae_vitb16", "deit_vitb16", "clip_vitb16")
DINOV2_PAIR = frozenset({"dinov2_vitb14", "dinov2_vits14"})
SEEDS = (42, 43, 44, 45)
FLOOR = 0.30
B = 2000
BOOT_SEED = 0
EPS = 1e-8
CROSS_REF_KEY = "cross_rbar"
_BRATS_ID = re.compile(r"(BraTS-GLI-[0-9]{5})(?:-[0-9]{3}(?:_slice[0-9]{3,})?)?")


def load_traces(root: Path, task: str, fms) -> tuple[dict, tuple, list]:
    """Return ({(fm, seed): dice}, test_ids, [(relpath, sha256)])."""
    traces, ids, files = {}, None, []
    for fm in fms:
        for seed in SEEDS:
            path = root / "per_case_dice" / task / fm / f"seed_{seed}.json"
            raw = path.read_bytes()
            d = json.loads(raw)
            if d["task"] != task or d["fm"] != fm or int(d["seed"]) != seed:
                raise ValueError(f"{path}: identity mismatch")
            x = np.asarray(d["per_case_dice"], dtype=np.float64)
            if x.ndim != 1 or not np.isfinite(x).all() or (x < 0).any() or (x > 1).any():
                raise ValueError(f"{path}: invalid Dice vector")
            tid = tuple(d["test_ids"])
            if ids is None:
                ids = tid
            elif tid != ids:
                raise ValueError(f"{path}: test_ids differ from first trace")
            traces[(fm, seed)] = x
            files.append((str(path.relative_to(root)), hashlib.sha256(raw).hexdigest()))
    return traces, ids, files


def bundle_mean_dice(traces: dict) -> dict:
    fms = sorted({fm for fm, _ in traces})
    return {fm: float(np.mean([traces[k].mean() for k in traces if k[0] == fm])) for fm in fms}


def pair_means(traces: dict, idx=None, same=lambda a, b: a == b) -> dict:
    """Pair-weighted mean Pearson over same / cross pairs (NaN-excluded)."""
    keys = sorted(traces)
    data = np.stack([traces[k] if idx is None else traces[k][idx] for k in keys])
    valid = data.std(axis=1) >= EPS
    corr = np.full((len(keys), len(keys)), np.nan)
    vi = np.flatnonzero(valid)
    if len(vi) > 1:
        corr[np.ix_(vi, vi)] = np.corrcoef(data[vi])
    buckets = {True: [], False: []}
    for i, j in itertools.combinations(range(len(keys)), 2):
        buckets[bool(same(keys[i][0], keys[j][0]))].append(corr[i, j])
    out = {}
    for name, flag in (("same", True), ("cross", False)):
        v = np.asarray(buckets[flag])
        v = v[~np.isnan(v)]
        out[name] = float(v.mean()) if v.size else float("nan")
        out[f"n_{name}_pairs"] = int(v.size)
    return out


def boot_indices(n: int, units=None):
    """Mirror of fmpool.estimators._bootstrap_case_indices (case or cluster)."""
    rng = np.random.default_rng(BOOT_SEED)
    groups = None
    if units is not None:
        lab = np.asarray(units)
        groups = [np.flatnonzero(lab == u) for u in sorted(set(units))]
    for _ in range(B):
        if groups is None:
            yield rng.integers(0, n, n)
        else:
            yield np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])


def bootstrap(traces: dict, n: int, units=None) -> dict:
    vals = {"seed_floor": [], "cross_pretrain": [], "diff": []}
    for idx in boot_indices(n, units):
        m = pair_means(traces, idx)
        for k, v in (("seed_floor", m["same"]), ("cross_pretrain", m["cross"]),
                     ("diff", m["same"] - m["cross"])):
            if not np.isnan(v):
                vals[k].append(v)
    return {k: {"ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                if v else None, "n_valid_boot": len(v)} for k, v in vals.items()}


def cluster_units(task: str, ids: tuple):
    if task == "acdc_lv":
        return "ACDC patient", [i.split("_frame", 1)[0] for i in ids]
    if task == "brats_wt":
        return "BraTS-GLI patient prefix", [_BRATS_ID.fullmatch(i).group(1) for i in ids]
    return None, None


def pair_share(kept: list) -> dict:
    k = len(kept)
    both = DINOV2_PAIR <= set(kept)
    seed_pairs = 6 * k
    total = seed_pairs + (16 if both else 0)
    return {"K": k, "both_dinov2_kept": both, "same_bundle_seed_pairs": seed_pairs,
            "total_same_pairs": total, "share": seed_pairs / total}


def diagnostic(traces: dict, ids: tuple, task: str) -> dict:
    means = bundle_mean_dice(traces)
    func = [fm for fm in VIT_SUBSET if fm in means and means[fm] >= FLOOR]
    sub = {k: v for k, v in traces.items() if k[0] in func}
    m = pair_means(sub)
    out = {"vit_mean_dice": means, "functional_vits": func,
           "n_functional": f"{len(func)}/{len(VIT_SUBSET)}",
           "seed_floor": m["same"], "cross_pretrain": m["cross"],
           "diff_floor_minus_cross_pretrain": m["same"] - m["cross"],
           "n_seed_pairs": m["n_same_pairs"], "n_cross_pretrain_pairs": m["n_cross_pairs"]}
    if len(func) >= 2:
        out["case_bootstrap"] = bootstrap(sub, len(ids))
        unit_name, units = cluster_units(task, ids)
        if units is not None:
            out["cluster_bootstrap"] = {"unit": unit_name, "n_units": len(set(units)),
                                        **bootstrap(sub, len(ids), units)}
    return out


def run_task(root: Path, task: str, within_cross: Path | None = None) -> dict:
    wc_path = (within_cross or root / "within_cross") / f"{task}.json"
    wc = json.loads(wc_path.read_text())
    traces, ids, files = load_traces(root, task, sorted(set(VIT_SUBSET) | set(wc["fms_available_before_floor"])))
    vit = {k: v for k, v in traces.items() if k[0] in VIT_SUBSET}
    res = {"task": task, "n_cases": len(ids), **diagnostic(vit, ids, task),
           "cross_ref": wc[CROSS_REF_KEY],
           "cross_ref_source": f"{wc_path.relative_to(root)}:{CROSS_REF_KEY}"}
    # Regression check: canonical floor + within/cross reproduce stats/within_cross.
    means = bundle_mean_dice(traces)
    kept = sorted(fm for fm, v in means.items() if v >= FLOOR)
    canon = pair_means({k: v for k, v in traces.items() if k[0] in kept},
                       same=lambda a, b: a == b or frozenset({a, b}) == DINOV2_PAIR)
    res["canonical_check"] = {
        "kept_matches": kept == sorted(wc["fms_kept_after_floor"]),
        "within_abs_err": abs(canon["same"] - wc["within_rbar"]),
        "cross_abs_err": abs(canon["cross"] - wc["cross_rbar"]),
        "n_within_pairs_recomputed": canon["n_same_pairs"]}
    share = pair_share(wc["fms_kept_after_floor"])
    share["n_within_pairs_json"] = wc["n_within_pairs"]
    share["matches_json"] = share["total_same_pairs"] == wc["n_within_pairs"]
    res["pair_share"] = share
    res["source_files"] = [{"path": p, "sha256": h} for p, h in files] + [
        {"path": str(wc_path.relative_to(root)),
         "sha256": hashlib.sha256(wc_path.read_bytes()).hexdigest()}]
    return res


def f3(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def ci(b) -> str:
    return "n/a" if not b or b["diff"]["ci95"] is None else "[{:.3f}, {:.3f}]".format(*b["diff"]["ci95"])


def write_md(rows: list, path: Path, meta: dict) -> None:
    short = {"dinov2_vitb14": "DINOv2-B", "biomedclip": "BiomedCLIP", "mae_vitb16": "MAE-B",
             "deit_vitb16": "DeiT-B", "clip_vitb16": "CLIP-B"}
    L = ["# ViT same-architecture / different-pretraining diagnostic (released traces)", "",
         f"Generated {meta['generated_at']} by `{meta['script']}`. Root: `{meta['root']}`.",
         "Definitions: see the script docstring for the ViT subset, functional floor, pair means and same-pair share.",
         f"Cross ref. = `stats/within_cross/<task>.json:{CROSS_REF_KEY}`. Case bootstrap B={B}, seed {BOOT_SEED}; "
         "cluster bootstrap (patient) is supplementary for slice tasks.", "",
         "## Table 1 - diagnostic per task", "",
         "| Task | Func. ViTs | Seed floor | ViT cross-pretrain | Cross ref. | Floor - cross-pretrain | 95% CI (case) | 95% CI (patient cluster) |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['task']} | {r['n_functional']} | {f3(r['seed_floor'])} | {f3(r['cross_pretrain'])} | "
                 f"{f3(r['cross_ref'])} | {f3(r['diff_floor_minus_cross_pretrain'])} | {ci(r.get('case_bootstrap'))} | "
                 f"{ci(r.get('cluster_bootstrap')) if 'cluster_bootstrap' in r else '-'} |")
    L += ["", "Per-ViT cross-seed mean Dice (floor 0.30; * = below floor, dropped):", ""]
    for r in rows:
        items = [f"{short[fm]} {r['vit_mean_dice'][fm]:.3f}{'' if fm in r['functional_vits'] else '*'}"
                 for fm in VIT_SUBSET]
        L.append(f"- {r['task']}: " + ", ".join(items) + f" (pairs: {r['n_seed_pairs']} seed, "
                 f"{r['n_cross_pretrain_pairs']} cross-pretrain)")
    L += ["", "## Table 2 - same-pair share per task", "",
          "| Task | K kept | Both DINOv2 kept | Same-bundle seed pairs | Total same pairs | Share | = n_within_pairs? |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        s = r["pair_share"]
        L.append(f"| {r['task']} | {s['K']} | {s['both_dinov2_kept']} | {s['same_bundle_seed_pairs']} | "
                 f"{s['total_same_pairs']} | {100 * s['share']:.1f}% | {s['matches_json']} |")
    L += ["", "Canonical regression check (recomputed within/cross vs stats/within_cross): " + "; ".join(
        f"{r['task']} kept={r['canonical_check']['kept_matches']} "
        f"|dw|={r['canonical_check']['within_abs_err']:.1e} |dc|={r['canonical_check']['cross_abs_err']:.1e}"
        for r in rows), ""]
    path.write_text("\n".join(L))


def selftest() -> None:
    rng = np.random.default_rng(1)
    n = 40
    fms = list(VIT_SUBSET) + ["resnet50"]
    base = {fm: rng.uniform(0.5, 0.9, n) for fm in fms}
    tr = {(fm, s): np.clip(base[fm] + rng.normal(0, 0.01, n), 0, 1) for fm in fms for s in SEEDS}
    for s in SEEDS:  # CLIP-B below floor -> dropped
        tr[("clip_vitb16", s)] = tr[("clip_vitb16", s)] * 0.2
    vit = {k: v for k, v in tr.items() if k[0] in VIT_SUBSET}
    d = diagnostic(vit, tuple(f"c{i}" for i in range(n)), "riga_cup")
    assert d["functional_vits"] == ["dinov2_vitb14", "biomedclip", "mae_vitb16", "deit_vitb16"], d
    assert d["n_seed_pairs"] == 6 * 4 and d["n_cross_pretrain_pairs"] == 16 * 6, d
    assert d["seed_floor"] > 0.95 and abs(d["cross_pretrain"]) < 0.3, d
    assert d["case_bootstrap"]["diff"]["n_valid_boot"] == B
    # all traces affine copies of one vector -> every r = 1 -> diff 0
    one = rng.uniform(0.4, 0.9, n)
    same = {(fm, s): 0.5 * one + 0.01 * i for i, fm in enumerate(VIT_SUBSET) for s in SEEDS}
    m = pair_means(same)
    assert abs(m["same"] - 1) < 1e-12 and abs(m["cross"] - 1) < 1e-12
    assert m["n_same_pairs"] == 30 and m["n_cross_pairs"] == 160
    # constant trace -> NaN pairs excluded
    const = dict(same)
    const[("mae_vitb16", 42)] = np.full(n, 0.5)
    assert pair_means(const)["n_same_pairs"] == 27
    # cluster bootstrap resamples whole units
    idx = next(boot_indices(6, ["a", "a", "b", "b", "c", "c"]))
    assert len(idx) == 6 and all(idx[i] // 2 == idx[i + 1] // 2 for i in (0, 2, 4))
    # pair share
    k11 = pair_share(sorted(set(VIT_SUBSET) | {"dinov2_vits14", "a", "b", "c", "d", "e"}))
    assert (k11["same_bundle_seed_pairs"], k11["total_same_pairs"]) == (66, 82)
    assert pair_share(["dinov2_vitb14", "x", "y"])["share"] == 1.0
    assert pair_share(["dinov2_vitb14", "dinov2_vits14", "a", "b", "c", "d", "e", "f", "g"])["total_same_pairs"] == 70
    print("selftest OK")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--within-cross", type=Path, default=None,
                    help="folder with within_cross/<task>.json (default: <root>/within_cross, as written by compute_all.py)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        selftest()
        return 0
    out = a.out or a.root / "postproc" / "vit_arch_diagnostic"
    out.mkdir(parents=True, exist_ok=True)
    rows = [run_task(a.root, t, a.within_cross) for t in TASKS]
    meta = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "script": str(Path(__file__).resolve().relative_to(REPO)), "root": str(a.root),
            "argv": sys.argv, "vit_subset": VIT_SUBSET, "floor": FLOOR, "B": B,
            "bootstrap_seed": BOOT_SEED, "cross_ref_key": CROSS_REF_KEY,
            "evidence_role": "post-hoc diagnostic on primary test traces",
            "note": "Seed floor / cross-pretrain are pair-weighted means; NaN pairs excluded."}
    (out / "vit_arch_diagnostic.json").write_text(json.dumps({"meta": meta, "tasks": rows}, indent=1))
    write_md(rows, out / "VIT_ARCH_DIAGNOSTIC.md", meta)
    print((out / "VIT_ARCH_DIAGNOSTIC.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
