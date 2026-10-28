#!/usr/bin/env python3
"""Medically pretrained encoders vs the incumbent 11-bundle pool (released traces).

Stdlib + numpy only; statistics are computed directly from aligned traces.

Notation, per task t in {riga_cup, riga_disc, kvasir}:
  members  = (bundle, seed) traces d_i in [0,1]^n on the shared test cases (n cases, same ids/order);
  floor    : bundle b is kept iff mean_seeds(mean_cases d_(b,s)) >= 0.30 (per bundle, as in the primary);
  r_ij     = Pearson(d_i, d_j) over cases;
  same(i,j)= bundle_i == bundle_j or {bundle_i, bundle_j} == {dinov2_vitb14, dinov2_vits14} (primary rule);
  I        = kept incumbent members (11 primary bundles x 4 seeds), M_m = members of medical encoder m.
Estimands:
  incumbent_cross   = mean_{i<j in I, not same(i,j)} r_ij   (asserted == primary within_cross cross_rbar)
  newcomer_to_inc_m = mean_{i in M_m, j in I} r_ij
  diff_m            = newcomer_to_inc_m - incumbent_cross
  pooled            = same with M = union of floor-passing medical encoders
  pooled_natural    = natural-only reference: M vs I minus BiomedCLIP, minus natural-natural cross
  med_mutual        = mean_{i in M_a, j in M_b, a != b} r_ij (and per encoder pair)
  pool15 gap        = mean_{same} r_ij - mean_{not same} r_ij over all kept members (11 + medical)
Uncertainty: case bootstrap, B=2000, numpy default_rng(0); each draw resamples the n case columns
with replacement and recomputes every estimand (both terms of each difference) from the same draw;
95% CI = 2.5/97.5 percentiles over draws with a finite value (count reported).
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

TASKS = ("riga_cup", "riga_disc", "kvasir")
MEDICAL = ("retfound_vitl16", "phikon", "rad_dino", "biomedparse")
FLOOR, B, SEED = 0.30, 2000, 0
DINO_PAIR = frozenset({"dinov2_vitb14", "dinov2_vits14"})


def load(root: Path, task: str, bundles) -> dict[tuple[str, int], dict]:
    out = {}
    for b in bundles:
        files = sorted((root / "per_case_dice" / task / b).glob("seed_*.json"))
        assert len(files) == 4, (task, b, len(files))
        for p in files:
            d = json.loads(p.read_text())
            assert (d["task"], d["fm"]) == (task, b), p
            out[(b, int(d["seed"]))] = d
    return out


def bundle_mean(traces: dict, b: str) -> float:
    return float(np.mean([np.mean(d["per_case_dice"]) for (x, _), d in traces.items() if x == b]))


def is_same(a: str, b: str) -> bool:
    return a == b or frozenset({a, b}) == DINO_PAIR


def groups_for(keys, inc, natural, med_kept) -> dict[str, np.ndarray]:
    bundle = [k[0] for k in keys]
    g: dict[str, list] = {"incumbent_cross": [], "pooled_to_inc": [], "pooled_to_natural": [],
                          "natural_cross": [], "med_mutual": [], "pool15_same": [], "pool15_cross": []}
    g |= {f"to_inc::{m}": [] for m in MEDICAL} | {f"to_natural::{m}": [] for m in MEDICAL}
    g |= {f"pair::{a}::{b}": [] for a, b in itertools.combinations(MEDICAL, 2)}
    for i, j in itertools.combinations(range(len(keys)), 2):
        a, b = bundle[i], bundle[j]
        same = is_same(a, b)
        both_pool = {a, b} <= (set(inc) | set(med_kept))
        if both_pool:
            g["pool15_same" if same else "pool15_cross"].append((i, j))
        if a in inc and b in inc and not same:
            g["incumbent_cross"].append((i, j))
        if a in natural and b in natural and not same:
            g["natural_cross"].append((i, j))
        for m, o in ((a, b), (b, a)):
            if m in MEDICAL and o in inc:
                g[f"to_inc::{m}"].append((i, j))
                if m in med_kept:
                    g["pooled_to_inc"].append((i, j))
                if o in natural:
                    g[f"to_natural::{m}"].append((i, j))
                    if m in med_kept:
                        g["pooled_to_natural"].append((i, j))
        if a in MEDICAL and b in MEDICAL and a != b:
            g[f"pair::{min(a, b, key=MEDICAL.index)}::{max(a, b, key=MEDICAL.index)}"].append((i, j))
            if a in med_kept and b in med_kept:
                g["med_mutual"].append((i, j))
    return {k: np.asarray(v, dtype=int).reshape(-1, 2) for k, v in g.items() if v}


def stats(X: np.ndarray, groups: dict[str, np.ndarray]) -> dict[str, float]:
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.corrcoef(X)
    s = {k: float(np.mean(R[v[:, 0], v[:, 1]])) for k, v in groups.items()}
    for m in MEDICAL:
        s[f"diff::{m}"] = s[f"to_inc::{m}"] - s["incumbent_cross"]
    s["pooled_diff"] = s["pooled_to_inc"] - s["incumbent_cross"]
    s["pooled_natural_diff"] = s["pooled_to_natural"] - s["natural_cross"]
    s["pool15_gap"] = s["pool15_same"] - s["pool15_cross"]
    return s


def bootstrap(X: np.ndarray, groups) -> dict[str, dict]:
    rng = np.random.default_rng(SEED)
    n = X.shape[1]
    draws = [stats(X[:, rng.integers(0, n, size=n)], groups) for _ in range(B)]
    out = {}
    for k in draws[0]:
        v = np.asarray([d[k] for d in draws])
        v = v[np.isfinite(v)]
        out[k] = {"ci95": [float(x) for x in np.percentile(v, [2.5, 97.5])], "n_valid": int(v.size)}
    return out


PRIMARY_WC = "within_cross/{task}.json"  # relative to --primary-root, as written by compute_all.py


def task_result(primary: Path, medroot: Path, task: str, primary_wc: str = PRIMARY_WC) -> dict:
    wc = json.loads((primary / primary_wc.format(task=task)).read_text())
    inc_all = load(primary, task, wc["fms_available_before_floor"])
    med = load(medroot, task, MEDICAL)
    inc = sorted(b for b in wc["fms_available_before_floor"] if bundle_mean(inc_all, b) >= FLOOR)
    assert inc == sorted(wc["fms_kept_after_floor"]), (task, inc)
    med_means = {m: bundle_mean(med, m) for m in MEDICAL}
    med_kept = [m for m in MEDICAL if med_means[m] >= FLOOR]
    natural = [b for b in inc if b != "biomedclip"]
    traces = {k: v for k, v in inc_all.items() if k[0] in inc} | med
    keys = sorted(traces)
    ids0 = traces[keys[0]]["test_ids"]
    assert all(traces[k]["test_ids"] == ids0 for k in keys), f"{task}: test_ids differ"
    X = np.asarray([traces[k]["per_case_dice"] for k in keys], dtype=float)
    groups = groups_for(keys, inc, natural, med_kept)
    point = stats(X, groups)
    assert all(np.isfinite(v) for v in point.values()), task
    # independent check against the primary pipeline (compute_all within_cross)
    assert abs(point["incumbent_cross"] - wc["cross_rbar"]) < 1e-9, (task, point["incumbent_cross"], wc["cross_rbar"])
    assert len(groups["incumbent_cross"]) == wc["n_cross_pairs"], task
    ci = bootstrap(X, groups)
    est = lambda k: {"estimate": point[k], **ci[k], "n_pairs": int(len(groups[k])) if k in groups else None}
    return {
        "n_cases": len(ids0), "incumbents_kept": inc, "medical_kept": med_kept,
        "incumbent_cross_primary_json": wc["cross_rbar"], "incumbent_cross": est("incumbent_cross"),
        "primary11_gap_json": wc["gap"], "primary11_gap_ci95_json": wc["case_bootstrap_95ci_gap"],
        "per_encoder": {m: {"mean_dice": med_means[m], "passes_floor": m in med_kept,
                            "newcomer_to_incumbent": est(f"to_inc::{m}"),
                            "diff_vs_incumbent_cross": {"estimate": point[f"diff::{m}"], **ci[f"diff::{m}"]},
                            "newcomer_to_natural_incumbents": est(f"to_natural::{m}")} for m in MEDICAL},
        "pooled_medical": {"newcomer_to_incumbent": est("pooled_to_inc"),
                           "diff_vs_incumbent_cross": {"estimate": point["pooled_diff"], **ci["pooled_diff"]},
                           "legacy_style_natural_only": {
                               "newcomer_to_natural": est("pooled_to_natural"), "natural_cross": est("natural_cross"),
                               "diff": {"estimate": point["pooled_natural_diff"], **ci["pooled_natural_diff"]}}},
        "medical_mutual": est("med_mutual"),
        "medical_pairs": {k.removeprefix("pair::").replace("::", "--"): est(k) for k in groups if k.startswith("pair::")},
        "pool15": {"n_bundles": len(inc) + len(med_kept), "same": est("pool15_same"), "cross": est("pool15_cross"),
                   "gap": {"estimate": point["pool15_gap"], **ci["pool15_gap"]}},
    }


def f(x): return f"{x:.3f}"


def fe(e): return f"{f(e['estimate'])} [{f(e['ci95'][0])}, {f(e['ci95'][1])}]"


def markdown(d: dict) -> str:
    L = ["# Medical encoders vs incumbent pool (released traces)", "",
         "newcomer→inc = mean Pearson of the encoder's 4 seeds to all kept incumbent members; "
         "inc cross = primary cross r̄ (same bundle and DINOv2-S/B excluded); diff = newcomer→inc − inc cross. "
         "Case bootstrap B=2000 seed 0; floor 0.30 per bundle.", "",
         "| task | encoder | mean Dice | newcomer→inc r̄ [CI] | inc cross r̄ | diff [CI] |",
         "|---|---|---:|---:|---:|---:|"]
    for t, r in d["tasks"].items():
        ic = f(r["incumbent_cross"]["estimate"])
        for m, e in r["per_encoder"].items():
            L.append(f"| {t} | {m}{'' if e['passes_floor'] else ' (below floor)'} | {f(e['mean_dice'])} "
                     f"| {fe(e['newcomer_to_incumbent'])} | {ic} | {fe(e['diff_vs_incumbent_cross'])} |")
        p = r["pooled_medical"]
        L.append(f"| {t} | pooled medical | — | {fe(p['newcomer_to_incumbent'])} | {ic} | {fe(p['diff_vs_incumbent_cross'])} |")
    L += ["", "| task | medical mutual r̄ [CI] | pooled diff, natural-only reference [CI] "
          "| 15-bundle same / cross / gap [CI] | 11-bundle primary gap [CI] |", "|---|---:|---:|---:|---:|"]
    for t, r in d["tasks"].items():
        q, g = r["pool15"], r["primary11_gap_ci95_json"]
        L.append(f"| {t} | {fe(r['medical_mutual'])} | {fe(r['pooled_medical']['legacy_style_natural_only']['diff'])} "
                 f"| {f(q['same']['estimate'])} / {f(q['cross']['estimate'])} / {fe(q['gap'])} "
                 f"(K={q['n_bundles']}) | {f(r['primary11_gap_json'])} [{f(g[0])}, {f(g[1])}] |")
    L += ["",
          "Medical pairs: " + "; ".join(f"{t}: " + ", ".join(f"{k} {f(v['estimate'])}" for k, v in r["medical_pairs"].items())
                                        for t, r in d["tasks"].items()), ""]
    return "\n".join(L)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--primary-root", type=Path, required=True)
    ap.add_argument("--medenc-root", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS), help="task subset (default: all three)")
    ap.add_argument("--medical", nargs="+", choices=MEDICAL, default=list(MEDICAL),
                    help="medical encoders to score (default: all four; omit phikon for trace sets without it)")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: --medenc-root)")
    ap.add_argument("--primary-wc", default=PRIMARY_WC,
                    help="primary within_cross JSON relative to --primary-root; {task} is substituted")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    global MEDICAL
    a = parse_args(argv)
    MEDICAL = tuple(m for m in MEDICAL if m in a.medical)  # keep the study order
    out_dir = a.out or a.medenc_root
    out_dir.mkdir(parents=True, exist_ok=True)
    d = {"script": "code/scripts/compute_medenc_pool.py", "bootstrap": {"B": B, "seed": SEED, "unit": "case"},
         "floor": FLOOR,
         **({"primary_wc": a.primary_wc} if a.primary_wc != PRIMARY_WC else {}),  # default output unchanged
         "tasks": {t: task_result(a.primary_root, a.medenc_root, t, a.primary_wc) for t in a.tasks}}
    (out_dir / "MEDENC_TABLE.json").write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")
    (out_dir / "MEDENC_TABLE.md").write_text(markdown(d))
    print(out_dir / "MEDENC_TABLE.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
