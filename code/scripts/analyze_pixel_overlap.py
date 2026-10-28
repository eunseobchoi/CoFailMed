#!/usr/bin/env python3
"""Pixel-level error-mask overlap on the released primary heads (local numpy analysis).

Inputs: results/pixel_overlap/raw/<worker>/out/<task>__<fm>/seed_<s>.npz from
evaluate_error_masks.py.  Each head is re-gated here: sha256 sidecar, PASS receipt, replayed per-case Dice
within 1e-7 of the locally collected released trace, identical case_ids and GT across heads.

Definitions (paper Appendix C.3, pixel-level error overlap): error mask = (sigmoid(logit) > 0.5) XOR GT
at 224x224; pair error-Dice = 2|A&B|/(|A|+|B|) per case; both-empty -> 1 ("uncond", primary) or
excluded ("cond").  FP = err & ~GT, FN = err & GT.  Per case, average the pairs of a stratum; then
average over cases.  Pool = bundles kept by the per-FM functional floor (cross-seed mean Dice >= 0.30).
Strata (primary convention, code/scripts/compute_all.py DINOV2_FAMILY_PAIRS): same_bundle (seed pairs),
same_family_ckpt (DINOv2-B x DINOv2-S), same = union of both, cross_family = all other pairs.
Also: cross_bundle (all distinct-bundle pairs), cross_family_broad (cross_family
minus broad same-family pairs ResNet-18xResNet-50 / CLIP-B/16xCLIP-L/14, FAMILY_GROUPS of
compute_cross_checkpoint_family.py).  CIs: case bootstrap B=2000, seed 0 (ACDC also patient-cluster).
usage: analyze_pixel_overlap.py        (writes PIXEL_OVERLAP_TABLE.{json,md} + panel_inputs/)
       analyze_pixel_overlap.py --tasks kvasir --raw DIR --traces DIR --primary-wc DIR --suffix _official
         (Kvasir official split: writes PIXEL_OVERLAP_TABLE_official.{json,md} only; no example panels)
"""
import argparse, hashlib, json, sys, time, warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/src"), str(ROOT / "code/stats"), str(ROOT / "code/scripts")]
from analysis.rater_sensitivity import BOOTSTRAP_B, DINOV2_PAIR, FUNCTIONAL_FLOOR, RANDOM_SEED, SEEDS  # noqa: E402
from compute_cross_checkpoint_family import FAMILY_GROUPS  # noqa: E402
from fmpool.estimators import functional_floor_filter  # noqa: E402

OUT = ROOT / "results/pixel_overlap"
RAW = OUT / "raw"
TRACES = ROOT / "results/per_case_dice"
PRIMARY_WC = ROOT / "results/within_cross"
TASKS = ("riga_cup", "riga_disc", "kvasir", "acdc_lv")
GATE = 1e-7
warnings.filterwarnings("ignore", "Mean of empty slice")  # cond variant: all-both-empty cases are NaN by definition
PANEL_MONO = [("dinov2_vitb14", s) for s in SEEDS]
PANEL_DIV = [(fm, 42) for fm in ("resnet50", "efficientnet_b0", "clip_vitl14", "biomedclip")]
STRATA = ("same_bundle", "same_family_ckpt", "same", "cross_family", "cross_bundle", "cross_family_broad")
KINDS = ("err", "fp", "fn")


def sha_ok(f):
    want = Path(str(f) + ".sha256").read_text().split()[0]
    return hashlib.sha256(f.read_bytes()).hexdigest() == want


class ReplayGateError(ValueError):
    """Invalid replay inputs; evidence is saved separately from scientific results."""


def fail_gate(task, gate):
    directory = OUT / "gate_failures"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{task}_{time.time_ns()}.json"
    path.write_text(json.dumps({"task": task, "status": "FAIL", "gate": gate,
        "inputs": {"raw": str(RAW), "traces": str(TRACES), "primary_wc": str(PRIMARY_WC)}}, indent=1))
    raise ReplayGateError(f"{task}: replay gate failed; evidence: {path}")


def load_task(task):
    heads, gate = {}, []
    try:
        primary = json.loads((PRIMARY_WC / f"{task}.json").read_text())
        declared = primary["fms_kept_after_floor"]
        available = primary.get("fms_available_before_floor", [])
        if (not isinstance(declared, list) or not declared or len(set(declared)) != len(declared)
                or not isinstance(available, list) or len(set(available)) != len(available)):
            raise ValueError("invalid primary bundle declaration")
        # Require all declared trace bundles, including those below the floor: no-floor
        # counts include them. The kept cohort alone must never define replay counts.
        bundles = {p.name for p in (TRACES / task).iterdir() if p.is_dir()}
        bundles.update(available)
        bundles.update(declared)
        expected = {(fm, seed) for fm in bundles for seed in SEEDS}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        fail_gate(task, [{"pass": False, "error": str(exc)}])
    seen = set()
    for f in sorted(RAW.glob(f"*/out/{task}__*/seed_*.npz")):
        g = {"file": str(f), "worker": f.parts[-4], "pass": False}
        gate.append(g)
        try:
            if not sha_ok(f):
                raise ValueError("checksum mismatch")
            rec = json.loads((f.parent / "cell_receipt.json").read_text())
            with np.load(f) as archive:
                z = dict(archive)
            fm, seed = str(z["fm"]), int(z["seed"])
            key = (fm, seed)
            g.update(fm=fm, seed=seed)
            if (key not in expected or key in seen or f.parent.name != f"{task}__{fm}"
                    or f.name != f"seed_{seed}.npz"):
                raise ValueError("unexpected, duplicate, or mislabeled bundle/seed")
            seen.add(key)
            local = json.loads((TRACES / task / fm / f"seed_{seed}.json").read_text())
            ids = [str(c) for c in z["case_ids"]]
            tdice = np.asarray(local["per_case_dice"])
            if (not ids or len(set(ids)) != len(ids) or ids != local["test_ids"]
                    or tdice.shape != (len(ids),) or z["dice"].shape != tdice.shape
                    or not np.isfinite(tdice).all() or not np.isfinite(z["dice"]).all()
                    or np.any((tdice < 0) | (tdice > 1))
                    or np.any((z["dice"] < 0) | (z["dice"] > 1))):
                raise ValueError("invalid Dice or case IDs")
            diff = float(np.max(np.abs(z["dice"] - tdice)))
            remote = float(rec["seeds"][str(seed)]["max_abs_trace_diff"])
            g.update(replay_trace_diff=remote, local_trace_diff=diff)
            if (rec["status"] != "PASS" or not 0 <= remote <= GATE or diff > GATE
                    or not np.array_equal(z["trace_dice"], tdice)):
                raise ValueError("receipt or Dice gate failed")
            if heads:
                ref = next(iter(heads.values()))
                if any(not np.array_equal(z[k], ref[k]) for k in ("case_ids", "gt_packed", "panel_idx")):
                    raise ValueError("cross-head IDs, GT, or panel indices mismatch")
            heads[key] = z
            g["pass"] = True
        except (OSError, ValueError, KeyError, TypeError, IndexError, EOFError) as exc:
            g["error"] = str(exc)
    for fm, seed in sorted(expected - seen):
        gate.append({"fm": fm, "seed": seed, "pass": False, "error": "missing expected member"})
    if not heads or any(not g["pass"] for g in gate):
        fail_gate(task, gate)
    dice = {k: np.asarray(z["dice"]) for k, z in heads.items()}
    kept = functional_floor_filter(dice, min_mean_dice=FUNCTIONAL_FLOOR, scope="per_fm")
    if {k[0] for k in kept} != set(declared):
        gate.append({"pass": False, "error": "primary pool mismatch",
                     "expected": sorted(declared), "actual": sorted({k[0] for k in kept})})
        fail_gate(task, gate)
    return heads, gate


def pair_values(err, gt):
    """err [M, N, B] packed, gt [N, B] packed -> {kind: (uncond [N, P], cond [N, P])} for triu pairs."""
    m, n, _ = err.shape
    iu = np.triu_indices(m, k=1)
    out = {k: (np.empty((n, len(iu[0]))), np.empty((n, len(iu[0])))) for k in KINDS}
    for c in range(n):
        e = np.unpackbits(err[:, c, :], axis=1).astype(np.float32)
        g = np.unpackbits(gt[c]).astype(np.float32)
        for kind, x in (("err", e), ("fp", e * (1 - g)), ("fn", e * g)):
            inter = x @ x.T
            s = np.diag(inter)
            den = (s[:, None] + s[None, :])[iu]
            d = np.divide(2 * inter[iu], den, out=np.full(den.shape, np.nan), where=den > 0)
            out[kind][0][c], out[kind][1][c] = np.where(den > 0, d, 1.0), d
    return out, iu


def strata_masks(keys, iu):
    a = [keys[i][0] for i in iu[0]]
    b = [keys[j][0] for j in iu[1]]
    same_b = np.array([x == y for x, y in zip(a, b)])
    fam = np.array([frozenset({x, y}) == DINOV2_PAIR for x, y in zip(a, b)])
    broad = np.array([x != y and FAMILY_GROUPS.get(x, x) == FAMILY_GROUPS.get(y, y) for x, y in zip(a, b)])
    same = same_b | fam
    return {"same_bundle": same_b, "same_family_ckpt": fam, "same": same, "cross_family": ~same,
            "cross_bundle": ~same_b, "cross_family_broad": ~same & ~broad}


def boot_ci(vec_a, vec_b, idx):
    """Point + percentile CIs of mean(a), mean(b), mean(a)-mean(b) over resampled cases (nan-aware)."""
    def stat(ix):
        return np.nanmean(vec_a[ix], axis=-1), np.nanmean(vec_b[ix], axis=-1)
    pa, pb = np.nanmean(vec_a), np.nanmean(vec_b)
    ra, rb = stat(idx)
    q = lambda r: [float(v) for v in np.nanpercentile(r, [2.5, 97.5])]  # noqa: E731
    return {"a": float(pa), "b": float(pb), "gap": float(pa - pb), "a_ci": q(ra), "b_ci": q(rb),
            "gap_ci": q(ra - rb), "p_gap_leq0": float(np.mean((ra - rb) <= 0))}


def cluster_index(case_ids, rng, b):
    pts = np.array([c.split("_")[0] for c in case_ids])
    uniq = np.unique(pts)
    members = [np.flatnonzero(pts == p) for p in uniq]
    draws = rng.integers(0, len(uniq), (b, len(uniq)))
    return [np.concatenate([members[p] for p in row]) for row in draws], len(uniq)


def summarize(task, heads, keys, vals, iu, case_ids):
    masks = strata_masks(keys, iu)
    n = len(case_ids)
    idx = np.random.default_rng(RANDOM_SEED).integers(0, n, (BOOTSTRAP_B, n))
    res = {"n_pairs": {s: int(m.sum()) for s, m in masks.items()}}
    for kind in KINDS:
        for var, arr in zip(("uncond", "cond"), vals[kind]):
            per_case = {}
            for s, m in masks.items():
                with np.errstate(invalid="ignore"):
                    per_case[s] = np.nanmean(arr[:, m], axis=1) if m.any() else np.full(n, np.nan)
            block = {s: {"mean": float(np.nanmean(v)) if m.any() else None,
                         "n_both_empty": int(np.isnan(vals[kind][1][:, masks[s]]).sum())}
                     for (s, v), m in zip(per_case.items(), masks.values())}
            block["gap_same_vs_cross_family"] = boot_ci(per_case["same"], per_case["cross_family"], idx)
            block["gap_same_bundle_vs_cross_bundle"] = boot_ci(per_case["same_bundle"], per_case["cross_bundle"], idx)
            block["gap_same_vs_cross_family_broad"] = boot_ci(per_case["same"], per_case["cross_family_broad"], idx)
            if kind == "err" and var == "uncond":
                for s in STRATA:
                    if masks[s].any():
                        r = np.nanmean(per_case[s][idx], axis=1)
                        block[s]["ci"] = [float(v) for v in np.percentile(r, [2.5, 97.5])]
                if task == "acdc_lv":
                    cl, npat = cluster_index(case_ids, np.random.default_rng(RANDOM_SEED), BOOTSTRAP_B)
                    ga = np.array([np.nanmean(per_case["same"][ix]) - np.nanmean(per_case["cross_family"][ix]) for ix in cl])
                    block["gap_same_vs_cross_family"]["patient_cluster_gap_ci"] = [float(v) for v in np.percentile(ga, [2.5, 97.5])]
                    block["gap_same_vs_cross_family"]["n_patients"] = npat
            res[f"{kind}_{var}"] = block
    return res


def analyze(task):
    heads, gate = load_task(task)
    dice = {k: np.asarray(z["dice"]) for k, z in heads.items()}
    kept = functional_floor_filter(dice, min_mean_dice=FUNCTIONAL_FLOOR, scope="per_fm")
    all_keys = sorted(heads)
    err = np.stack([heads[k]["err_packed"] for k in all_keys])
    ref = heads[all_keys[0]]
    case_ids = [str(c) for c in ref["case_ids"]]
    vals, iu = pair_values(err, ref["gt_packed"])
    floor_keys = [k for k in all_keys if k in kept]
    pos = [all_keys.index(k) for k in floor_keys]
    sub_iu = np.triu_indices(len(floor_keys), k=1)
    col = {pair: i for i, pair in enumerate(zip(*iu))}
    sel = [col[(pos[i], pos[j])] for i, j in zip(*sub_iu)]
    fvals = {k: (u[:, sel], c[:, sel]) for k, (u, c) in vals.items()}
    bundles_all = sorted({k[0] for k in all_keys})
    bundles_kept = sorted({k[0] for k in floor_keys})
    out = {"task": task, "n_cases": len(case_ids), "n_heads_replayed": len(all_keys),
           "bundles_replayed": bundles_all, "K_after_floor": len(bundles_kept), "bundles_kept": bundles_kept,
           "bundles_dropped_by_floor": sorted(set(bundles_all) - set(bundles_kept)),
           "primary_pool_match": True,  # enforced by load_task before any pixel aggregation
           "bundle_mean_dice": {b: float(np.mean([dice[(b, s)].mean() for s in SEEDS if (b, s) in dice])) for b in bundles_all},
           "gate": gate, "floor": summarize(task, heads, floor_keys, fvals, sub_iu, case_ids),
           "no_floor": summarize(task, heads, all_keys, vals, iu, case_ids)}
    return out, heads


def example_panels(task, heads, rows_img):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fp_c, fn_c, gt_c = np.array([0.93, 0.14, 0.18]), np.array([0.13, 0.38, 0.93]), (0.10, 0.85, 0.25)
    d = OUT / "panel_inputs"
    d.mkdir(exist_ok=True)
    ref = heads[PANEL_MONO[0]]
    pidx, pids, gt = ref["panel_idx"], [str(c) for c in ref["panel_case_ids"]], ref["panel_gt"]
    members = PANEL_MONO + PANEL_DIV
    pred = np.stack([heads[k]["panel_pred"] for k in members])            # [8 members, 8 cases, H, W]
    dice = np.array([[float(heads[k]["dice"][i]) for i in pidx] for k in members])
    img = rows_img if rows_img is not None else np.full(gt.shape + (3,), 64, np.uint8)
    np.savez_compressed(d / f"{task}_panel_arrays.npz", members=np.array([f"{a}__seed{b}" for a, b in members]),
                        panel_idx=pidx, panel_case_ids=np.array(pids), gt=gt, pred=pred, dice=dice, image=img,
                        image_is_placeholder=rows_img is None)
    manifest = []
    for c in range(len(pidx)):
        err = pred[:, c] ^ gt[c][None]
        fig, axes = plt.subplots(2, 6, figsize=(13, 4.6))
        base = img[c].astype(np.float32) / 255.0
        if base.ndim == 2 or base.shape[-1] == 1:
            base = np.repeat(base.reshape(*gt.shape[1:], 1), 3, -1)
        row_ed = []
        for r, grp in enumerate((range(4), range(4, 8))):
            ax = axes[r, 0]
            ax.imshow(base); ax.contour(gt[c].astype(float), [0.5], colors=[gt_c], linewidths=1.2)
            ax.set_title("Input + GT", fontsize=9)
            ov = np.zeros(gt.shape[1:], int)
            eds = []
            for col, m in enumerate(grp, start=1):
                o = base.copy()
                fp, fn = pred[m, c] & ~gt[c], gt[c] & ~pred[m, c]
                o[fp] = 0.38 * o[fp] + 0.62 * fp_c
                o[fn] = 0.38 * o[fn] + 0.62 * fn_c
                ax = axes[r, col]
                ax.imshow(np.clip(o, 0, 1)); ax.contour(gt[c].astype(float), [0.5], colors=[gt_c], linewidths=0.9)
                a, s = members[m]
                ax.set_title(f"{a} s{s}" if r == 0 else a, fontsize=9)
                ax.text(0.97, 0.04, f"Dice={dice[m, c]:.2f}", transform=ax.transAxes, ha="right", fontsize=8,
                        color="w", bbox=dict(fc="k", alpha=0.7, pad=1.5))
                ov += err[m]
            for i in grp:
                for j in grp:
                    if i < j:
                        den = err[i].sum() + err[j].sum()
                        eds.append(1.0 if den == 0 else 2.0 * (err[i] & err[j]).sum() / den)
            row_ed.append(float(np.mean(eds)))
            ax = axes[r, 5]
            ax.imshow(ov, cmap="hot", vmin=0, vmax=4, interpolation="nearest")
            ax.set_title("Error overlap (0-4)", fontsize=9)
            ax.text(0.97, 0.04, f"err-Dice={row_ed[-1]:.2f}", transform=ax.transAxes, ha="right", fontsize=8,
                    color="w", bbox=dict(fc="k", alpha=0.7, pad=1.5))
        for ax in axes.flat:
            ax.set_xticks([]); ax.set_yticks([])
        axes[0, 0].set_ylabel("DINOv2-B, 4 seeds", fontsize=9)
        axes[1, 0].set_ylabel("4 distinct bundles", fontsize=9)
        fig.suptitle(f"{task} case {pids[c]} (test idx {int(pidx[c])}); red=FP blue=FN green=GT boundary"
                     + ("" if rows_img is not None else "; no input image (placeholder background)"), fontsize=9)
        fig.tight_layout()
        png = d / f"{task}_case{c}_{int(pidx[c])}.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        manifest.append({"png": png.name, "case_id": pids[c], "test_idx": int(pidx[c]), "gt_area": int(gt[c].sum()),
                         "mono_mean_pair_err_dice": row_ed[0], "distinct_mean_pair_err_dice": row_ed[1],
                         "mono_dice": dice[:4, c].tolist(), "distinct_dice": dice[4:, c].tolist()})
    return {"members_mono": [f"{a}__seed{b}" for a, b in PANEL_MONO], "members_distinct": [f"{a}__seed{b}" for a, b in PANEL_DIV],
            "case_rule": "test cases with GT area > 0 sorted by (area, case_id); middle 8",
            "image": "input image" if rows_img is not None else "placeholder (no image replayed)", "cases": manifest}


def panel_images(task):
    f = sorted(RAW.glob(f"*/out/{task}__dinov2_vitb14/panel_images.npz"))
    return np.load(f[0])["image"] if f and sha_ok(f[0]) else None


def brats_panel():
    heads, gate = {}, []
    for f in sorted(RAW.glob("*/out/brats_wt__*/seed_*.npz")):
        rec = json.loads((f.parent / "cell_receipt.json").read_text())
        z = np.load(f)
        fm, seed = str(z["fm"]), int(z["seed"])
        local = json.loads((TRACES / "brats_wt" / fm / f"seed_{seed}.json").read_text())
        diff = float(np.max(np.abs(z["dice"] - np.asarray(local["per_case_dice"]))))
        ok = sha_ok(f) and rec["status"] == "PASS" and diff <= GATE and [str(c) for c in z["case_ids"]] == local["test_ids"]
        gate.append({"fm": fm, "seed": seed, "worker": f.parts[-4], "pass": bool(ok),
                     "replay_trace_diff": rec["seeds"][str(seed)]["max_abs_trace_diff"], "local_trace_diff": diff})
        if ok:
            heads[(fm, seed)] = z
    ref = heads[PANEL_MONO[0]]
    for k, z in heads.items():
        assert np.array_equal(z["panel_idx"], ref["panel_idx"]) and np.array_equal(z["panel_gt"], ref["panel_gt"]), k
    return heads, gate


def md_table(results):
    L = ["# Pixel-level error-mask overlap on released primary heads", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M %Z')} by `code/scripts/analyze_pixel_overlap.py`.",
         "Error mask = (sigmoid(logit)>0.5) XOR GT at 224x224; pair error-Dice per case (both-empty = 1), stratum mean per case, "
         "then mean over cases. Pool = bundles kept by the per-FM 0.30 functional floor. Primary strata: same = same-bundle seed "
         "pairs + DINOv2-B x DINOv2-S; cross = all other pairs. 95% CIs: case bootstrap B=2000, seed 0.", "",
         "| Task | N | K | same-bundle | same-family ckpt (DINOv2-B x S) | same (primary) | cross-family | gap same-cross [95% CI] | gap no-floor |",
         "|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda b, s: f"{b[s]['mean']:.3f} [{b[s]['ci'][0]:.3f},{b[s]['ci'][1]:.3f}]" if b[s].get("ci") else "n/a"  # noqa: E731
    for r in results:
        b = r["floor"]["err_uncond"]
        g = b["gap_same_vs_cross_family"]
        L.append(f"| {r['task']} | {r['n_cases']} | {r['K_after_floor']} | {fmt(b, 'same_bundle')} | {fmt(b, 'same_family_ckpt')} | "
                 f"{fmt(b, 'same')} | {fmt(b, 'cross_family')} | {g['gap']:.3f} [{g['gap_ci'][0]:.3f},{g['gap_ci'][1]:.3f}] | "
                 f"{r['no_floor']['err_uncond']['gap_same_vs_cross_family']['gap']:.3f} |")
    L += ["", "| Task | same-bundle vs all cross-bundle gap [CI] | same vs cross (broad family excl.) gap | "
          "FP-only gap | FN-only gap | cond. (both-empty excluded) gap | ACDC patient-cluster CI |", "|---|---|---|---|---|---|---|"]
    for r in results:
        f = r["floor"]
        lb = f["err_uncond"]["gap_same_bundle_vs_cross_bundle"]
        cl = f["err_uncond"]["gap_same_vs_cross_family"].get("patient_cluster_gap_ci")
        L.append(f"| {r['task']} | {lb['a']:.3f} vs {lb['b']:.3f}: {lb['gap']:.3f} [{lb['gap_ci'][0]:.3f},{lb['gap_ci'][1]:.3f}] | "
                 f"{f['err_uncond']['gap_same_vs_cross_family_broad']['gap']:.3f} | "
                 f"{f['fp_uncond']['gap_same_vs_cross_family']['gap']:.3f} | {f['fn_uncond']['gap_same_vs_cross_family']['gap']:.3f} | "
                 f"{f['err_cond']['gap_same_vs_cross_family']['gap']:.3f} | "
                 + (f"[{cl[0]:.3f},{cl[1]:.3f}] ({f['err_uncond']['gap_same_vs_cross_family']['n_patients']} pts)" if cl else "-") + " |")
    L += ["", "Pairs per stratum and bundles dropped by the floor are in PIXEL_OVERLAP_TABLE.json (`floor.n_pairs`, "
          "`bundles_dropped_by_floor`). Replay gate per head: `gate` (replay and local trace diffs)."]
    return "\n".join(L) + "\n"


def main(argv=None):
    global OUT, RAW, TRACES, PRIMARY_WC
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS),
                    help="task subset; restricted runs write tables only (no figure panels)")
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--traces", type=Path, default=TRACES, help="per_case_dice root of the locally collected traces")
    ap.add_argument("--primary-wc", type=Path, default=PRIMARY_WC, help="within_cross dir of the primary pool")
    ap.add_argument("--out", type=Path, default=OUT, help="output directory")
    ap.add_argument("--suffix", default="", help="output name suffix; non-empty = table only (no example/BraTS panels)")
    a = ap.parse_args(argv)
    OUT, RAW, TRACES, PRIMARY_WC = a.out, a.raw, a.traces, a.primary_wc
    OUT.mkdir(parents=True, exist_ok=True)
    if a.suffix or set(a.tasks) != set(TASKS):
        return table_only(a.tasks, a.suffix)
    results, figs = [], {}
    for task in a.tasks:
        r, heads = analyze(task)
        results.append(r)
        if task == "riga_cup":
            figs[task] = example_panels(task, heads, panel_images(task))
    bh, bgate = brats_panel()
    figs["brats_wt"] = example_panels("brats_wt", bh, panel_images("brats_wt"))
    figs["brats_wt"]["gate"] = bgate
    (OUT / "panel_inputs" / "brats_wt_panel_cases.json").write_text(json.dumps(
        {"rule": figs["brats_wt"]["case_rule"], "cases": [{k: c[k] for k in ("case_id", "test_idx", "gt_area")}
                                                         for c in figs["brats_wt"]["cases"]]}, indent=1))
    payload = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "script": str(Path(__file__).resolve().relative_to(ROOT)),
               "definitions": __doc__.strip(), "bootstrap": {"B": BOOTSTRAP_B, "seed": RANDOM_SEED, "unit": "case"},
               "tasks": results, "panels": figs}
    (OUT / "PIXEL_OVERLAP_TABLE.json").write_text(json.dumps(payload, indent=1))
    (OUT / "PIXEL_OVERLAP_TABLE.md").write_text(md_table(results))
    ng = [g for r in results for g in r["gate"]] + bgate
    print(f"gate: {sum(g['pass'] for g in ng)}/{len(ng)} heads pass")
    print((OUT / "PIXEL_OVERLAP_TABLE.md").read_text())


def table_only(tasks, suffix):
    """Pixel-overlap table for a task subset from an alternative replay (e.g. the Kvasir official split)."""
    try:
        results = [analyze(task)[0] for task in tasks]
    except ReplayGateError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    payload = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "script": str(Path(__file__).resolve().relative_to(ROOT)),
               "definitions": __doc__.strip(), "bootstrap": {"B": BOOTSTRAP_B, "seed": RANDOM_SEED, "unit": "case"},
               "inputs": {"raw": str(RAW), "traces": str(TRACES), "primary_wc": str(PRIMARY_WC)},
               "tasks": results, "panels": {}}
    (OUT / f"PIXEL_OVERLAP_TABLE{suffix}.json").write_text(json.dumps(payload, indent=1))
    (OUT / f"PIXEL_OVERLAP_TABLE{suffix}.md").write_text(md_table(results))
    ng = [g for r in results for g in r["gate"]]
    print(f"gate: {sum(g['pass'] for g in ng)}/{len(ng)} heads pass")
    print((OUT / f"PIXEL_OVERLAP_TABLE{suffix}.md").read_text())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReplayGateError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
