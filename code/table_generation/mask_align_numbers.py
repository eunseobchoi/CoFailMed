r"""Mask-alignment sensitivity macros for the numbers builder (stdlib only).

Reads results/mask_resize/SUMMARY.json (nearest-exact vs primary nearest resizing: within/cross and the paired gap
difference; its study producer script is not released, and the released replay compute_paired_gap_difference.py
writes different key names) and MASK_OFFSET.json (per-axis centroid offset between nearest and nearest-exact masks on
the real 224x224 masks; its producer needs the masks and is not released). The summary key `v3_legacy_nearest` is the
primary nearest mask convention and `nearest_exact` the sensitivity.

Macros (T = Cup, Disc, Acdc, Kvasir): MaskAlignGap<T>, MaskAlignGapCI<T> (nearest-exact gap, primary CI),
MaskAlignK<T>, MaskAlignDelta<T>, MaskAlignDeltaCI<T> (paired gap difference nearest-exact minus primary nearest on shared bootstrap
draws; 4 decimals), MaskAlignMaxAbsDelta (4 decimals), MaskAlignOffsetPx (pooled mean |per-axis offset|, output px,
2 decimals), MaskAlignOffsetPxRange ([min, max] over tasks of the per-task mean |offset|), MaskAlignSwitch (text).
"""
from __future__ import annotations

SUMMARY = "results/mask_resize/SUMMARY.json"
OFFSET = "results/mask_resize/MASK_OFFSET.json"
# ACDC-LV sensitivity uses the official 100/50 patient split and the same summary schema.
SUMMARY_ACDC_OFFICIAL = "results/mask_resize_acdc/SUMMARY.json"
FAM = "S10-mask-align"
MEASURED = ("kvasir", "acdc_lv", "riga_cup", "riga_disc")  # rows follow the builder's TASKS order and labels (BraTS not measured)
SWITCH = r"\texttt{FMPOOL\_MASK\_RESIZE=nearest-exact} (default: \texttt{nearest})"


def input_paths() -> list[str]:
    return [SUMMARY, OFFSET, SUMMARY_ACDC_OFFICIAL]


def summary_path(api, task: str) -> str:
    return SUMMARY_ACDC_OFFICIAL if task == "acdc_lv" and getattr(api, "ACDC_OFFICIAL", False) else SUMMARY


def _f4(api, x) -> str:
    s = f"{api.num(x):.4f}"
    return "0.0000" if s == "-0.0000" else s


def _ci4(api, v) -> str:
    api.fci(v)  # validates a finite ordered pair
    return f"[{_f4(api, v[0])}, {_f4(api, v[1])}]"


def build(b, api) -> dict[str, str]:
    b.put("MaskAlignSwitch", SWITCH)
    tasks = [x for x in api.TASKS if x[0] in MEASURED]
    used = sorted({summary_path(api, t) for t, _, _ in tasks})
    o, deltas, rows = b.load(OFFSET), [], []
    for t, tok, label in tasks:
        src, r = summary_path(api, t), ("tasks", t)  # Official ACDC source; other tasks use SUMMARY.
        d = b.load(src)
        gap = b.num3(f"MaskAlignGap{tok}", d, src, *r, "nearest_exact", "gap")
        ci = b.ci(f"MaskAlignGapCI{tok}", d, src, *r, "nearest_exact", "gap_primary_ci")
        k = b.count(f"MaskAlignK{tok}", d, src, *r, "nearest_exact", "K")
        delta = b.value(f"MaskAlignDelta{tok}", lambda r=r: _f4(api, api.dig(d, src, *r, "gap_diff_exact_minus_v3")))
        dci = b.value(f"MaskAlignDeltaCI{tok}", lambda r=r: _ci4(api, api.dig(d, src, *r, "gap_diff_ci95")))
        v3 = b.cell(lambda r=r: f"{api.f3(api.dig(d, src, *r, 'v3_legacy_nearest', 'gap'))} "
                                f"{api.fci(api.dig(d, src, *r, 'v3_legacy_nearest', 'gap_primary_ci'))}")
        k3 = b.cell(lambda r=r: api.fint(api.dig(d, src, *r, "v3_legacy_nearest", "K")))
        rows.append([label, f"{k3}/{k}", v3, f"{gap} {ci}", f"{delta} {dci}"])
        deltas.append(lambda d=d, src=src, r=r: abs(api.num(api.dig(d, src, *r, "gap_diff_exact_minus_v3"))))
    b.value("MaskAlignMaxAbsDelta", lambda: _f4(api, max(f() for f in deltas)))
    b.value("MaskAlignOffsetPx", lambda: f"{api.num(api.dig(o, OFFSET, 'pooled_mean_abs_axis')):.2f}")
    per_task = lambda: [api.num(api.dig(o, OFFSET, t, "mean_abs_axis")) for t, _, _ in tasks]  # noqa: E731
    b.value("MaskAlignOffsetPxRange", lambda: f"[{min(per_task()):.2f}, {max(per_task()):.2f}]")
    return {"table_mask_align.tex": api.table(
        b, used + [OFFSET], "Task & K (primary/exact) & primary Delta [primary CI] & nearest-exact Delta [primary CI] & "
        "paired Delta exact minus primary [95% CI, shared draws; ACDC patients]", rows,
        ["tasks/<task>/{v3_legacy_nearest,nearest_exact}/{gap,gap_primary_ci,K}; tasks/<task>/{gap_diff_exact_minus_v3,"
         "gap_diff_ci95}"], ["BraTS was not evaluated in the mask-resizing sensitivity."])}
