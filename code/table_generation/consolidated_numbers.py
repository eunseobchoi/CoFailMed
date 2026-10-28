"""Current paper panels, projected verbatim from existing number-builder rows."""
from __future__ import annotations

PRIMARY, ROBUST, FAILTH, MEFF = ("table_primary_corrected.tex", "table_robustness_corrected.tex",
                                 "table_app_failure_thresholds.tex", "table_app_meff.tex")
UNET, FT, MEDSUM, XCKPT, PIXEL = ("table_unet_skip.tex", "table_ft.tex", "table_medenc_pool_summary.tex",
                                  "table_distinct_checkpoint.tex", "table_pixel_overlap.tex")
HELDOUT, POOLSEL, ITEM = "table_heldout_rows.tex", "table_pool_selection_freq.tex", "table_app_item_difficulty.tex"
FLOOR_WORDS = (("0.20", "Twenty"), ("0.25", "TwentyFive"), ("0.35", "ThirtyFive"))  # 0.40 is a robustness cell
UNET11, UNET9 = "UNet-skip readout, 11 bundles", "UNet-skip readout, 9 bundles (CLIP excluded)"
RANDINIT, MEDENC15 = "random-initialisation CNN", "medical-encoder extended pool (15 bundles)"
NA = "--"
# Spleen passes the floor only through empty targets; Heart has no functional pool.
POOLSEL_NA = {"Spleen": "n/a (degenerate)", "Heart": "n/a (no functional pool)"}
REQUIRES = "cells are copied from current result rows or composed from corrected_numbers.tex macros"


class Bodies:
    """Current rows and source paths, with a checked row lookup."""

    def __init__(self, api, files):
        self.api, self.files = api, files

    def rows(self, name):
        return self.files[name][1]

    def row(self, name, label, second=None):
        for r in self.rows(name):
            if r[0] == label and (second is None or r[1] == second):
                return r
        raise self.api.Fatal(f"{name}: no row {label!r}" + (f" / {second!r}" if second else ""))

    def sources(self, *names):
        return [p for n in names for p in self.files[n][0]]


def mci(m, name):
    """Macro pair <name>, <name>CI in the E-column layout."""
    return f"{m[name]} {m[name + 'CI']}"


def primary_delta(m, tok):
    """Corrected primary Delta over its primary-unit CI (the Table 1 value and cluster/case CI)."""
    return f"{m[f'Gap{tok}Corrected']} {m[f'Gap{tok}PrimaryCI']}"


def estimator_variants(b, api, B):
    m, rows = b.macros, []
    for t, tok, label in api.TASKS:
        r = B.row(ROBUST, label)
        floors = [f"{m[f'Gap{tok}Floor{w}']} ({m[f'K{tok}Floor{w}']})" for _, w in FLOOR_WORDS]
        rows.append([label, primary_delta(m, tok), m[f"Gap{tok}NoFloor"], *floors, r[7], r[5], r[6],
                     m[f"GapMetaLeaveOut{tok}"], f"{m[f'N{tok}CalibUnits']}/{m[f'N{tok}EvalUnits']}",
                     m[f"K{tok}CalibSelected"], r[1], r[3]])
    return B.sources(PRIMARY, ROBUST) + [api.INDEP], rows


def failure_events(b, api, B):
    m, rows = b.macros, []
    for t, tok, label in api.TASKS:
        f33, r, e = B.row(FAILTH, label, "P33"), B.row(ROBUST, label), B.row(MEFF, label)
        alt = [mci(m, f"{n}Gap{tok}PThirtyThree") for n in ("Q", "Disagreement", "DoubleFault")]
        rows.append([label, f33[5], r[4], *alt, f33[6], e[5], e[7], e[1]])
    return B.sources(FAILTH, ROBUST, MEFF), rows


def controls(b, api, B):
    m, rows = b.macros, []
    lab = lambda t: "Kvasir (official)" if t == "kvasir" else api.LABEL[t]  # noqa: E731
    unet_tasks, rand = api.SECONDARY_FAMILIES[0][3], api.SECONDARY_FAMILIES[1]
    for t in unet_tasks:
        u = B.row(UNET, api.LABEL[t])
        rows.append([lab(t), UNET11, u[1], u[2], u[3], u[4], u[5], NA])
    for t in unet_tasks:
        tok, u = api.TOK[t], B.row(UNET, api.LABEL[t])
        rows.append([lab(t), UNET9, m[f"UnetSkipNineK{tok}"], m[f"UnetSkipNineWithinRbar{tok}"],
                     m[f"UnetSkipNineCrossRbar{tok}"], mci(m, f"UnetSkipNineGap{tok}"), u[7], NA])
    rows += [[f[0], f"FT, encoder lr {f[1]}", f[2], f[3], f[4], f[5], f[6], f[7]] for f in B.rows(FT)]
    for t in rand[3]:
        tok = api.TOK[t]
        rows.append([lab(t), RANDINIT, m[f"RandInitK{tok}"], m[f"RandInitWithinRbar{tok}"], m[f"RandInitCrossRbar{tok}"],
                     mci(m, f"RandInitGap{tok}"), NA, NA])
    by_label = {label: (t, tok) for t, tok, label in api.TASKS}
    for s in B.rows(MEDSUM):
        t, tok = by_label[s[0]]
        rows.append([lab(t), MEDENC15, m[f"MedEncExtPool{tok}K"], m[f"MedEncExtPool{tok}Same"],
                     m[f"MedEncExtPool{tok}Cross"], mci(m, f"MedEncExtPool{tok}Gap"), NA, NA])
    rand_files = [api.kv(rand[1], t).format(t=t) for t in rand[3]]  # Official ACDC source.
    return B.sources(UNET, FT, MEDSUM) + rand_files, rows


def checkpoint_tiers(b, api, B):
    m, rows = b.macros, []
    for t, tok, label in api.TASKS:
        x = B.row(XCKPT, label)
        rows.append([label, *x[1:7], m[f"Gap{tok}SameBundleOnly"], m[f"Gap{tok}DinoPairsAsCross"],
                     f"{m[f'Rbar{tok}DistinctCkptFamily']} ({m[f'NPairs{tok}DistinctCkptFamily']})",
                     m[f"Rbar{tok}CrossExclFamily"], m[f"Gap{tok}DistinctCkptFamily"],
                     m[f"Gap{tok}SameBundleVsCrossExclFamily"]])
    return B.sources(XCKPT) + [api.INDEP], rows


def pixel_full(b, api, B):
    m, rows = b.macros, []
    for t in api.PIXEL_ORDER:  # one split per row (the primary), so the plain task labels apply
        tok, p = api.TOK[t], B.row(PIXEL, api.LABEL[t])
        rows.append([api.LABEL[t], *p[1:8], mci(m, f"PixelGapFpOnly{tok}"),
                     mci(m, f"PixelGapFnOnly{tok}"), mci(m, f"PixelGapSameBundleVsCrossBundle{tok}")])
    return B.sources(PIXEL), rows


def poolsel_cells(m, B, tok, label):
    """low/splits, share % [CI], unsupported, support-valid share; tasks outside the rule get POOLSEL_NA text and --."""
    p = B.row(POOLSEL, label)
    if f"PoolSelFreq{tok}" in m:
        return [f"{m[f'PoolSelLow{tok}']}/{m[f'PoolSelSplits{tok}']}",
                f"{m[f'PoolSelFreq{tok}']}\\% {m[f'PoolSelFreq{tok}CI']}", p[3], p[2]]
    return [POOLSEL_NA.get(tok, "n/a"), NA, NA, NA]


def heldout_poolsel(b, api, B):
    m, rows, hn = b.macros, [], api.heldout_numbers
    for t, tok, label in api.TASKS:
        p = B.row(PRIMARY, label)
        rows.append([label, f"{m[f'K{tok}']}/{m[f'KAvail{tok}']}", p[6], p[7], primary_delta(m, tok), p[8],
                     *poolsel_cells(m, B, tok, label)])
    for t, tok, label in hn.HELDOUT:
        h = B.row(HELDOUT, label)
        rows.append([label, *h[1:6], *poolsel_cells(m, B, tok, label)])
    return B.sources(PRIMARY, HELDOUT, POOLSEL), rows


def item_full(b, api, B):
    return B.sources(ITEM), B.rows(ITEM)


TABLES = (("table_app_estimator_variants.tex", estimator_variants), ("table_app_failure_events_p33.tex", failure_events),
          ("table_app_controls.tex", controls), ("table_app_checkpoint_tiers.tex", checkpoint_tiers),
          ("table_app_pixel_overlap_full.tex", pixel_full), ("table_app_heldout_poolsel.tex", heldout_poolsel),
          ("table_app_item_difficulty_full.tex", item_full))


# Display panels project already-generated cells; tuple indices stack related cells, never recalculate them.
PANELS = {
    "primary": ("table_primary_corrected.tex", (0, (5, 8), (6, 7), 1, 2, 3, 4), None, None),
    "rater_cup": ("table_per_rater.tex", (1, 2, (3, 4), 5, 6, (7, 8), 9), 0, 7),
    "rater_disc": ("table_per_rater.tex", (1, 2, (3, 4), 5, 6, (7, 8), 9), 7, 14),
    "pixel_main": ("table_app_pixel_overlap_full.tex", (0, 1, 2, 5, 6, 7), None, None),
    "pixel_types": ("table_app_pixel_overlap_full.tex", (0, 3, 4, 8, 9, 10), None, None),
    "floors": ("table_app_estimator_variants.tex", tuple(range(7)), None, None),
    "estimators_core": ("table_app_estimator_variants.tex", (0, 7, 8, 9), None, None),
    "estimators_adjusted": ("table_app_estimator_variants.tex", (0, 10, 11, 12, 13), None, None),
    "failure_gaps": ("table_app_failure_events_p33.tex", tuple(range(6)), None, None),
    "failure_pools": ("table_app_failure_events_p33.tex", (0, 6, 7, 8, 9), None, None),
    "item_effect": ("table_app_item_difficulty_full.tex", tuple(range(5)), None, None),
    "item_panels": ("table_app_item_difficulty_full.tex", (0, 5, 6, 7, 8, 9), None, None),
    "meff_fixed": ("table_app_meff.tex", tuple(range(5)), None, None),
    "meff_percentile": ("table_app_meff.tex", (0, 5, 6, 7, 8), None, None),
    "checkpoint_tiers": ("table_app_checkpoint_tiers.tex", tuple(range(7)), None, None),
    "checkpoint_grouping": ("table_app_checkpoint_tiers.tex", (0, 7, 8, 9, 10, 11, 12), None, None),
    "heldout": ("table_app_heldout_poolsel.tex", tuple(range(6)), None, None),
    "pool_selection": ("table_app_heldout_poolsel.tex", (0, 6, 7, 8, 9), None, None),
    "controls_readout": ("table_app_controls.tex", tuple(range(7)), 0, 8),
    "controls_other_summary": ("table_app_controls.tex", (0, 1, 2, (3, 4), 5), 8, None),
    "controls_ft_contrast": ("table_app_controls.tex", (0, 1, 6, 7), 8, 14),
    "failure_thresholds_stacked": ("table_app_failure_thresholds.tex", (0, 1, 2, (3, 4), 5, 6, 7), None, None),
    "tau_sweep": ("table_app_tau_sweep.tex", tuple(range(5)), None, None),
    "medical": ("table_medenc_pool.tex", tuple(range(6)), None, None),
    "medical_pool_mutual": ("table_medenc_pool_summary.tex", (0, 1, 2), None, None),
    "medical_pool_extended": ("table_medenc_pool_summary.tex", (0, 3, 4), None, None),
}


def readable_panels(b, api, files):
    out = {}
    for name, (source, columns, start, stop) in PANELS.items():
        original = files[source][1][start:stop]
        projected = [[r[i] if isinstance(i, int) else r"\makecell[c]{" + r"\\".join(r[j] for j in i) + "}"
                      for i in columns] for r in original]
        if name == "primary":
            for row in projected:
                # Highlight the dependence gap and its primary resampling-unit interval.
                for col in (3, 5 if row[0] in ("ACDC LV", "BraTS FG") else 4):
                    row[col] = r"\cellcolor{TableFocus}" + row[col]
        text = api.table(b, files[source][0], f"Display panel {name}; source columns {columns}", projected,
                         [f"verbatim cells from {source}; row slice [{start}:{stop}]"], [REQUIRES])
        if name in ("controls_readout", "tau_sweep", "medical"):
            lines, previous, i = [], None, 0
            for line in text.splitlines():
                if line and not line.startswith(("%", r"\addlinespace")):
                    row = original[i]
                    group = ("FT" if row[1].startswith("FT,") else row[1]) if name.startswith("controls_") else row[0]
                    if previous is not None and group != previous:
                        lines[-1] = r"\addlinespace[4pt]"
                    previous, i = group, i + 1
                lines.append(line)
            text = "\n".join(lines) + "\n"
        out[f"panel_{name}.tex"] = text
    return out


def build(b, api, files) -> dict[str, str]:
    B, out = Bodies(api, files), {}
    for name, fn in TABLES:
        rels, rows = fn(b, api, B)
        out[name] = (list(dict.fromkeys(rels)), rows)
    return readable_panels(b, api, {**files, **out})
