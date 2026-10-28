"""Held-out task and calibration-selected pool tables."""
from __future__ import annotations

HELDOUT = (("isic2018", "Isic", "ISIC-2018"), ("msd_hippocampus", "Hippocampus", "MSD Hippocampus"),  # display labels as in the text
           ("msd_spleen", "Spleen", "MSD Spleen"), ("msd_heart", "Heart", "MSD Heart"))
HELDOUT_WC = "results/heldout/stats/within_cross/{t}.json"
NO_POOL = "results/heldout/stats/no_pool/{t}.json"  # compute_all raised: empty pool after floor
ISIC_OFFICIAL_ROOT = "results/heldout/isic_official"


def wc_rel(t: str, pattern: str = HELDOUT_WC) -> str:
    """Stats path for a held-out task; ISIC routed to the official-split subfolder for the public cohort."""
    if t == "isic2018":
        return pattern.replace("results/heldout/", ISIC_OFFICIAL_ROOT + "/").format(t=t)
    return pattern.format(t=t)
NO_POOL_CELL = r"no bundle $\geq$ floor"  # Recorded empty pool.
POOLSEL = "results/heldout/pool_selection/L2PS_low_meff.json"  # seven-task selection protocol
# (official Kvasir + official ACDC 100/50) and the official ISIC-2018 Task 1 train/test held-out row.
# Held-out tasks excluded from pool selection by the protocol.
EXTRA = (("msd_spleen", "Spleen", "MSD Spleen"), ("msd_heart", "Heart", "MSD Heart"))
SPLEEN_FG = "results/heldout/spleen_pred_foreground/spleen_pred_foreground.json"  # Predicted-foreground replay
L2PS_PRIMARY = ("kvasir", "acdc_lv", "brats_wt", "riga_cup", "riga_disc")  # + ISIC, Hippocampus = the 7 pool-selection tasks


def l2ps(api):
    """(json key, token, label) of the 7 pool-selection tasks: primary rows in the builder's TASKS order, then the two held-out."""
    return [(t, api.TOK[t], api.LABEL[t]) for t in api.in_task_order(L2PS_PRIMARY)] + list(HELDOUT[:2])

# Split unit per held-out task (dataset protocol facts from the official-source comparison, not results):
# caption-note macros HeldoutSplitUnit<T>. None of these splits is patient-level.
SPLIT_UNIT = {
    "Isic": "image-level analysis of the official ISIC-2018 Task 1 training (2594) / test (1000) split; "
            "partial patient identifiers are available",
    "Hippocampus": "volume-level, not patient-level, 208/52 over the 260 training volumes: the 390 MSD volumes are "
                   "left/right crops of 195 subjects and no subject mapping is released",
    "Spleen": "volume-level random split into 33 training and 8 test volumes (seed 42)",
    "Heart": "volume-level random split into 16 training and 4 test volumes (seed 42)",
}


def _pct(api, x) -> str:
    return f"{100 * api.num(x):.1f}"


def _pct_ci(api, v) -> str:
    api.fci(v)  # validates a finite ordered pair
    return f"[{_pct(api, v[0])}, {_pct(api, v[1])}]"


def heldout_rows(b, api) -> tuple[list[list[str]], list[str]]:
    """Table rows and, per task, the stats file actually read (within_cross, or the recorded no_pool status)."""
    rows, rels = [], []
    for t, tok, label in HELDOUT:
        b.put(f"HeldoutSplitUnit{tok}", SPLIT_UNIT[tok])
        rel = wc_rel(t)
        names = [f"Heldout{q}{tok}" for q in ("K", "KAvail", "WithinRbar", "CrossRbar", "Gap")]
        names += [f"HeldoutGap{tok}CI", f"HeldoutNTest{tok}", f"HeldoutPoolStatus{tok}"]
        no_pool = not (b.root / rel).is_file() and (b.root / wc_rel(t, NO_POOL)).is_file()
        if no_pool:
            rel = wc_rel(t, NO_POOL)  # recorded empty pool; within/cross undefined

        rels.append(rel)
        d = b.load(rel)
        kept = b.lookup([], d, rel, "fms_kept_after_floor")
        if no_pool and kept:
            raise api.Fatal(f"{rel}: no-pool record contains retained bundles")
        functional = len(kept) >= 2
        k = b.value(names[0], lambda: api.fint(len(api.dig(d, rel, "fms_kept_after_floor"))))
        avail = b.value(names[1], lambda: api.fint(len(api.dig(d, rel, "fms_available_before_floor"))))
        if functional:
            within, cross = (b.num3(names[i], d, rel, key) for i, key in ((2, "within_rbar"), (3, "cross_rbar")))
            gap = b.value(names[4], lambda: api.f3(api.wc_gap(d, rel)))
            ci = b.ci(names[5], d, rel, "primary_bootstrap_95ci_gap")
        else:
            within, cross, gap, ci = (b.put(names[i], "n/a") for i in (2, 3, 4, 5))
            b.notes.append(f"{label}: no functional pool after the 0.30 floor (K={len(kept)}); gap not defined")
        n = b.count(names[6], d, rel, "n_test")
        status = b.put(names[7], "functional" if functional else "no functional pool")
        if no_pool:
            rows.append([label, f"{k}/{avail}", "--", "--", "--", n, NO_POOL_CELL])
        else:
            rows.append([label, f"{k}/{avail}", within, cross, f"{gap} {ci}", n, status])
    return rows, rels


def _task_names(tok):
    return [f"PoolSel{q}{tok}" for q in ("Low", "Splits", "Unsupported")] + \
           [f"PoolSelFreq{tok}", f"PoolSelFreq{tok}CI", f"PoolSelFreq{tok}SupportValid"]


def _task_row(b, api, d, rel, t, tok, label):
    names, s = _task_names(tok), ("per_task", t)
    low = b.count(names[0], d, rel, *s, "outcome_counts", "detected_low_M_eff")
    n = b.count(names[1], d, rel, *s, "primary_low_M_eff_fraction_all_splits", "denominator")
    uns = b.count(names[2], d, rel, *s, "outcome_counts", "unsupported")
    freq = b.value(names[3], lambda: _pct(api, api.dig(d, rel, *s, "primary_low_M_eff_fraction_all_splits", "value")))
    ci = b.value(names[4], lambda: _pct_ci(api, api.dig(d, rel, *s, "primary_low_M_eff_fraction_all_splits", "ci95")))
    fsv_value = lambda: api.opt(d, rel, *s, "low_M_eff_fraction_support_valid", "value")  # noqa: E731 null: no support
    fsv = b.value(names[5], lambda: "n/a" if fsv_value() is None else _pct(api, fsv_value()))
    return [label, f"{low}/{n} = {freq}\\% {ci}", f"{fsv}\\%", uns]


def pool_rows(b, api) -> list[list[str]]:
    d = b.load(POOLSEL)
    expected = {t for t, _, _ in l2ps(api)}
    if set(d["tasks"]) != expected:
        raise api.Fatal("pool selection must contain the seven paper tasks")
    rows = [_task_row(b, api, d, POOLSEL, t, tok, label) for t, tok, label in l2ps(api)]
    path = ("all_task_split_counts",)
    for name, key in (("Low", "detected_low_M_eff"), ("Splits", "n_task_splits"), ("Unsupported", "unsupported")):
        b.count("PoolSelOverall" + name, d, POOLSEL, *path, key)
    for name, key in (("Freq", "low_M_eff_fraction_all_task_splits"), ("FreqSupportValid", "low_M_eff_fraction_support_valid")):
        b.value("PoolSelOverall" + name, lambda key=key: _pct(api, api.dig(d, POOLSEL, *path, key)))
    b.put("PoolSelOverallTasks", api.fint(len(expected)))
    for task, token, label in EXTRA:
        status = "degenerate" if task == "msd_spleen" else "no functional pool"
        b.put("PoolSelStatus" + token, status)
        rows.append([label, status, "--", "--"])
    return rows


def spleen_foreground(b, api) -> None:
    """SpleenPred* counts from the gated replay (predicted foreground pixels per head and test case)."""
    rel = SPLEEN_FG
    names = ("SpleenPredHeads", "SpleenPredHeadsAllEmpty", "SpleenPredCaseHeads", "SpleenPredNonemptyCaseHeads",
             "SpleenTestCases", "SpleenGtEmptyCases", "SpleenEmptyGtDiceOnePairs", "SpleenNonemptyGtDiceZeroPairs")
    d = b.load(rel)

    def counts():
        if api.dig(d, rel, "gt_audit_match") is not True:
            raise api.Missing(f"{rel}:gt_audit_match is not true")
        heads = [api.dig(d, rel, "heads", h, "pred_pixels") for h in api.dig(d, rel, "heads")]
        dice = [api.dig(d, rel, "heads", h, "dice") for h in api.dig(d, rel, "heads")]
        cases = api.dig(d, rel, "case_ids")
        gt = {c: api.dig(d, rel, "per_case", c, "gt_pixels") for c in cases}
        return (len(heads), sum(not any(h.values()) for h in heads), len(heads) * len(cases),
                sum(v > 0 for h in heads for v in h.values()), len(cases), sum(g == 0 for g in gt.values()),
                sum(h[c] == 1 for h in dice for c in cases if gt[c] == 0),
                sum(h[c] == 0 for h in dice for c in cases if gt[c] > 0))
    for i, n in enumerate(names):
        b.value(n, lambda i=i: api.fint(counts()[i]))


def build(b, api):
    (h_rows, h_rels), p_rows = heldout_rows(b, api), pool_rows(b, api)
    spleen_foreground(b, api)
    return {"table_heldout_rows.tex": (h_rels, h_rows),
            "table_pool_selection_freq.tex": ([POOLSEL], p_rows)}
