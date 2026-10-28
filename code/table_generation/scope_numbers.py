"""Generate scoped pipeline and empty-error-mask diagnostics from verified JSON."""

CONTROLS = "results/controls/results.json"
AMOS_AUDIT = "results/controls/amos_audit.json"


def input_paths():
    return [CONTROLS, AMOS_AUDIT]


def build(b, api):
    files, empty_rows = {}, []
    for task in api.PIXEL_ORDER:
        rel = api.kv(api.PIXEL, task)
        data = {r["task"]: r for r in b.load(rel)["tasks"]}
        tok = api.TOK[task]
        for block, name in (("fp", "Fp"), ("fn", "Fn")):
            pre = f"PixelEmpty{name}{tok}"
            counts = []
            for group, gt in (("same", "Same"), ("cross_family", "Cross")):
                n = b.count(pre + gt, data, rel, task, "floor", block + "_uncond", group, "n_both_empty")
                total = b.put(pre + gt + "Total", str(data[task]["n_cases"] * data[task]["floor"]["n_pairs"][group]))
                counts.append(f"{n}/{total}")
            path = (task, "floor", block + "_cond", "gap_same_vs_cross_family")
            gap = b.num3(pre + "Gap", data, rel, *path, "gap")
            ci = b.ci(pre + "GapCI", data, rel, *path, "gap_ci")
            empty_rows.append([api.LABEL[task], block.upper(), *counts, f"{gap} {ci}"])
        # The full XOR-error comparison has no both-empty case-pairs.
        assert all(data[task]["floor"]["err_uncond"][g]["n_both_empty"] == 0 for g in ("same", "cross_family"))
    files["table_pixel_empty.tex"] = api.table(b, list(dict.fromkeys(api.kv(api.PIXEL, t) for t in api.PIXEL_ORDER)),
        "Task & Error & both-empty/total same case-pairs & cross case-pairs & conditional gap [CI]", empty_rows)

    d, rows = b.load(api.AMOS), []
    audit = b.load(AMOS_AUDIT)
    b.put("AmosOrgans", str(len(audit["common_organs"])))
    for key, name in (("undefined_case_organ_cells_per_system", "MissingCells"),
                      ("minimum_finite_organs_per_case", "MinOrgans"),
                      ("maximum_finite_organs_per_case", "MaxOrgans")):
        b.count("Amos" + name, audit, AMOS_AUDIT, "missingness", key)
    for mod, mtok in (("ct", "CT"), ("mri", "MRI")):
        pre = "Amos" + mtok
        rows.append([mtok, b.macros[pre + "N"], *[f'{b.macros[pre+n]} {b.macros[pre+n+"CI"]}' for n in ("SameR", "CrossR", "Gap", "Meff")]])
        path = ("by_modality", mod, "phase2_secondary", "nonpearson_P33")
        b.num3(pre + "Threshold", d, api.AMOS, *path, "threshold", "value")
        for metric, name in api.L1A_METRICS:
            mp = (*path, "estimators", metric)
            val = b.num3(pre + name + "Gap", d, api.AMOS, *mp, "point", "gap_same_minus_cross")
            ci = b.ci(pre + name + "GapCI", d, api.AMOS, *mp, "case_bootstrap_95ci", "gap_same_minus_cross", "ci95")
    files["panel_amos.tex"] = api.table(b, [api.AMOS], "Modality & N & same [CI] & cross [CI] & gap [CI] & M_eff [CI]", rows)

    d, rows = b.load(CONTROLS), []
    for key, tok, task, label in (("unet_cup", "UnetCup", "RIGA Cup", "UNet-skip"),
                                ("unet_disc", "UnetDisc", "RIGA Disc", "UNet-skip"),
                                ("nnunet_cup100", "NnunetCup", "RIGA Cup", "nnU-Net")):
        for metric, name in api.L1A_METRICS:
            pre, path = "Pipeline" + tok + name, ("E4", key, "estimators", metric)
            val = b.num3(pre + "Gap", d, CONTROLS, *path, "point", "gap_same_minus_cross")
            ci = b.ci(pre + "GapCI", d, CONTROLS, *path, "case_bootstrap_95ci", "gap_same_minus_cross", "ci95")

    rows = []
    for task, tok in (("kvasir", "Kvasir"), ("riga_cup", "Cup")):
        for arm, atok, label in ((None, "Frozen", "Frozen"), ("lr1e-4", "FtFour", r"FT $10^{-4}$"), ("lr1e-5", "FtFive", r"FT $10^{-5}$")):
            key = f"ft_{task}_{arm}" if arm else f"frozen_{task}"
            pre, path = "Scope" + tok + atok, ("E1", key, "primary_P33")
            threshold = b.num3(pre + "Threshold", d, CONTROLS, *path, "threshold", "value")
            val = b.num3(pre + "Meff", d, CONTROLS, *path, "M_eff", "value")
            ci = b.ci(pre + "MeffCI", d, CONTROLS, *path, "M_eff", "ci95")
            supported = b.count(pre + "Supported", d, CONTROLS, *path, "support", "n_pools_supported")
            total = b.count(pre + "Total", d, CONTROLS, *path, "support", "n_pools_total")
            diff = "--"
            if arm:
                pp = ("paired_comparisons", key + "_minus_frozen")
                dv = b.num3(pre + "Difference", d, CONTROLS, *pp, "difference")
                dc = b.ci(pre + "DifferenceCI", d, CONTROLS, *pp, "paired_ci95")
                diff = r"\multicolumn{1}{>{\columncolor{TableFocus}}c}{\estci{" + f"{dv} {dc}" + "}}"
            rows.append([api.LABEL[task], label, threshold, f"{val} {ci}", diff, f"{supported}/{total}"])
    files["table_adaptation_meff.tex"] = api.table(b, [CONTROLS], "Task & Recipe & P33 & M_eff [CI] & FT minus frozen [CI] & supported/total", rows)
    return files
