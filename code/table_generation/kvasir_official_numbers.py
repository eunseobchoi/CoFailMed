"""Official Kvasir primary and content-group sensitivity numbers."""
from __future__ import annotations

KO = "results/kvasir_official"
SUMMARY = KO + "/KVASIR_OFFICIAL_SUMMARY.json"
PRIMARY_WC = "results/primary/stats/within_cross/kvasir.json"
XCKPT = KO + "/xckpt/DISTINCT_CHECKPOINT_TABLE.json"
MEDENC = KO + "/medenc/MEDENC_TABLE.json"
UNET_WC = KO + "/unet/stats/within_cross/kvasir.json"
P = "KvasirOfficial"
DICE_FAMILY = {"primary": "Primary", "xckpt": "Xckpt", "medenc": "MedEnc", "unet": "UnetSkip"}


def split(b, api):
    rel = SUMMARY
    names = [P + n for n in ("TrainN", "TestN", "NearDupGroups", "NearDupAllInTrain", "TestOverlapGrouped",
                             "SplitCommit", "Overlap")]
    d, s = b.load(rel), ("split",)
    for n, key in zip(names[:3], ("train_n", "test_n", "near_duplicate_groups")):  # Overlap is also emitted below
        b.count(n, d, rel, *s, key)
    b.value(names[3], lambda: {True: "yes", False: "no"}[api.dig(d, rel, *s, "near_duplicate_groups_all_in_train")])
    b.count(names[4], d, rel, *s, "test_overlap_with_grouped_v2_test")
    b.count(P + "Overlap", d, rel, *s, "test_overlap_with_grouped_v2_test")
    b.value(names[5], lambda: api.tex(str(api.dig(d, rel, *s, "provenance", "commit"))[:7]))


def primary(b, api):
    rel = PRIMARY_WC
    names = [P + n for n in ("Gap", "GapCaseCI", "GapPrimaryCI", "GapHierCI", "WithinRbar", "CrossRbar", "K",
                             "KAvail", "NTest", "Dropped")]
    d = b.load(rel)
    b.value(names[0], lambda: api.f3(api.wc_gap(d, rel)))
    for n, key in zip(names[1:4], ("case_bootstrap_95ci_gap", "primary_bootstrap_95ci_gap", "hierarchical_bootstrap_95ci_gap")):
        b.ci(n, d, rel, key)
    b.num3(names[4], d, rel, "within_rbar")
    b.num3(names[5], d, rel, "cross_rbar")
    b.value(names[6], lambda: api.fint(len(api.dig(d, rel, "fms_kept_after_floor"))))
    b.value(names[7], lambda: api.fint(len(api.dig(d, rel, "fms_available_before_floor"))))
    b.count(names[8], d, rel, "n_test")
    b.value(names[9], lambda: ", ".join(api.tex(x) for x in sorted(set(api.dig(d, rel, "fms_available_before_floor"))
                                                                   - set(api.dig(d, rel, "fms_kept_after_floor")))) or "none")


def xckpt(b, api):
    rel, r = XCKPT, ("rows", "kvasir")
    keys = (("K", "K_after_floor", "count"), ("Bundles", "n_bundles_before_floor", "count"),
            ("SameBundleRbar", "same_bundle_rbar", "num"), ("SameFamilyRbar", "same_family_ckpt_rbar", "num"),
            ("CrossFamilyRbar", "cross_family_rbar", "num"), ("Gap", "ckpt_gap", "num"),
            ("GapCI", "ckpt_gap_ci95", "ci"), ("GapPrimaryPool", "primary11_ckpt_gap", "num"),
            ("GapPrimaryPoolCI", "primary11_ckpt_gap_ci95", "ci"))
    d = b.load(rel)
    for n, key, kind in keys:
        {"count": b.count, "num": b.num3, "ci": b.ci}[kind](P + "Xckpt" + n, d, rel, *r, key)
    if b.lookup([], d, rel, "extra_cells_missing"):
        b.notes.append(f"{rel}: extra_cells_missing is not empty")


def medenc(b, api):
    rel, t = MEDENC, ("tasks", "kvasir")
    d, m = b.load(rel), P + "MedEnc"
    b.num3(m + "IncumbentCross", d, rel, *t, "incumbent_cross", "estimate")
    pm = (*t, "pooled_medical")
    b.num3(m + "PooledToIncumbent", d, rel, *pm, "newcomer_to_incumbent", "estimate")
    b.ci(m + "PooledToIncumbentCI", d, rel, *pm, "newcomer_to_incumbent", "ci95")
    b.num3(m + "PooledDiff", d, rel, *pm, "diff_vs_incumbent_cross", "estimate")
    b.ci(m + "PooledDiffCI", d, rel, *pm, "diff_vs_incumbent_cross", "ci95")
    b.num3(m + "ExtPoolGap", d, rel, *t, "pool15", "gap", "estimate")
    b.ci(m + "ExtPoolGapCI", d, rel, *t, "pool15", "gap", "ci95")
    b.count(m + "ExtPoolK", d, rel, *t, "pool15", "n_bundles")
    b.num3(m + "Mutual", d, rel, *t, "medical_mutual", "estimate")
    b.ci(m + "MutualCI", d, rel, *t, "medical_mutual", "ci95")
    for e in b.lookup({}, d, rel, *t, "per_encoder"):
        pe, pre = (*t, "per_encoder", e), m + api.camel(e)
        b.num3(pre + "MeanDice", d, rel, *pe, "mean_dice")
        b.num3(pre + "ToIncumbent", d, rel, *pe, "newcomer_to_incumbent", "estimate")
        b.ci(pre + "ToIncumbentCI", d, rel, *pe, "newcomer_to_incumbent", "ci95")
        b.num3(pre + "Diff", d, rel, *pe, "diff_vs_incumbent_cross", "estimate")
        b.ci(pre + "DiffCI", d, rel, *pe, "diff_vs_incumbent_cross", "ci95")


def unet(b, api):
    rel = UNET_WC
    names = [P + "UnetSkip" + n for n in ("K", "WithinRbar", "CrossRbar", "Gap", "GapCI")]
    d = b.load(rel)
    b.value(names[0], lambda: api.fint(len(api.dig(d, rel, "fms_kept_after_floor"))))
    b.num3(names[1], d, rel, "within_rbar")
    b.num3(names[2], d, rel, "cross_rbar")
    b.value(names[3], lambda: api.f3(api.wc_gap(d, rel)))
    b.ci(names[4], d, rel, "primary_bootstrap_95ci_gap")


def grouped_sens(b, api):
    """Kvasir content-group split sensitivity values."""
    sfx = "KvasirGroupedSens"
    rel = api.GROUPED_ROOT + "/stats/within_cross/kvasir.json"
    d = b.load(rel)

    b.value("Gap" + sfx, lambda: api.f3(api.wc_gap(d, rel)))
    for n, key in (("CI", "primary_bootstrap_95ci_gap"), ("CaseCI", "case_bootstrap_95ci_gap"),
                   ("HierCI", "hierarchical_bootstrap_95ci_gap")):
        b.ci("Gap" + sfx + n, d, rel, key)
    b.value("K" + sfx, lambda: api.fint(len(api.dig(d, rel, "fms_kept_after_floor"))))
    b.num3("WithinRbar" + sfx, d, rel, "within_rbar")
    b.num3("CrossRbar" + sfx, d, rel, "cross_rbar")
    b.count("NTest" + sfx, d, rel, "n_test")
    x, r = b.load(api.XCKPT), ("rows", "kvasir")
    b.num3("XckptGap" + sfx, x, api.XCKPT, *r, "ckpt_gap")
    b.ci("XckptGap" + sfx + "CI", x, api.XCKPT, *r, "ckpt_gap_ci95")
    b.count("XckptK" + sfx, x, api.XCKPT, *r, "K_after_floor")
    b.num3("XckptGapPrimaryPool" + sfx, x, api.XCKPT, *r, "primary11_ckpt_gap")
    b.ci("XckptGapPrimaryPool" + sfx + "CI", x, api.XCKPT, *r, "primary11_ckpt_gap_ci95")
    m, t = b.load(api.MEDENC), ("tasks", "kvasir")
    b.num3("MedEncPooledKvasirDiffGroupedSens", m, api.MEDENC, *t, "pooled_medical", "diff_vs_incumbent_cross", "estimate")
    b.ci("MedEncPooledKvasirDiffGroupedSensCI", m, api.MEDENC, *t, "pooled_medical", "diff_vs_incumbent_cross", "ci95")
    b.num3("MedEncExtPoolKvasirGapGroupedSens", m, api.MEDENC, *t, "pool15", "gap", "estimate")
    b.ci("MedEncExtPoolKvasirGapGroupedSensCI", m, api.MEDENC, *t, "pool15", "gap", "ci95")
    b.count("MedEncExtPoolKvasirKGroupedSens", m, api.MEDENC, *t, "pool15", "n_bundles")
    px, pr = api.PIXEL, None
    for row in b.lookup([], b.load(px), px, "tasks"):
        pr = row if isinstance(row, dict) and row.get("task") == "kvasir" else pr
    u_ = ("floor", "err_uncond")
    b.num3("PixelGap" + sfx, pr, px, *u_, "gap_same_vs_cross_family", "gap")
    b.ci("PixelGap" + sfx + "CI", pr, px, *u_, "gap_same_vs_cross_family", "gap_ci")
    for key, name in api.PIXEL_STRATA:
        b.num3(f"Pixel{name}{sfx}", pr, px, *u_, key, "mean")
    urel = api.SECONDARY_FAMILIES[0][1].format(t="kvasir")
    names = ["UnetSkipGap" + sfx, "UnetSkipGap" + sfx + "CI", "UnetSkipK" + sfx]
    u = b.load(urel)
    b.value(names[0], lambda: api.f3(api.wc_gap(u, urel)))
    b.ci(names[1], u, urel, "primary_bootstrap_95ci_gap")
    b.value(names[2], lambda: api.fint(len(api.dig(u, urel, "fms_kept_after_floor"))))


def dice(b, api):
    """Per-bundle official mean Dice from the recorded summary JSON."""
    rel = SUMMARY
    d = b.load(rel)
    fams = b.lookup({}, d, rel, "families")
    for fam, tok in DICE_FAMILY.items():
        for fm in sorted(((fams.get(fam) or {}).get("bundles") or {})):
            k, name = ("families", fam, "bundles", fm), f"{P}Dice{tok}{api.camel(fm)}"
            off = (((fams[fam]["bundles"][fm]) or {}).get("official"))
            b.num3(name, d, rel, *k, "official", "mean_dice") if off else b.put(name, "--")


def build(b, api):
    split(b, api)
    primary(b, api)
    xckpt(b, api)
    medenc(b, api)
    unet(b, api)
    grouped_sens(b, api)
    dice(b, api)
