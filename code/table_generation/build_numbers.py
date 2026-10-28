
"""Generate paper numbers from recorded result JSON (stdlib only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scope_numbers
import consolidated_numbers  # Current panels reuse the rows and macros below.
import gpu_hours_numbers  # noqa: E402  GPU wall-clock per experiment family (own module)
import heldout_numbers  # noqa: E402  Held-out rows and pool-selection frequencies (own module)
import kvasir_official_numbers  # noqa: E402  Kvasir official primary and content-group split sensitivity (own module)
import mask_align_numbers  # noqa: E402  Nearest-exact mask-alignment sensitivity (own module)

CR = "results/primary"
GROUPED_ROOT = "results/kvasir_grouped"
ACDC_OFFICIAL = True
PP = CR + "/postproc"
WC = CR + "/stats/within_cross/{t}.json"
MEFF = CR + "/stats/m_eff/{t}.json"
FACTS = CR + "/reconciliation/FACTS.json"
INDEP = CR + "/independent_check/values.json"
CALIB = PP + "/compute_all/out/calibration_split/{t}.json"
SUBJ = PP + "/compute_all/out/subject_level/{t}.json"
AUDIT = PP + "/audit_sensitivity/audit_sensitivity.json"
QCP = PP + "/quality_controlled_pairs/quality_controlled_pairs.json"
FAIL = PP + "/failure_event_summary/failure_event_summary.json"
ITEM = PP + "/item_difficulty_robustness/item_difficulty_robustness.json"
TAU = PP + "/tau_sweep/tau_sweep.json"
L1A = PP + "/l1a_metrics/L1A_alternative_diversity.json"
E1 = PP + "/e1_meff_configs/out/metrics.json"
VIT = PP + "/vit_arch_diagnostic/vit_arch_diagnostic.json"
PER_RATER = "results/raters/L1D_per_rater_{part}.json"
XCKPT = "results/checkpoints/DISTINCT_CHECKPOINT_TABLE.json"
XCKPT_CI_NOTE = ("95\\% CIs: ACDC LV and BraTS FG resample patients (cluster bootstrap, B=2000; slice-level case "
                 "CIs in \\texttt{gap\\_ci\\_case}); other tasks resample images (case bootstrap, B=2000).")
MEDENC = "results/medical/MEDENC_TABLE.json"
PIXEL = "results/spatial/PIXEL_OVERLAP_TABLE.json"
AMOS = "results/amos/metrics.json"
NNUNET = "results/nnunet/primary_dice/summary_{t}.json"  # Summary of primary-Dice (estimators.dice_per_case) re-scored traces
NNUNET_ACDC_OFFICIAL = "results/nnunet/acdc_official/primary_dice/summary_acdc_lv_official.json"
# ACDC macros use the official 100/50 patient split.
NNUNET_FILE = {"riga_cup": NNUNET.format(t="riga_cup"), "acdc_lv": NNUNET_ACDC_OFFICIAL}
NNUNET_POOL = "M5_seed13_primary"  # E1 fixed pool: the 5-fold ensemble of seed 13
KVASIR_SPLIT = "code/data/splits/kvasir_official_split.json"

# (json key, macro token, display label): THE task label/order map. Every generated body prints these labels in this
# order (held-out tasks after the primary rows; the sub-modules read TASKS/LABEL/in_task_order from here). Rows that mix
# Kvasir splits are labelled "Kvasir (official)" / "Kvasir (content-group split[, sensitivity])" (ft_rows, consolidated).
TASKS = (("kvasir", "Kvasir", "Kvasir"), ("acdc_lv", "Acdc", "ACDC LV"), ("brats_wt", "Brats", "BraTS FG"),
         ("riga_cup", "Cup", "RIGA Cup"), ("riga_disc", "Disc", "RIGA Disc"))
TOK = {t: tok for t, tok, _ in TASKS}
LABEL = {t: label for t, _, label in TASKS}


def in_task_order(keys) -> tuple[str, ...]:
    """The task json keys in `keys`, in the canonical TASKS order."""
    return tuple(t for t, _, _ in TASKS if t in set(keys))


VIT_ORDER = in_task_order(TOK)
PIXEL_ORDER = in_task_order(("kvasir", "acdc_lv", "riga_cup", "riga_disc"))
NNUNET_TASKS = in_task_order(("acdc_lv", "riga_cup"))
PIXEL_STRATA = (("same_bundle", "SameBundle"), ("same_family_ckpt", "SameFamilyCkpt"), ("same", "SameGroup"),
                ("cross_family", "CrossFamily"), ("cross_bundle", "CrossBundle"))
FLOORS = (0.0, 0.2, 0.25, 0.3, 0.35, 0.4)  # audit_sensitivity floor grid (protocol constants, not results)
TAUS = (0.7, 0.75, 0.8, 0.85, 0.9)  # tau_sweep grid
# L1A "kappa_never_fail" convention -> manuscript text (\KappaNeverFailConvention); unknown values are fatal (KeyError).
KAPPA_NEVER_FAIL_TEXT = {"official": r"both-never-fail pairs scored $\kappa=1$ as in the official implementation",
                         "exclude": r"both-never-fail pairs excluded ($\kappa$ undefined)"}
L1A_METRICS = (("kappa", "Kappa"), ("q_statistic", "Q"), ("disagreement_rate", "Disagreement"),
               ("double_fault_rate", "DoubleFault"))
E1_THRESHOLDS = (("P33", "PThirtyThree"), ("fixed_0.85", "FixedEightyFive"))
RATER_CONVENTIONS = (("primary_gt_specific_floor", "Gt"), ("fixed_consensus_pool_sensitivity", "Fixed"))
ANNOTATIONS = ("rater_1", "rater_2", "rater_3", "rater_4", "rater_5", "rater_6", "consensus")
MEDENC_NAMES = {"retfound_vitl16": "RETFound", "phikon": "Phikon", "rad_dino": "RAD-DINO", "biomedparse": "BiomedParse"}
FAIL_PERCENTILES = (25, 33, 50)  # empirical failure-threshold percentiles in failure_event_summary.json
# Secondary families read from compute_all.py within_cross outputs: (family, stats file, macro prefix, tasks).
# Missing task inputs are reported explicitly rather than replaced by numbers.
SECONDARY_FAMILIES = (("unet-skip", "results/readout/stats/within_cross/{t}.json", "UnetSkip",
                       ("kvasir", "acdc_lv", "riga_cup", "riga_disc")),
                      ("randinit", "results/from_scratch/stats/within_cross/{t}.json", "RandInit",
                       ("acdc_lv", "riga_cup")))
# Primary 1x1 within/cross restricted to the nine multiscale bundles (compute_all.py on a symlink root of the
# corrected primary traces), the matched column of table_unet_skip.tex; macro names carry no "Rbar".
PRIMARY_NINE = ("primary-nine-bundle", "results/readout/primary_9bundle/stats/within_cross/{t}.json",
                "PrimaryNineBundle", SECONDARY_FAMILIES[0][3])
# UNet-skip sensitivity excludes the two CLIP bundles (9 bundles). The UNet-skip family
# above is the 11-bundle pool; its matched 1x1 column is the corrected primary itself (the primary pool is these 11 bundles;
# recomputed on a symlink root in results/readout/primary_11bundle, identical on every statistic key).
UNET_NINE = ("unet-skip-nine-bundle", "results/readout/stats_9bundle/within_cross/{t}.json",
             "UnetSkipNine", SECONDARY_FAMILIES[0][3])
# Official Kvasir sources for secondary-family rows.
KO = "results/kvasir_official"
KVASIR_OFFICIAL_FILES = {XCKPT: KO + "/xckpt/DISTINCT_CHECKPOINT_TABLE.json", MEDENC: KO + "/medenc/MEDENC_TABLE.json",
                         SECONDARY_FAMILIES[0][1]: KO + "/unet/stats/within_cross/{t}.json",
                         PRIMARY_NINE[1]: KO + "/primary_9bundle/stats/within_cross/{t}.json",
                         UNET_NINE[1]: KO + "/unet_9bundle/stats/within_cross/{t}.json",
                         PIXEL: "results/spatial/PIXEL_OVERLAP_TABLE_official.json"}


# Official ACDC 100/50 sources for secondary-family rows.
# MEDENC has no ACDC row on any split (kvasir, riga_cup, riga_disc only); the ViT diagnostic (VIT) is computed from the root.
OFF = "results/acdc_official"
ACDC_OFFICIAL_FILES = {XCKPT: OFF + "/xckpt/DISTINCT_CHECKPOINT_TABLE.json",
                       SECONDARY_FAMILIES[0][1]: OFF + "/unet/stats/within_cross/{t}.json",
                       SECONDARY_FAMILIES[1][1]: OFF + "/randinit/stats/within_cross/{t}.json",
                       PRIMARY_NINE[1]: OFF + "/acdc_9bundle/stats/within_cross/{t}.json",
                       UNET_NINE[1]: OFF + "/unet_9bundle/stats/within_cross/{t}.json",
                       PIXEL: "results/spatial/PIXEL_OVERLAP_TABLE_official_acdc.json"}


def kv(rel: str, t: str) -> str:
    """Use the official Kvasir/ACDC row; other tasks retain their named source."""
    if t == "kvasir":
        return KVASIR_OFFICIAL_FILES.get(rel, rel)
    if t == "acdc_lv":
        return ACDC_OFFICIAL_FILES.get(rel, rel)
    return rel


# End-to-end fine-tuning + 1x1 head, from compute_ft_table.py's FT_TABLE.json (both pre-declared encoder-lr arms,
# none selected). Macros \Ft<Quantity><Arm><Task>: Gap, Gap..CI, WithinRbar, CrossRbar, K, NCollapsed, MeanDice..<Enc>;
# \FrozenMatched{Gap,GapCI,K}<Arm><Task> (frozen 1x1 on the FT-kept bundles, same split root); \FtGapExclCollapsed<Arm><Task>{,CI}
# only where a collapsed head was excluded; \FtMinusFrozenGap<Arm><Task> + \FtMinusFrozenGapCI<Arm><Task> (paired, same case draws);
# \FtBelowFrozen<Arm><Task> (bundles with FT mean Dice below frozen). Arm: LrEMinusFour (1e-4) / LrEMinusFive (1e-5).
FT_TABLE = "results/finetuning/FT_TABLE.json"
FT_ARMS = (("lr1e-4", "LrEMinusFour", r"$10^{-4}$"), ("lr1e-5", "LrEMinusFive", r"$10^{-5}$"))
NAMING = r"""% One \newcommand per numeric value. Task tokens: Kvasir, Acdc, Brats, Cup, Disc.
% CI = 95% interval; PrimaryCI uses patients for ACDC/BraTS and images otherwise.
% Correlations, Dice and intervals use three decimals; counts use integers.
% GroupedSens identifies the Kvasir content-group split sensitivity."""
ONES = ("Zero One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve Thirteen Fourteen Fifteen Sixteen "
        "Seventeen Eighteen Nineteen").split()
TENS = ("", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety")


class Missing(Exception):
    """An expected file or key is absent, null or of the wrong type."""


class Fatal(Exception):
    """Unrecoverable input problem: malformed JSON, or anything missing in the primary layer."""


def words(n: int) -> str:
    if n >= 100:
        hundreds, rest = divmod(n, 100)
        return words(hundreds) + "Hundred" + (words(rest) if rest else "")
    if n < 20:
        return ONES[n]
    tens, ones = divmod(n, 10)
    return TENS[tens] + (ONES[ones] if ones else "")


def camel(name: str) -> str:
    return "".join(words(int(p)) if p.isdigit() else p.capitalize() for p in re.findall(r"[A-Za-z]+|\d+", name))


def num(x) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise Missing(f"not a finite number: {x!r}")
    return float(x)


def pct(x) -> str:
    return words(round(num(x) * 100))


def f3(x) -> str:
    s = f"{num(x):.3f}"
    return "0.000" if s == "-0.000" else s


def fci(v) -> str:
    if not (isinstance(v, list) and len(v) == 2):
        raise Missing(f"not a 2-element interval: {v!r}")
    lo, hi = num(v[0]), num(v[1])
    if lo > hi:
        raise Missing(f"interval lower bound above upper bound: {v!r}")
    return f"[{f3(lo)}, {f3(hi)}]"


def fint(x) -> str:
    if isinstance(x, bool) or not isinstance(x, int):
        raise Missing(f"not an integer count: {x!r}")
    return str(x)


def tex(s: str) -> str:
    return str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def keypath(keys) -> str:
    return "/".join(map(str, keys))


def dig(d, rel, *keys):
    if d is None:
        raise Missing(f"{rel} (file not found)")
    cur = d
    for i, k in enumerate(keys):
        try:
            cur = cur[k]
        except (KeyError, IndexError, TypeError):
            raise Missing(f"{rel}:{keypath(keys[:i + 1])}") from None
    if cur is None:
        raise Missing(f"{rel}:{keypath(keys)} is null")
    return cur


def opt(d, rel, *keys):
    """Like dig, but an explicit JSON null in the last key returns None (absent key is still Missing)."""
    parent = dig(d, rel, *keys[:-1]) if len(keys) > 1 else d
    if not isinstance(parent, dict) or keys[-1] not in parent:
        raise Missing(f"{rel}:{keypath(keys)}")
    return parent[keys[-1]]


def find(items, rel, desc, pred):
    for it in items if isinstance(items, list) else ():
        if isinstance(it, dict) and pred(it):
            return it
    raise Missing(f"{rel}:{desc} (no matching entry)")


def wc_gap(d, rel) -> float:
    within, cross, gap = (num(dig(d, rel, k)) for k in ("within_rbar", "cross_rbar", "gap"))
    if abs(gap - (within - cross)) > 1e-9:
        raise Missing(f"{rel}: gap != within_rbar - cross_rbar")
    return gap


def close(x, v) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and abs(x - v) < 1e-9


class Build:
    """Validated result inputs and the macro values derived from them."""

    def __init__(self, root: Path):
        self.root = root
        self.data, self.sha, self.macros = {}, {}, {}
        self.notes = []
        self.meff_tau = {}

    def load(self, rel):
        if rel not in self.data:
            path = self.root / rel
            if not path.is_file():
                raise Fatal(f"required input missing: {rel}")
            raw = path.read_bytes()
            try:
                self.data[rel] = json.loads(raw)
            except ValueError as exc:
                raise Fatal(f"malformed JSON in {rel}: {exc}") from None
            self.sha[rel] = hashlib.sha256(raw).hexdigest()
        return self.data[rel]

    def put(self, name, text):
        if not re.fullmatch(r"[A-Za-z]+", name) or name in self.macros:
            raise AssertionError(f"invalid or duplicate macro name: {name}")
        self.macros[name] = text
        return text

    def value(self, name, fn):
        try:
            return self.put(name, fn())
        except Missing as exc:
            raise Fatal(f"{name}: {exc}") from None

    def lookup(self, default, d, rel, *keys):
        return dig(d, rel, *keys)

    def cell(self, fn):
        return fn()

    def num3(self, name, d, rel, *keys):
        return self.value(name, lambda: f3(dig(d, rel, *keys)))

    def ci(self, name, d, rel, *keys):
        return self.value(name, lambda: fci(dig(d, rel, *keys)))

    def count(self, name, d, rel, *keys):
        return self.value(name, lambda: fint(dig(d, rel, *keys)))

    def text(self, name, d, rel, *keys):
        return self.value(name, lambda: tex(dig(d, rel, *keys)))


# Primary analyses
def primary_task(b, t, tok, label):
    rel = WC.format(t=t)
    d = b.load(rel)
    gap, within, cross = wc_gap(d, rel), num(dig(d, rel, "within_rbar")), num(dig(d, rel, "cross_rbar"))
    kept, avail = dig(d, rel, "fms_kept_after_floor"), dig(d, rel, "fms_available_before_floor")
    if not (isinstance(kept, list) and isinstance(avail, list) and set(kept) <= set(avail)):
        raise Fatal(f"{rel}: fms_kept_after_floor / fms_available_before_floor malformed")
    if "cluster_unit" not in d:
        raise Fatal(f"{rel}: missing key cluster_unit")
    clustered, dropped = d["cluster_unit"] is not None, sorted(set(avail) - set(kept))
    P = b.put
    row = [label, P(f"Gap{tok}Corrected", f3(gap)), b.ci(f"Gap{tok}CaseCI", d, rel, "case_bootstrap_95ci_gap"),
           b.ci(f"Gap{tok}ClusterCI", d, rel, "primary_bootstrap_95ci_gap") if clustered else "--",
           b.ci(f"Gap{tok}HierCI", d, rel, "hierarchical_bootstrap_95ci_gap"), P(f"K{tok}", str(len(kept))),
           P(f"WithinRbar{tok}", f3(within)), P(f"CrossRbar{tok}", f3(cross)), b.count(f"NTest{tok}", d, rel, "n_test")]
    b.ci(f"Gap{tok}PrimaryCI", d, rel, "primary_bootstrap_95ci_gap")
    P(f"KAvail{tok}", str(len(avail)))
    P(f"NDropped{tok}", str(len(dropped)))
    P(f"Dropped{tok}", ", ".join(tex(x) for x in dropped) or "none")
    for key, name in (("n_units", "NUnits"), ("n_within_pairs", "NWithinPairs"), ("n_cross_pairs", "NCrossPairs")):
        b.count(f"{name}{tok}", d, rel, key)
    return row, clustered, fint(dig(d, rel, "B"))


def primary_meff(b):
    for t, tok, _ in TASKS:
        rel = MEFF.format(t=t)
        d = b.load(rel)
        w, s = "Tau" + pct(dig(d, rel, "tau")), ("diverse_pools_summary",)
        b.meff_tau[t] = w
        b.num3(f"Meff{tok}{w}", d, rel, *s, "M_eff_mean")
        b.num3(f"Meff{tok}{w}CovExact", d, rel, *s, "M_eff_covariance_exact_mean")
        b.num3(f"RhoFail{tok}{w}", d, rel, *s, "rho_fail_mean")
        for key, name in (("n_valid_compositions", "NValid"), ("n_seed_compositions", "NSeedCompositions"),
                          ("n_ineligible_compositions", "NIneligible"), ("pool_size_fms", "PoolSize")):
            b.count(f"Meff{tok}{w}{name}", d, rel, *s, key)


def mean_dice(b):
    rel = CR + "/comparison/comparison.json"
    d = b.load(rel)
    for x in dig(d, rel, "bundles"):
        tok, enc = TOK.get(dig(x, rel, "task")), camel(dig(x, rel, "fm"))
        if tok is None:
            raise Fatal(f"{rel}: unknown task {x.get('task')!r}")
        b.num3(f"Dice{tok}{enc}", x, rel, "corrected_mean_dice")


def cohorts(b):
    d = b.load(FACTS)
    b.count("CellsCollected", d, FACTS, "A1_roster", "receipt_totals", "cells_collected")
    b.count("HeadsCollected", d, FACTS, "A1_roster", "receipt_totals", "heads_collected")
    for key, name in (("train_n", "KvasirSplitTrainN"), ("test_n", "KvasirSplitTestN")):
        b.count(name, d, FACTS, "C2", "kvasir", "split_file", key)
    kvasir_near_dups(b, d)
    for part in ("train", "test"):
        for unit in ("slices", "sessions", "patients"):
            b.count(f"Brats{part.capitalize()}{unit.capitalize()}", d, FACTS, "C3", "corrected_split", part, unit)
    b.count("BratsTrainTestPatientOverlap", d, FACTS, "C3", "corrected_split", "train_test_patient_overlap")


def kvasir_near_dups(b, facts):
    """Distinct near-duplicate groups from the split file FACTS names (path and sha256 must match FACTS)."""
    rec = ("C2", "kvasir", "split_file")
    if dig(facts, FACTS, *rec, "path") != KVASIR_SPLIT:
        raise Fatal(f"{FACTS}: C2 kvasir split_file.path is not {KVASIR_SPLIT}")
    split = b.load(KVASIR_SPLIT)
    if b.sha[KVASIR_SPLIT] != dig(facts, FACTS, *rec, "sha256"):
        raise Fatal(f"{KVASIR_SPLIT}: sha256 differs from FACTS C2 kvasir split_file.sha256")
    groups = dig(split, KVASIR_SPLIT, "near_duplicate_groups")
    if not (isinstance(groups, list) and all(isinstance(g, list) for g in groups)):
        raise Fatal(f"{KVASIR_SPLIT}: near_duplicate_groups is not a list of id lists")
    b.put("KvasirNearDupGroupsRaw", str(len(groups)))
    b.put("KvasirNearDupGroups", str(len({frozenset(g) for g in groups})))


def sensitivity(b):
    d = b.load(INDEP)
    for t, tok, _ in TASKS:
        b.num3(f"Gap{tok}NoFloor", d, INDEP, "corrected", t, "gap_no_floor")
        s = ("sensitivity", t)
        b.num3(f"Gap{tok}SameBundleOnly", d, INDEP, *s, "same_bundle_only__dino_pairs_dropped", "gap")
        b.num3(f"Gap{tok}DinoPairsAsCross", d, INDEP, *s, "same_bundle_only__dino_pairs_moved_to_cross", "gap")
        f = (*s, "vs_cross_excl_family")
        b.num3(f"Rbar{tok}DistinctCkptFamily", d, INDEP, *f, "distinct_ckpt_family_r")
        b.num3(f"Gap{tok}DistinctCkptFamily", d, INDEP, *f, "gap_family")
        b.count(f"NPairs{tok}DistinctCkptFamily", d, INDEP, *f, "n_family")
        b.num3(f"Rbar{tok}CrossExclFamily", d, INDEP, *f, "cross_r")
        b.num3(f"Gap{tok}SameBundleVsCrossExclFamily", d, INDEP, *f, "gap_same_bundle")


def primary(b):
    rows, clustered, boots = [], set(), set()
    for t, tok, label in TASKS:
        row, is_cl, boot = primary_task(b, t, tok, label)
        rows.append(row)
        boots.add(boot)
        if is_cl:
            clustered.add(t)
    if len(boots) != 1:
        raise Fatal(f"bootstrap B differs across within_cross files: {sorted(boots)}")
    b.put("BootstrapB", boots.pop())
    primary_meff(b)
    mean_dice(b)
    cohorts(b)
    sensitivity(b)
    return rows, clustered


# Control and sensitivity analyses
def rob_calib(b, t, tok):
    rel = CALIB.format(t=t)
    d = b.load(rel)
    b.count(f"N{tok}CalibUnits", d, rel, "n_calibration_units")
    b.count(f"N{tok}EvalUnits", d, rel, "n_evaluation_units")
    b.value(f"K{tok}CalibSelected", lambda: fint(len(dig(d, rel, "fms_selected_on_calibration"))))
    gap = b.num3(f"Gap{tok}CalibSplit", d, rel, "evaluation", "gap")
    return f"{gap} {b.ci(f'Gap{tok}CalibSplitCI', d, rel, 'evaluation', 'case_bootstrap_95ci_gap')}"


def rob_subject(b, t, tok):
    rel = SUBJ.format(t=t)
    d = b.load(rel)
    b.count(f"N{tok}Subjects", d, rel, "n_subjects")
    b.num3(f"Gap{tok}Subject", d, rel, "subject_aggregated", "gap")
    b.ci(f"Gap{tok}SubjectCI", d, rel, "subject_aggregated", "bootstrap_95ci_gap")


def rob_audit(b, t, tok):
    d, row = b.load(AUDIT), ("task_rows", t)
    floor = {}
    for fl in FLOORS:
        def entry(fl=fl):
            return find(dig(d, AUDIT, *row, "floor_sweep"), AUDIT, f"task_rows/{t}/floor_sweep[floor={fl}]",
                        lambda x: close(x.get("floor"), fl))
        w = "Floor" + pct(fl)
        floor[fl] = (b.value(f"Gap{tok}{w}", lambda: f3(dig(entry(), AUDIT, "gap"))),
                     b.value(f"K{tok}{w}", lambda: fint(len(dig(entry(), AUDIT, "fms_kept")))))

    def stat(s):
        return num(dig(find(dig(d, AUDIT, *row, "metric_sensitivity_floor_0.30"), AUDIT,
                            f"task_rows/{t}/metric_sensitivity_floor_0.30[stat={s}]", lambda x: x.get("stat") == s),
                       AUDIT, "gap"))

    def loo():
        gaps = [num(dig(x, AUDIT, "gap")) for x in dig(d, AUDIT, *row, "leave_one_fm_out_floor_0.30")]
        if not gaps:
            raise Missing(f"{AUDIT}:task_rows/{t}/leave_one_fm_out_floor_0.30 is empty")
        return gaps

    icc = b.value(f"Gap{tok}Icc", lambda: f3(stat("icc")))
    b.value(f"Gap{tok}IccMinusPearson", lambda: f3(stat("icc") - stat("pearson")))
    b.value(f"N{tok}Loo", lambda: fint(len(loo())))
    loo_range = b.value(f"Gap{tok}LooRange", lambda: fci([min(loo()), max(loo())]))
    return icc, loo_range, f"{floor[0.4][0]} ({floor[0.4][1]})"


def rob_meta(b):
    d, m = b.load(AUDIT), ("primary_floor_0.30_meta_summary",)
    for key, name in (("mean_gap", "Mean"), ("median_gap", "Median"), ("min_gap", "Min"), ("max_gap", "Max")):
        b.num3(f"GapMeta{name}", d, AUDIT, *m, key)
    for t, tok, _ in TASKS:
        b.value(f"GapMetaLeaveOut{tok}", lambda: f3(dig(find(
            dig(d, AUDIT, *m, "leave_one_task_out"), AUDIT, f"leave_one_task_out[{t}]",
            lambda x: x.get("left_out_task") == t), AUDIT, "mean_gap")))


def rob_qcp(b, t, tok):
    d, r = b.load(QCP), ("task_rows", t)
    b.num3(f"BetaSame{tok}Rsq", d, QCP, *r, "quality_control_model", "r2")
    beta = b.num3(f"BetaSame{tok}", d, QCP, *r, "same_group_beta")
    return f"{beta} {b.ci(f'BetaSame{tok}CI', d, QCP, *r, 'same_group_beta_ci95_case_bootstrap')}"


def rob_fail(b, t, tok):
    """Failure-event phi at each empirical percentile threshold; returns {percentile: appendix cells}."""
    d, pre, out = b.load(FAIL), f"Phi{tok}", {}
    for pc in FAIL_PERCENTILES:
        def r(pc=pc):
            return find(dig(d, FAIL, "task_rows", t, "rows"), FAIL, f"task_rows/{t}/rows[percentile={pc}]",
                        lambda x: x.get("percentile") == pc)
        w = "P" + words(pc)
        vals = {name: b.value(pre + name + w, lambda key=key: f3(dig(r(), FAIL, key)))
                for key, name in (("same_fail_phi", "Same"), ("cross_fail_phi", "Cross"), ("threshold", "Threshold"),
                                  ("joint_fail_rate_ratio", "JointFailRatio"), ("gap_fail_phi", "Gap"))}
        ratio_ci = b.value(pre + "JointFailRatio" + w + "CI",
                          lambda: fci(dig(r(), FAIL, "primary_bootstrap_95ci_joint_fail_rate_ratio")))
        gap_ci = b.value(pre + "Gap" + w + "CI", lambda: fci(dig(r(), FAIL, "primary_bootstrap_95ci_gap_fail_phi")))
        n = {name: b.value(pre + name + w, lambda key=key: fint(dig(r(), FAIL, key)))
             for key, name in (("n_same_valid_phi_pairs", "NSameValidPairs"), ("n_cross_valid_phi_pairs", "NCrossValidPairs"),
                               ("n_same_pairs", "NSamePairs"), ("n_cross_pairs", "NCrossPairs"),
                               ("n_constant_failure_members", "NConstantFailureMembers"))}
        out[pc] = [f"P{pc}", vals["Threshold"], vals["Same"], vals["Cross"], f"{vals['Gap']} {gap_ci}",
                   f"{vals['JointFailRatio']} {ratio_ci}",
                   f"{n['NSameValidPairs']}/{n['NSamePairs']}, {n['NCrossValidPairs']}/{n['NCrossPairs']}"]
    return out


def rob_l1a(b, t, tok):
    d, cells = b.load(L1A), {}
    for key, name in L1A_METRICS:
        m = ("tasks", t, "thresholds", "empirical_p33", "metrics", key)
        gap = b.num3(f"{name}Gap{tok}PThirtyThree", d, L1A, *m, "gap")
        cells[key] = f"{gap} {b.ci(f'{name}Gap{tok}PThirtyThreeCI', d, L1A, *m, 'case_bootstrap_95ci', 'gap')}"
        b.count(f"{name}{tok}PThirtyThreeNValidSame", d, L1A, *m, "n_valid_same_pairs")
        b.count(f"{name}{tok}PThirtyThreeNValidCross", d, L1A, *m, "n_valid_cross_pairs")
    return cells["kappa"]


def rob_item(b, t, tok):
    d, a = b.load(ITEM), ("task_rows", t, "disjoint_panel_item_difficulty_floor_0.30")
    b.num3(f"Gap{tok}ResidMinusRaw", d, ITEM, *a, "residualized_gap_minus_raw_gap")
    b.ci(f"Gap{tok}ResidMinusRawCI", d, ITEM, *a, "bootstrap", "residualized_minus_raw_gap_95ci")
    b.num3(f"Attenuation{tok}Ratio", d, ITEM, *a, "attenuation_ratio_residual_over_raw")


def rob_e1(b, t, tok):
    d, cells = b.load(E1), {}

    def pool():
        cfg = find(dig(d, E1, "configurations"), E1, f"configurations[released_1x1,{t},complete]",
                   lambda c: c.get("config") == "released_1x1" and c.get("task") == t and c.get("status") == "complete")
        return find(dig(cfg, E1, "pools"), E1, f"{t}/pools[diverse4_enumeration]",
                    lambda p: p.get("pool") == "diverse4_enumeration")

    for src, name in E1_THRESHOLDS:
        def row(src=src):
            return find(dig(pool(), E1, "rows"), E1, f"{t}/diverse4_enumeration/rows[{src}]",
                        lambda r: src in ((r.get("threshold") or {}).get("sources") or ()))
        pre = f"MeffFourPool{tok}{name}"
        value = b.value(pre, lambda: f3(dig(row(), E1, "M_eff", "value")))
        cells[src] = f"{value} {b.value(pre + 'CI', lambda: fci(dig(row(), E1, 'M_eff', 'ci95')))}"
        b.value(pre + "Threshold", lambda: f3(dig(row(), E1, "threshold", "value")))
        b.value(pre + "RhoFail", lambda: f3(dig(row(), E1, "rho_fail", "value")))
        b.value(pre + "NSupported", lambda: fint(dig(row(), E1, "support", "n_pools_supported")))
        b.value(pre + "NPools", lambda: fint(dig(row(), E1, "support", "n_pools_total")))
        b.value(pre + "NUndefined", lambda: fint(dig(row(), E1, "support", "n_pools_undefined")))
    return cells["P33"]


def robustness(b, clustered):
    """Table 2 rows; also returns the per-task failure-threshold rows for the appendix."""
    rob_meta(b)
    b.value("KappaNeverFailConvention", lambda: KAPPA_NEVER_FAIL_TEXT[dig(b.load(L1A), L1A, "kappa_never_fail")])
    rows, fail = [], {}
    for t, tok, label in TASKS:
        if t in clustered:
            rob_subject(b, t, tok)
        rob_item(b, t, tok)
        icc, loo_range, floor40 = rob_audit(b, t, tok)
        fail[t] = rob_fail(b, t, tok)
        rows.append([label, rob_calib(b, t, tok), fail[t][33][4], rob_qcp(b, t, tok), rob_l1a(b, t, tok),
                     icc, loo_range, floor40, rob_e1(b, t, tok)])
    return rows, fail


def vit_arch(b):
    d = b.load(VIT)
    tasks = {x.get("task"): x for x in b.lookup([], d, VIT, "tasks") if isinstance(x, dict)}
    for t in VIT_ORDER:
        tok, g = TOK[t], (lambda *k, t=t: dig(tasks, VIT, t, *k))
        b.value(f"VitFunctional{tok}", lambda: f"{len(g('functional_vits'))}/{len(dig(d, VIT, 'meta', 'vit_subset'))}")
        b.value(f"VitSeedFloor{tok}", lambda: f3(g("seed_floor")))
        b.value(f"VitCrossPretrain{tok}", lambda: f3(g("cross_pretrain")))
        b.value(f"VitSeedFloor{tok}CI", lambda: fci(g("case_bootstrap", "seed_floor", "ci95")))
        b.value(f"VitCrossPretrain{tok}CI", lambda: fci(g("case_bootstrap", "cross_pretrain", "ci95")))
        b.value(f"VitCrossRef{tok}", lambda: f3(g("cross_ref")))
        b.value(f"VitDiff{tok}", lambda: f3(g("diff_floor_minus_cross_pretrain")))
        b.value(f"VitDiff{tok}CaseCI", lambda: fci(g("case_bootstrap", "diff", "ci95")))
        if isinstance(tasks.get(t), dict) and "cluster_bootstrap" in tasks[t]:
            b.value(f"VitDiff{tok}ClusterCI", lambda: fci(g("cluster_bootstrap", "diff", "ci95")))
        b.value(f"VitSeedPairs{tok}", lambda: fint(g("n_seed_pairs")))
        b.value(f"VitCrossPretrainPairs{tok}", lambda: fint(g("n_cross_pretrain_pairs")))
        vit_share(b, tasks, t, tok, g)


def vit_share(b, tasks, t, tok, g):
    a = b.value(f"SameBundleSeedPairs{tok}", lambda: fint(g("pair_share", "same_bundle_seed_pairs")))
    n = b.value(f"TotalSamePairs{tok}", lambda: fint(g("pair_share", "total_same_pairs")))
    share = b.value(f"SamePairSharePercent{tok}", lambda: f"{100 * num(g('pair_share', 'share')):.1f}")
    b.put(f"SamePairShare{tok}", rf"{a}/{n} = {share}\%")
    if isinstance(tasks.get(t), dict) and (tasks[t].get("pair_share") or {}).get("matches_json") is not True:
        b.notes.append(f"{VIT}: {t} pair_share.matches_json is not true")


def per_rater(b):
    rows = []
    for part, ptok in (("cup", "Cup"), ("disc", "Disc")):
        rel = PER_RATER.format(part=part)
        d = b.load(rel)
        if d is not None and d.get("status") != "PASS":
            b.notes.append(f"{rel}: status is {d.get('status')!r}, not PASS")
        for conv, ctok in RATER_CONVENTIONS:
            b.ci(f"PerRater{ptok}{ctok}SingleRaterRange", d, rel, conv, "headline", "single_rater_gap_range")
            b.num3(f"PerRater{ptok}{ctok}ConsensusAboveSingleMax", d, rel, conv, "headline", "amount_above_single_rater_max")
        for ann in ANNOTATIONS:
            cells = [LABEL[f"riga_{part}"], ann.replace("_", " ").capitalize()]
            for conv, ctok in RATER_CONVENTIONS:
                pre, a = f"PerRater{ptok}{ctok}{camel(ann)}", (conv, "annotations", ann)
                k, m = b.count(pre + "K", d, rel, *a, "n_bundles"), b.count(pre + "NMembers", d, rel, *a, "n_members")
                cells += [f"{k}/{m}", b.num3(pre + "WithinRbar", d, rel, *a, "within_rbar"),
                          b.num3(pre + "CrossRbar", d, rel, *a, "cross_rbar"),
                          f"{b.num3(pre + 'Gap', d, rel, *a, 'gap')} {b.ci(pre + 'GapCI', d, rel, *a, 'gap_case_bootstrap_95ci')}"]
            rows.append(cells)
    return rows


def xckpt(b):
    rows = []
    for rel in dict.fromkeys(kv(XCKPT, t) for t, _, _ in TASKS):
        if dig(b.load(rel), rel, "extra_cells_missing"):
            raise Fatal(f"incomplete checkpoint pool: {rel}")
    for t, tok, label in TASKS:
        rel = kv(XCKPT, t)  # per-task source (the official-split Kvasir row when Kvasir is primary on it)
        d = b.load(rel)
        r = ("rows", t)
        avail = b.count(f"XckptBundles{tok}", d, rel, *r, "n_bundles_before_floor")
        k = b.count(f"XckptK{tok}", d, rel, *r, "K_after_floor")
        tiers = [f"{b.num3(f'Xckpt{n}Rbar{tok}', d, rel, *r, key + '_rbar')} ({b.count(f'Xckpt{n}Pairs{tok}', d, rel, *r, key + '_n_pairs')})"
                 for key, n in (("same_bundle", "SameBundle"), ("same_family_ckpt", "SameFamily"), ("cross_family", "CrossFamily"))]
        gap = f"{b.num3(f'XckptGap{tok}', d, rel, *r, 'ckpt_gap')} {b.ci(f'XckptGap{tok}CI', d, rel, *r, 'ckpt_gap_ci95')}"
        prim = f"{b.num3(f'XckptGapPrimaryPool{tok}', d, rel, *r, 'primary11_ckpt_gap')} {b.ci(f'XckptGapPrimaryPool{tok}CI', d, rel, *r, 'primary11_ckpt_gap_ci95')}"
        b.value(f"XckptDropped{tok}", lambda: ", ".join(tex(x) for x in sorted(dig(d, rel, *r, "dropped_by_floor"))) or "none")
        row = b.lookup({}, d, rel, *r)
        row = row if isinstance(row, dict) else {}
        if row.get("gap_ci_cluster") is not None:  # quoted CI above is patient-cluster; keep slice-level case CI
            b.ci(f"XckptGap{tok}CaseCI", d, rel, *r, "gap_ci_case")
            b.ci(f"XckptGapPrimaryPool{tok}CaseCI", d, rel, *r, "primary11_gap_ci_case")
        if row.get("subject_level_ckpt_gap") is not None:
            b.num3(f"XckptSubjectGap{tok}", d, rel, *r, "subject_level_ckpt_gap", "gap")
            b.ci(f"XckptSubjectGap{tok}CI", d, rel, *r, "subject_level_ckpt_gap", "ci95")
            b.count(f"XckptSubjectN{tok}", d, rel, *r, "subject_level_ckpt_gap", "n_units")
        if row and row.get("l1b_gate") != "PASS":
            b.notes.append(f"{rel}: {t} l1b_gate is not PASS")
        rows.append([label, f"{k}/{avail}", *tiers, gap, prim])
    b.put("XckptCINote", XCKPT_CI_NOTE)
    return rows


def medenc_task(b, tasks, t, tok, label):
    g, enc_rows = (lambda *k: dig(tasks, MEDENC, t, *k)), []
    inc = b.num3(f"MedEncIncumbentCross{tok}", tasks, MEDENC, t, "incumbent_cross", "estimate")
    b.ci(f"MedEncIncumbentCross{tok}CI", tasks, MEDENC, t, "incumbent_cross", "ci95")
    for e in b.lookup([], tasks, MEDENC, t, "medical_kept"):
        pre, pe = f"MedEnc{camel(e)}{tok}", ("per_encoder", e)
        if (((tasks.get(t) or {}).get("per_encoder") or {}).get(e) or {}).get("passes_floor") is not True:
            b.notes.append(f"{MEDENC}: {t}/{e} passes_floor is not true")
        to_inc = f"{b.num3(pre + 'ToIncumbent', tasks, MEDENC, t, *pe, 'newcomer_to_incumbent', 'estimate')} " \
                 f"{b.ci(pre + 'ToIncumbentCI', tasks, MEDENC, t, *pe, 'newcomer_to_incumbent', 'ci95')}"
        diff = f"{b.num3(pre + 'Diff', tasks, MEDENC, t, *pe, 'diff_vs_incumbent_cross', 'estimate')} " \
               f"{b.ci(pre + 'DiffCI', tasks, MEDENC, t, *pe, 'diff_vs_incumbent_cross', 'ci95')}"
        enc_rows.append([label, MEDENC_NAMES.get(e, tex(e)), b.num3(pre + "MeanDice", tasks, MEDENC, t, *pe, "mean_dice"),
                         to_inc, inc, diff])
    pm, pre = ("pooled_medical",), f"MedEncPooled{tok}"
    to_inc = f"{b.num3(pre + 'ToIncumbent', tasks, MEDENC, t, *pm, 'newcomer_to_incumbent', 'estimate')} " \
             f"{b.ci(pre + 'ToIncumbentCI', tasks, MEDENC, t, *pm, 'newcomer_to_incumbent', 'ci95')}"
    diff = f"{b.num3(pre + 'Diff', tasks, MEDENC, t, *pm, 'diff_vs_incumbent_cross', 'estimate')} " \
           f"{b.ci(pre + 'DiffCI', tasks, MEDENC, t, *pm, 'diff_vs_incumbent_cross', 'ci95')}"
    enc_rows.append([label, "pooled medical", "--", to_inc, inc, diff])
    return enc_rows, medenc_summary(b, tasks, t, tok, label)


def medenc_summary(b, tasks, t, tok, label):
    nat = ("pooled_medical", "legacy_style_natural_only", "diff")
    natural = f"{b.num3(f'MedEncPooled{tok}NaturalOnlyDiff', tasks, MEDENC, t, *nat, 'estimate')} " \
              f"{b.ci(f'MedEncPooled{tok}NaturalOnlyDiffCI', tasks, MEDENC, t, *nat, 'ci95')}"
    mutual = f"{b.num3(f'MedEncMutual{tok}', tasks, MEDENC, t, 'medical_mutual', 'estimate')} " \
             f"{b.ci(f'MedEncMutual{tok}CI', tasks, MEDENC, t, 'medical_mutual', 'ci95')}"
    for pair in b.lookup({}, tasks, MEDENC, t, "medical_pairs"):
        b.num3(f"MedEncPair{tok}" + "".join(camel(x) for x in pair.split("--")), tasks, MEDENC, t, "medical_pairs", pair, "estimate")
    ext = {}
    for key, name in (("same", "Same"), ("cross", "Cross"), ("gap", "Gap")):
        ext[key] = b.num3(f"MedEncExtPool{tok}{name}", tasks, MEDENC, t, "pool15", key, "estimate")
        b.ci(f"MedEncExtPool{tok}{name}CI", tasks, MEDENC, t, "pool15", key, "ci95")
    ext_k = b.count(f"MedEncExtPool{tok}K", tasks, MEDENC, t, "pool15", "n_bundles")
    prim = f"{b.num3(f'MedEncPrimaryPool{tok}Gap', tasks, MEDENC, t, 'primary11_gap_json')} " \
           f"{b.ci(f'MedEncPrimaryPool{tok}GapCI', tasks, MEDENC, t, 'primary11_gap_ci95_json')}"
    return [label, mutual, natural,
            f"{ext['same']} / {ext['cross']} / {ext['gap']} {b.macros[f'MedEncExtPool{tok}GapCI']} (K={ext_k})", prim]


def medenc(b):
    d = b.load(MEDENC)
    tasks, enc_rows, summary = b.lookup({}, d, MEDENC, "tasks"), [], []
    for t, tok, label in TASKS:
        if t not in tasks:  # the medical-encoder family covers riga_cup, riga_disc and kvasir only
            continue
        rows, srow = medenc_task(b, b.lookup({}, b.load(kv(MEDENC, t)), kv(MEDENC, t), "tasks"), t, tok, label)
        enc_rows += rows
        summary.append(srow)
    return enc_rows, summary


def pixel_task(b, tasks, t, tok):
    fl, u = ("floor",), ("floor", "err_uncond")
    b.count(f"PixelN{tok}", tasks, PIXEL, t, "n_cases")
    b.count(f"PixelK{tok}", tasks, PIXEL, t, "K_after_floor")
    cells = {}
    for key, name in PIXEL_STRATA:
        value = b.num3(f"Pixel{name}{tok}", tasks, PIXEL, t, *u, key, "mean")
        cells[key] = f"{value} {b.ci(f'Pixel{name}{tok}CI', tasks, PIXEL, t, *u, key, 'ci')}"
        b.count(f"PixelNPairs{name}{tok}", tasks, PIXEL, t, *fl, "n_pairs", key)
    g = (*u, "gap_same_vs_cross_family")
    cells["gap"] = f"{b.num3(f'PixelGap{tok}', tasks, PIXEL, t, *g, 'gap')} {b.ci(f'PixelGap{tok}CI', tasks, PIXEL, t, *g, 'gap_ci')}"
    if "patient_cluster_gap_ci" in (((((tasks.get(t) or {}).get("floor") or {}).get("err_uncond") or {})
                                     .get("gap_same_vs_cross_family")) or {}):
        b.ci(f"PixelGap{tok}ClusterCI", tasks, PIXEL, t, *g, "patient_cluster_gap_ci")
        b.count(f"PixelClusterN{tok}", tasks, PIXEL, t, *g, "n_patients")
    b.num3(f"PixelGap{tok}NoFloor", tasks, PIXEL, t, "no_floor", "err_uncond", "gap_same_vs_cross_family", "gap")
    sb = (*u, "gap_same_bundle_vs_cross_bundle")
    b.num3(f"PixelGapSameBundleVsCrossBundle{tok}", tasks, PIXEL, t, *sb, "gap")
    b.ci(f"PixelGapSameBundleVsCrossBundle{tok}CI", tasks, PIXEL, t, *sb, "gap_ci")
    b.num3(f"PixelGapBroad{tok}", tasks, PIXEL, t, *u, "gap_same_vs_cross_family_broad", "gap")
    for block, name in (("fp_uncond", "FpOnly"), ("fn_uncond", "FnOnly"), ("err_cond", "Cond")):
        b.num3(f"PixelGap{name}{tok}", tasks, PIXEL, t, *fl, block, "gap_same_vs_cross_family", "gap")
        b.ci(f"PixelGap{name}{tok}CI", tasks, PIXEL, t, *fl, block, "gap_same_vs_cross_family", "gap_ci")
    return cells


def pixel(b):
    tasks = {}
    for rel in dict.fromkeys(kv(PIXEL, t) for t in PIXEL_ORDER):  # Kvasir row from the official-split replay (kv)
        found = {x.get("task"): x for x in b.lookup([], b.load(rel), rel, "tasks") if isinstance(x, dict)}
        tasks.update({t: found[t] for t in PIXEL_ORDER if kv(PIXEL, t) == rel and t in found})
    rows = []
    for t in PIXEL_ORDER:
        tok = TOK[t]
        if (tasks.get(t) or {}).get("primary_pool_match") is not True:
            b.notes.append(f"{PIXEL}: {t} primary_pool_match is not true")
        c = pixel_task(b, tasks, t, tok)
        rows.append([LABEL[t], b.macros[f"PixelN{tok}"], b.macros[f"PixelK{tok}"], c["same_bundle"],
                     c["same_family_ckpt"], c["same"], c["cross_family"], c["gap"]])
    return rows


def amos(b):
    """Native-3D AMOS22 comparison from its archived metrics."""
    d = b.load(AMOS)
    for mod, mtok in (("ct", "CT"), ("mri", "MRI")):
        m, pre = ("by_modality", mod), f"Amos{mtok}"
        b.count(pre + "N", d, AMOS, *m, "n")
        for key, name in (("same_system_mrseg_fold0_x_fold1", "SameR"), ("cross_system_mean_of_two_pairs", "CrossR"),
                          ("gap_same_minus_cross_mean", "Gap")):
            b.num3(pre + name, d, AMOS, *m, "pearson_r", key)
            b.ci(pre + name + "CI", d, AMOS, *m, "bootstrap", "percentile_ci95", key)
        meff = (*m, "phase2_secondary", "M_eff")
        b.num3(pre + "Meff", d, AMOS, *meff, "selection", "primary_P33", "M_eff", "value")
        b.ci(pre + "MeffCI", d, AMOS, *meff, "selection", "primary_P33", "M_eff", "ci95")
        b.count(pre + "MeffNominal", d, AMOS, *meff, "M_operational_members")
        for system in b.lookup({}, d, AMOS, *m, "mean_macro_dice_13"):
            b.num3(pre + "Dice" + camel(system), d, AMOS, *m, "mean_macro_dice_13", system)


def app_tau(b):
    dt, rows = b.load(TAU), []
    for t, tok, label in TASKS:
        for tau in TAUS:
            def row(tau=tau):
                return find(dig(dt, TAU, "tasks", t, "rows"), TAU, f"tasks/{t}/rows[tau={tau}]",
                            lambda x: close(x.get("tau"), tau))
            cells = [b.value(f"Tau{name}{tok}Tau{pct(tau)}", lambda key=key: f3(dig(row(), TAU, key)))
                     for key, name in (("mono_recovery", "Mono"), ("diverse_recovery", "Diverse"), ("delta", "Delta"))]
            rows.append([label, f"{tau:.2f}", *cells])
    return rows


def app_qc(b):
    d, rows = b.load(QCP), []
    for t, tok, label in TASKS:
        r = ("task_rows", t)
        k = b.count(f"QcK{tok}", d, QCP, *r, "n_fms_after_floor")
        pairs = f"{b.count(f'QcNSamePairs{tok}', d, QCP, *r, 'n_same_group_pairs')}/" \
                f"{b.count(f'QcNCrossPairs{tok}', d, QCP, *r, 'n_cross_group_pairs')}"
        raw = b.num3(f"QcRawGap{tok}", d, QCP, *r, "raw_gap_from_pair_rows")
        boot = b.count(f"QcBootstrapB{tok}", d, QCP, *r, "bootstrap", "B_completed")
        rows.append([label, k, pairs, raw, f"{b.macros[f'BetaSame{tok}']} {b.macros[f'BetaSame{tok}CI']}",
                     b.macros[f"BetaSame{tok}Rsq"], boot])
    return rows


def app_item(b, clustered):
    """Item-difficulty summary with panel coverage and the permutation null control."""
    d, audit, summary = b.load(ITEM), b.load(AUDIT), []
    for t, tok, label in TASKS:
        a = ("task_rows", t, "disjoint_panel_item_difficulty_floor_0.30")
        ps = (*a, "panel_spread")
        pn = ("task_rows", t, "permutation_negative_control_floor_0.30")
        raw = f"{b.num3(f'ItemRawGap{tok}', d, ITEM, *a, 'raw', 'gap')} {b.ci(f'ItemRawGap{tok}CI', d, ITEM, *a, 'bootstrap', 'raw_gap_95ci')}"
        res = f"{b.num3(f'ItemResidGap{tok}', d, ITEM, *a, 'residualized', 'gap')} " \
              f"{b.ci(f'ItemResidGap{tok}CI', d, ITEM, *a, 'bootstrap', 'residualized_gap_95ci')}"
        att = f"{b.macros[f'Attenuation{tok}Ratio']} {b.ci(f'Attenuation{tok}RatioCI', d, ITEM, *a, 'bootstrap', 'attenuation_ratio_95ci')}"
        spread = f"{b.count(f'ItemPanelsEligible{tok}', d, ITEM, *ps, 'n_eligible')}/{b.count(f'ItemPanels{tok}', d, ITEM, *ps, 'n_panels')} " \
                 f"({b.count(f'ItemPanelsPositive{tok}', d, ITEM, *ps, 'positive')})"
        null = b.value(f"PermNull{tok}CI", lambda: fci([dig(audit, AUDIT, *pn, "null_q025"), dig(audit, AUDIT, *pn, "null_q975")]))
        null += f" ($f_{{\\mathrm{{perm}}}}$={b.num3(f'PermNullP{tok}', audit, AUDIT, *pn, 'null_p_ge_observed')}, " \
                f"n={b.count(f'PermNullN{tok}', audit, AUDIT, *pn, 'valid_permutations')})"
        subj = b.num3(f"ItemSubjectResidMinusRaw{tok}", d, ITEM, "task_rows", t,
                      "subject_level_disjoint_panel_item_difficulty_floor_0.30", "residualized_gap_minus_raw_gap") \
            if t in clustered else "--"
        summary.append([label, raw, res, f"{b.macros[f'Gap{tok}ResidMinusRaw']} {b.macros[f'Gap{tok}ResidMinusRawCI']}", att,
                        spread, b.ci(f"ItemPanelRange{tok}", d, ITEM, *ps, "range"),
                        b.num3(f"ItemPanelSd{tok}", d, ITEM, *ps, "sd"), null, subj])
    return summary


def app_meff(b):
    rows = []
    for t, tok, label in TASKS:
        m, p = f"Meff{tok}{b.meff_tau[t]}", f"MeffFourPool{tok}PThirtyThree"
        rows.append([label, b.macros[m], b.macros[m + "CovExact"], f"{b.macros[m + 'NValid']}/{b.macros[m + 'NSeedCompositions']}",
                     b.macros[m + "NIneligible"], f"{b.macros[p]} {b.macros[p + 'CI']}", "n/a",
                     f"{b.macros[p + 'NSupported']}/{b.macros[p + 'NPools']}", b.macros[p + "NUndefined"]])
    return rows


def family(b, name, pattern, prefix, tasks, rbar="Rbar"):
    """Within/cross macros for one control configuration."""
    out = {}
    for t in tasks:
        tok, rel = TOK[t], kv(pattern, t).format(t=t)
        names = {"k": f"{prefix}K{tok}", "within": f"{prefix}Within{rbar}{tok}", "cross": f"{prefix}Cross{rbar}{tok}",
                 "gap": f"{prefix}Gap{tok}", "ci": f"{prefix}Gap{tok}CI"}
        d = b.load(rel)
        out[t] = {"k": b.value(names["k"], lambda: fint(len(dig(d, rel, "fms_kept_after_floor")))),
                  "within": b.num3(names["within"], d, rel, "within_rbar"),
                  "cross": b.num3(names["cross"], d, rel, "cross_rbar"),
                  "gap": b.value(names["gap"], lambda: f3(wc_gap(d, rel))),
                  "ci": b.ci(names["ci"], d, rel, "primary_bootstrap_95ci_gap")}
    return out


def nnunet(b):
    """Statistics for nnU-Net models evaluated on the same test cases."""
    rows = []
    for t in NNUNET_TASKS:
        tok, rel = TOK[t], NNUNET_FILE[t]
        n = {k: f"Nnunet{tok}{k}" for k in ("Epochs", "Models", "NTest", "MeanDice", "WithinRbar", "CrossRbar",
                                             "Gap", "GapCI", "Meff", "MeffCI")}
        d = b.load(rel)
        p33 = lambda: dig(find(dig(d, rel, "m_eff_e1", "pools"), rel, f"m_eff_e1/pools[{NNUNET_POOL}]",  # noqa: E731
                               lambda p: p.get("pool") == NNUNET_POOL), rel, "selection", "primary_P33", "M_eff")
        v = {"Epochs": b.count(n["Epochs"], d, rel, "trainer_epochs"), "Models": b.count(n["Models"], d, rel, "n_models"),
             "NTest": b.count(n["NTest"], d, rel, "n_test"), "MeanDice": b.num3(n["MeanDice"], d, rel, "mean_dice_all"),
             "WithinRbar": b.num3(n["WithinRbar"], d, rel, "within_fold_rbar"),
             "CrossRbar": b.num3(n["CrossRbar"], d, rel, "cross_fold_rbar"),
             "Gap": b.num3(n["Gap"], d, rel, "delta_within_minus_cross"),
             "GapCI": b.ci(n["GapCI"], d, rel, "delta_ci95_case_bootstrap"),
             "Meff": b.value(n["Meff"], lambda: f3(dig(p33(), rel, "value"))),
             "MeffCI": b.value(n["MeffCI"], lambda: fci(dig(p33(), rel, "ci95")))}
        rows.append([LABEL[t], v["Epochs"], v["Models"], v["NTest"], v["MeanDice"], v["WithinRbar"], v["CrossRbar"],
                     f"{v['Gap']} {v['GapCI']}", f"{v['Meff']} {v['MeffCI']}"])
    return rows


def unet_rows(fam):
    return [[LABEL[t], v["k"], v["within"], v["cross"], f"{v['gap']} {v['ci']}",
             rf"\Gap{TOK[t]}Corrected{{}} \Gap{TOK[t]}PrimaryCI (K \K{TOK[t]})",
             rf"\UnetSkipNineGap{TOK[t]}{{}} \UnetSkipNineGap{TOK[t]}CI (K \UnetSkipNineK{TOK[t]})",
             rf"\PrimaryNineBundleGap{TOK[t]}{{}} \PrimaryNineBundleGap{TOK[t]}CI (K \PrimaryNineBundleK{TOK[t]})"]
            for t, v in fam.items()]


def ft_rows(b):
    """Fine-tuning macros and paired-comparison table rows."""
    sets = [("kvasir_official", "Kvasir", "", "Kvasir (official)"),
            ("kvasir_grouped_v2", "Kvasir", "GroupedSens", "Kvasir (content-group split, sensitivity)"),
            ("riga_cup", "Cup", "", LABEL["riga_cup"])]
    d = b.load(FT_TABLE)
    rows = []
    for key, tok, sfx, label in sets:
        for arm, arm_tok, arm_label in FT_ARMS:
            n = lambda q: f"{q}{arm_tok}{tok}{sfx}"  # noqa: E731
            names = [n("FtK"), n("FtWithinRbar"), n("FtCrossRbar"), n("FtGap"), n("FtGap") + "CI", n("FrozenMatchedGap"),
                     n("FrozenMatchedGap") + "CI", n("FrozenMatchedK"), n("FtNCollapsed"), n("FtMinusFrozenGap"),
                     n("FtMinusFrozenGapCI"), n("FtBelowFrozen")]
            a = ("rows", key, arm)
            v = [b.value(names[0], lambda: fint(dig(d, FT_TABLE, *a, "ft", "k"))),
                 b.num3(names[1], d, FT_TABLE, *a, "ft", "within_rbar"), b.num3(names[2], d, FT_TABLE, *a, "ft", "cross_rbar"),
                 b.num3(names[3], d, FT_TABLE, *a, "ft", "gap"), b.ci(names[4], d, FT_TABLE, *a, "ft", "primary_bootstrap_95ci_gap"),
                 b.num3(names[5], d, FT_TABLE, *a, "frozen_matched", "gap"),
                 b.ci(names[6], d, FT_TABLE, *a, "frozen_matched", "primary_bootstrap_95ci_gap"),
                 b.value(names[7], lambda: fint(dig(d, FT_TABLE, *a, "frozen_matched", "k"))),
                 b.value(names[8], lambda: fint(len(dig(d, FT_TABLE, *a, "collapsed_heads")))),
                 b.num3(names[9], d, FT_TABLE, *a, "ft_minus_frozen_gap", "point"),
                 b.ci(names[10], d, FT_TABLE, *a, "ft_minus_frozen_gap", "ci95"),
                 b.text(names[11], d, FT_TABLE, *a, "ft_below_frozen_list")]
            for fm in b.lookup({}, d, FT_TABLE, *a, "mean_dice"):
                b.num3(n("FtMeanDice") + camel(fm), d, FT_TABLE, *a, "mean_dice", fm, "ft")
            if "excl_collapsed" in b.lookup({}, d, FT_TABLE, *a):
                b.num3(n("FtGapExclCollapsed"), d, FT_TABLE, *a, "excl_collapsed", "ft", "gap")
                b.ci(n("FtGapExclCollapsed") + "CI", d, FT_TABLE, *a, "excl_collapsed", "ft", "primary_bootstrap_95ci_gap")
            rows.append([label, arm_label, v[0], v[1], v[2], f"{v[3]} {v[4]}", f"{v[5]} {v[6]} (K {v[7]})", f"{v[9]} {v[10]}"])
    return rows


# ------------------------------------------------------------------------------------------- output
def header(b, rels, lead="% "):
    lines = [f"{lead}GENERATED by code/scripts/{Path(__file__).name} from result JSON only; do not edit by hand.",
             f"{lead}Sources (sha256  path):"]
    return lines + [f"{lead}  {b.sha[r]}  {r}" for r in rels]


def table(b, rels, columns, rows, keys=(), notes=()):
    lines = header(b, rels) + [f"% Columns: {columns}", *[f"% Note: {n}" for n in notes],
                               r"% Requires \input{generated/corrected_numbers.tex}."]
    lines += [f"% JSON keys: {k}" for k in keys]
    body = ("\n" + r"\addlinespace[1pt]" + "\n").join(" & ".join(r) + r" \\" for r in rows)
    return "\n".join(lines) + "\n" + body + "\n"


def macros_file(b):
    lines = header(b, list(b.sha)) + [NAMING]
    lines += [r"\newcommand{\%s}{%s}" % (k, v) for k, v in b.macros.items()]
    return "\n".join(lines) + "\n"


def status_file(b, outputs):
    lines = ["# Generated table inputs", "", "| Output | Content |", "|---|---|"]
    lines += [f"| `{name}` | {description} |" for name, description in outputs]
    lines += ["", "## Input checksums", "", "| SHA-256 | Path |", "|---|---|"]
    lines += [f"| `{checksum}` | `{path}` |" for path, checksum in b.sha.items()]
    if b.notes:
        lines += ["", "## Input notes", ""] + [f"- {note}" for note in b.notes]
    return "\n".join(lines) + "\n"


def secondary(b, primary_rows, clustered):
    wc = [WC.format(t=t) for t, _, _ in TASKS]
    robust_rows, fail = robustness(b, clustered)
    vit_arch(b)
    med_rows, med_summary = medenc(b)
    item_rows = app_item(b, clustered)
    fams = {name: family(b, name, pattern, prefix, tasks) for name, pattern, prefix, tasks in SECONDARY_FAMILIES}
    family(b, *PRIMARY_NINE, rbar="")
    family(b, *UNET_NINE)
    amos(b)
    # Keep current rows in memory for consolidation; only paper-referenced bodies are rendered.
    bodies = {
        "table_primary_corrected.tex": (wc, primary_rows),
        "table_robustness_corrected.tex": ([CALIB.format(t=t) for t, _, _ in TASKS] + [FAIL, QCP, L1A, AUDIT, E1], robust_rows),
        "table_per_rater.tex": ([PER_RATER.format(part=p) for p in ("cup", "disc")], per_rater(b)),
        "table_distinct_checkpoint.tex": (list(dict.fromkeys(kv(XCKPT, t) for t, _, _ in TASKS)), xckpt(b)),
        "table_medenc_pool.tex": (list(dict.fromkeys(kv(MEDENC, t) for t, _, _ in TASKS)), med_rows),
        "table_medenc_pool_summary.tex": (list(dict.fromkeys(kv(MEDENC, t) for t, _, _ in TASKS)), med_summary),
        "table_pixel_overlap.tex": (list(dict.fromkeys(kv(PIXEL, t) for t in PIXEL_ORDER)), pixel(b)),
        "table_unet_skip.tex": ([kv(p, t).format(t=t) for p in (SECONDARY_FAMILIES[0][1], UNET_NINE[1], PRIMARY_NINE[1])
                                 for t in SECONDARY_FAMILIES[0][3]], unet_rows(fams["unet-skip"])),
        "table_ft.tex": ([FT_TABLE], ft_rows(b)),
        "table_nnunet.tex": ([NNUNET_FILE[t] for t in NNUNET_TASKS], nnunet(b)),
        "table_app_tau_sweep.tex": ([TAU], app_tau(b)),
        "table_app_failure_thresholds.tex": ([FAIL], [[label, *fail[t][pc]] for t, _, label in TASKS for pc in FAIL_PERCENTILES]),
        "table_app_item_difficulty.tex": ([ITEM, AUDIT], item_rows),
        "table_app_meff.tex": ([MEFF.format(t=t) for t, _, _ in TASKS] + [E1], app_meff(b)),
    }
    bodies.update(heldout_numbers.build(b, sys.modules[__name__]))
    kvasir_official_numbers.build(b, sys.modules[__name__])
    gpu_hours_numbers.build(b, sys.modules[__name__])
    files = {"table_app_qc_pairs.tex": table(b, [QCP], "Task & K & pairs & raw gap & beta [CI] & R squared & B", app_qc(b))}
    files.update(mask_align_numbers.build(b, sys.modules[__name__]))
    files.update(scope_numbers.build(b, sys.modules[__name__]))
    files.update(consolidated_numbers.build(b, sys.modules[__name__], bodies))
    files["corrected_numbers.tex"] = macros_file(b)
    files["INPUTS.md"] = status_file(b, [(n, f"{len(v.splitlines())} lines") for n, v in files.items()])
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build public LaTeX numbers from result JSON files.")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]), help="project root")
    ap.add_argument("--out", default=None, help="output dir (default <root>/paper/generated)")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    out = Path(args.out) if args.out else root / "paper/generated"
    b = Build(root)
    try:
        t1_rows, clustered = primary(b)
        files = secondary(b, t1_rows, clustered)
    except (Fatal, Missing) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (out / name).write_text(text)
    print(f"wrote {len(files)} files to {out}: {len(b.macros)} numeric macros")
    return 0


if __name__ == "__main__":
    sys.exit(main())
