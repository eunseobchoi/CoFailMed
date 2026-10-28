"""Configuration for the seven-task encoder-family pool study."""

from __future__ import annotations

from analysis.common import OUTPUT_ROOT

TASKS = (
    "kvasir",
    "acdc_lv",
    "brats_wt",
    "riga_cup",
    "riga_disc",
    "isic2018",
    "msd_hippocampus",
)
PRIMARY_TASKS = frozenset(TASKS[:5])


def scope_tasks(requested=None) -> tuple[tuple[str, ...], dict | None]:
    """Study TASKS by default (no scope record). ``--tasks`` may only restrict the study list; the
    restriction keeps study order and per-task seeds and returns an explicit scope record."""
    if not requested:
        return TASKS, None
    unknown = sorted(set(requested) - set(TASKS))
    if unknown:
        raise ValueError(f"--tasks {unknown} not in the seven-task study {TASKS}")
    tasks = tuple(t for t in TASKS if t in set(requested))
    return tasks, {
        "label": f"RESTRICTED SCOPE: {len(tasks)} of {len(TASKS)} study tasks; "
                 "not the full seven-task study",
        "tasks": tasks,
        "omitted_tasks": tuple(t for t in TASKS if t not in tasks),
    }
POOL_SIZE = 4
SPLIT_COUNT = 200
DEFAULT_OUTPUT = OUTPUT_ROOT / "pool_selection.json"
DEFAULT_TABLE = OUTPUT_ROOT / "pool_selection.md"
RULE_NAMES = ("architecture_diversity",)
METRIC_NAMES = ("all_fail_probability", "conditional_recovery", "marginal_dice")

RULE_DEFINITION = (
    "For each calibration split, retain bundles with calibration mean Dice >= 0.30 "
    "averaged over their four heads; all heads of passing bundles remain candidates. "
    "Choose each encoder family's highest-calibration-mean head, then the top "
    "M=4 family representatives; break ties by candidate ID. Split 50% of units "
    "with NumPy PCG64, using patients for ACDC and BraTS and cases otherwise. "
    "Calibration means weight patients equally on ACDC and BraTS. ISIC uses images; "
    "images of a known patient can occur in both halves. The threshold is the "
    "calibration-best functional head's per-case P33 (slice scores on ACDC/BraTS), "
    "not a percentile of patient means, fixed on evaluation. "
    "No evaluation values affect eligibility, the threshold, or selection."
)
THRESHOLD_JUSTIFICATION = (
    "Per-case P33 is computed from the calibration-best functional member using "
    "calibration cases only and held fixed on the evaluation half."
)
