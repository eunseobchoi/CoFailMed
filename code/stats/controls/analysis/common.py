"""Functional Dice floor and patient identifiers for the control scorers."""
from __future__ import annotations
from fmpool.case_ids import brats_patient_id

FUNCTIONAL_FLOOR = 0.30


def subject_ids(task: str, case_ids: tuple[str, ...]) -> tuple[str, ...] | None:
    if task == "acdc_lv":
        return tuple(case_id.split("_frame", 1)[0] for case_id in case_ids)
    if task in {"brats_wt", "brats_wt_multimodal", "brats_multiclass", "brats_wt_3d"}:
        return tuple(brats_patient_id(case_id) for case_id in case_ids)
    return None
