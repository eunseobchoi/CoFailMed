"""BraTS-GLI patient identity and versioned patient split validation (stdlib only)."""
from __future__ import annotations

import random
import re

_BRATS_ID = re.compile(r"(BraTS-GLI-[0-9]{5})(?:-[0-9]{3}(?:_slice[0-9]{3,})?)?")
_SESSION_ID = re.compile(r"BraTS-GLI-[0-9]{5}-[0-9]{3}")


def brats_patient_id(case_id: str) -> str:
    """Return the patient prefix from a patient, full session, or slice ID.

    The three digits after the five-digit patient ID identify a timepoint,
    not an independent patient. Invalid identifiers fail closed.
    """
    match = _BRATS_ID.fullmatch(case_id)
    if match is None:
        raise ValueError(f"invalid BraTS-GLI case ID: {case_id!r}")
    return match.group(1)


def _session_splits(sessions, seed: int) -> dict[str, str]:
    sessions = sorted(sessions)
    if len(sessions) != len(set(sessions)) or any(
        not _SESSION_ID.fullmatch(session) for session in sessions
    ):
        raise ValueError("invalid or duplicate BraTS session IDs")
    patients = sorted({brats_patient_id(session) for session in sessions})
    random.Random(seed).shuffle(patients)
    test = set(patients[:round(len(patients) * 0.2)])
    return {s: "test" if brats_patient_id(s) in test else "train" for s in sessions}


def brats_patient_split(case_entries, sessions, seed: int = 42) -> dict:
    """Assign all visits together; retain sessions with no eligible 2D slices."""
    entries = sorted([list(e) for e in case_entries])
    assignments = _session_splits(sessions, seed)
    manifest = {
        "schema_version": "brats_patient_v2", "seed": seed,
        "cluster_unit": "BraTS-GLI patient prefix",
        "_session_splits": assignments, "_case_entries": entries,
        **{split: [cid for cid, session, _ in entries if assignments[session] == split]
           for split in ("train", "val", "test")},
    }
    validate_brats_split(manifest)
    return manifest


def validate_brats_split(manifest: dict, sessions=None, case_entries=None) -> None:
    """Validate identity, index, partition and optionally the actual raw cohort.

    Pass discovered sessions and freshly selected raw case entries in loaders.
    Without these arguments this only validates manifest consistency, not raw data.
    """
    if manifest.get("schema_version") != "brats_patient_v2":
        raise ValueError("BraTS requires a brats_patient_v2 manifest")
    assignments = manifest["_session_splits"]
    seed = manifest["seed"]
    if type(seed) is not int or assignments != _session_splits(assignments, seed):
        raise ValueError("BraTS session assignments differ from seeded patient 80/20 split")
    if sessions is not None and set(sessions) != set(assignments):
        raise ValueError("BraTS actual session cohort differs from manifest")
    index = {}
    for cid, session, axi in manifest["_case_entries"]:
        if (session not in assignments or type(axi) is not int or axi < 0
                or cid != f"{session}_slice{axi:03d}"):
            raise ValueError(f"noncanonical BraTS case entry: {(cid, session, axi)!r}")
        if cid in index:
            raise ValueError(f"duplicate BraTS case entry: {cid}")
        index[cid] = (session, axi)
    if case_entries is not None:
        actual = [tuple(e) for e in case_entries]
        if len(actual) != len(set(actual)) or set(actual) != {
            (cid, session, axi) for cid, (session, axi) in index.items()
        }:
            raise ValueError("BraTS actual selected slice index differs from manifest")
    seen = set()
    for split in ("train", "val", "test"):
        for cid in manifest[split]:
            if cid in seen or cid not in index:
                raise ValueError(f"duplicate or unknown BraTS split case: {cid}")
            seen.add(cid)
            if assignments[index[cid][0]] != split:
                raise ValueError(f"BraTS loaded-index split mismatch: {cid}")
    if seen != set(index):
        raise ValueError("BraTS split cases do not cover the loaded index")
