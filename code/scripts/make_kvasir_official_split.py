"""Build ``data/splits/kvasir_official_split.json`` from the official Kvasir-SEG 880/120 lists.

Source: Jha et al., IEEE Access 2021 (arXiv 2011.07631), files ``Data-split/{train,val}.txt`` of
DebeshJha/Kvasir-SEG@6f375457209b27b1da42dbc5734ab271060b15dc (byte-identical to
DebeshJha/2020-MediaEval-Medico-polyp-segmentation@d594c08b7f3a1cc40ef878284b5fc29de822bf18
``kvasir-seg-train-val/{train,val}.txt``). The official ``val`` list is the evaluation set here
(``test``); ``val`` stays empty (no selection), exactly as in ``kvasir_grouped_v2``.
Raw content hashes and near-duplicate groups are copied from the committed grouped v2 manifest,
whose groups were produced by ``fmpool.datasets.kvasir._content_groups`` on the same 1000 files.

Usage: python make_kvasir_official_split.py TRAIN_TXT VAL_TXT GROUPED_V2_JSON OUT_JSON RETRIEVAL_UTC
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SCHEME = "official_jha2021_880_120"
SOURCE = {
    "paper": "Jha et al., Real-Time Polyp Detection, Localization and Segmentation in Colonoscopy "
             "Using Deep Learning, IEEE Access 2021 (arXiv 2011.07631): 880 training / 120 validation images",
    "repository": "https://github.com/DebeshJha/Kvasir-SEG",
    "commit": "6f375457209b27b1da42dbc5734ab271060b15dc",
    "paths": {"train": "Data-split/train.txt", "test": "Data-split/val.txt"},
    "git_blob_sha1": {"train": "6f777ef7651767f4aae0e96637083fa3d8d5bfb1",
                      "test": "d390bd3c0329d39bc0b8a6dd8255b04f10dad584"},
    "mirror": {"repository": "https://github.com/DebeshJha/2020-MediaEval-Medico-polyp-segmentation",
               "commit": "d594c08b7f3a1cc40ef878284b5fc29de822bf18",
               "paths": {"train": "kvasir-seg-train-val/train.txt", "test": "kvasir-seg-train-val/val.txt"},
               "byte_identical": True},
}


def build(train_txt: Path, val_txt: Path, grouped: dict, retrieval_utc: str) -> dict:
    raw = {"train": train_txt.read_bytes(), "test": val_txt.read_bytes()}
    for role, data in raw.items():
        blob_sha = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if blob_sha != SOURCE["git_blob_sha1"][role]:
            raise ValueError(f"Kvasir official {role} source differs from the pinned Git blob")
    train, test = raw["train"].decode("utf-8").split(), raw["test"].decode("utf-8").split()
    ids = set(grouped["content_sha256"])
    if (len(train), len(test)) != (880, 120) or len(set(train)) != 880 or len(set(test)) != 120:
        raise ValueError("official lists must hold 880 + 120 unique IDs")
    if set(train) & set(test) or set(train) | set(test) != ids:
        raise ValueError("official lists must partition the 1000 pinned Kvasir-SEG IDs")
    groups = sorted({tuple(sorted(g)) for g in grouped["near_duplicate_groups"]})
    in_train = all(set(g) <= set(train) for g in groups)
    if not in_train:
        raise ValueError("a near-duplicate group is not entirely in the official train list")
    return {
        "train": sorted(train), "val": [], "test": sorted(test),
        "split_schema_version": "kvasir_official_v1",
        "split_scheme": SCHEME,
        "test_role": "retrospective_evaluation_official_val_list",
        "near_duplicate_groups": [list(g) for g in groups],
        "near_duplicate_groups_all_in_train": in_train,
        "task": "kvasir_official",
        "provenance": {**SOURCE,
                       "file_sha256": {role: hashlib.sha256(data).hexdigest() for role, data in raw.items()},
                       "retrieval_utc": retrieval_utc,
                       "ordering": "IDs sorted; official file order is not used"},
        "content_sha256": grouped["content_sha256"],
        "content_scan": grouped["content_scan"],
        "group_rule": grouped["group_rule"],
        "groups_from": "kvasir_grouped_v2_split.json near_duplicate_groups (deduplicated)",
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("TRAIN_TXT", "VAL_TXT", "GROUPED_V2_JSON", "OUT_JSON", "RETRIEVAL_UTC"):
        parser.add_argument(name)
    train_txt, val_txt, grouped_json, out, retrieval_utc = vars(parser.parse_args(argv)).values()
    split = build(Path(train_txt), Path(val_txt), json.loads(Path(grouped_json).read_text()), retrieval_utc)
    with Path(out).open("x", encoding="utf-8") as fh:  # never overwrite a committed split
        json.dump(split, fh, indent=2)
    print(out, len(split["train"]), len(split["test"]), split["near_duplicate_groups_all_in_train"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
