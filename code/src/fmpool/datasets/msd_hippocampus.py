"""MSD Task04_Hippocampus central-axial-slice loader (task ``msd_hippocampus``).

Dataset: 260 imagesTr 3D MRI volumes. Labels {0: background, 1: head,
2: body}; binary foreground = ``label > 0`` (head + body merged).
The 80/20 split is VOLUME-level, not patient-level: MSD Hippocampus volumes are
left/right crops of 195 subjects (390 volumes) and no subject mapping is released.
"""
from __future__ import annotations

from fmpool.datasets._msd_common import MSDCenterSliceDataset


class MSDHippocampusDataset(MSDCenterSliceDataset):
    """MSD Task04_Hippocampus binary head+body segmentation."""

    task_key = "msd_hippocampus"
    split_task_name = "msd_hippocampus"
    provenance_extra = (
        "Task04_Hippocampus (260 imagesTr). Binary foreground = "
        "(label > 0) merging head (1) and body (2)."
    )
