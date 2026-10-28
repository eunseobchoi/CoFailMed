r"""GPU-hour macros for the numbers builder (stdlib only).

Reads recorded per-family GPU occupancy from results/compute/GPU_HOURS.json.

Macros: GpuHours<F> for F in Primary, Xckpt, MedEnc, UnetSkip, Ft, Nnunet, RandInit, PerRater, PixelReplay, Heldout,
MaskAlign, KvasirOfficial, OfficialSplits, FeatureCache (1 decimal; "not recorded" when the family has no record), GpuHoursTotal (sum of the
family entries), GpuHoursDistinct (union across families per GPU), GpuModel (recorded nvidia-smi model name(s)), GpuCount.
"""
from __future__ import annotations

GPU_HOURS = "results/compute/GPU_HOURS.json"
FAMILIES = (("primary", "Primary"), ("xckpt", "Xckpt"), ("medenc", "MedEnc"), ("unet_skip", "UnetSkip"), ("ft", "Ft"),
            ("nnunet", "Nnunet"), ("randinit", "RandInit"), ("per_rater", "PerRater"), ("pixel_replay", "PixelReplay"),
            ("heldout", "Heldout"), ("mask_align", "MaskAlign"), ("kvasir_official", "KvasirOfficial"), ("official_splits", "OfficialSplits"), ("feature_cache", "FeatureCache"))
SCALARS = (("GpuHoursTotal", "total_gpu_hours_sum_of_families"), ("GpuHoursDistinct", "distinct_gpu_busy_hours_union_across_families"))


def input_paths() -> list[str]:
    return [GPU_HOURS]


def _hours(api, x) -> str:
    return "not recorded" if x is None else f"{api.num(x):.1f}"


def build(b, api) -> dict[str, str]:
    d = b.load(GPU_HOURS)
    for key, tok in FAMILIES:  # An explicit null records an unmeasured family; absent keys are errors.
        b.value(f"GpuHours{tok}", lambda key=key: _hours(api, api.opt(d, GPU_HOURS, "families", key, "gpu_hours")))
    for name, key in SCALARS:
        b.value(name, lambda key=key: _hours(api, api.dig(d, GPU_HOURS, key)))
    b.text("GpuModel", d, GPU_HOURS, "gpu_model")
    b.count("GpuCount", d, GPU_HOURS, "n_gpus")
    return {}
