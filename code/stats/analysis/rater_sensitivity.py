#!/usr/bin/env python3
"""RIGA per-rater ground-truth sensitivity from saved predictions.

The inference mode loads cached frozen features and saved 1x1 decoder
heads, forms each prediction once, and scores it against six individual masks
plus the 4-of-6 consensus.  The analysis mode applies the paper's per-FM
functional floor and symmetric seed-pair estimator, validates the task-specific
consensus result, and writes the requested JSON and Markdown table.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ANNOTATIONS = tuple([f"rater_{index}" for index in range(1, 7)] + ["consensus"])
BUNDLES = (
    "biomedclip",
    "clip_vitb16",
    "clip_vitl14",
    "convnext_tiny",
    "deit_vitb16",
    "dinov2_vitb14",
    "dinov2_vits14",
    "efficientnet_b0",
    "mae_vitb16",
    "resnet18",
    "resnet50",
)
SEEDS = (42, 43, 44, 45)
DINOV2_PAIR = frozenset({"dinov2_vitb14", "dinov2_vits14"})
FUNCTIONAL_FLOOR = 0.30
BOOTSTRAP_B = 2000
RANDOM_SEED = 0
TASK_CONFIG = {
    "riga_cup": {
        "display_name": "RIGA Cup",
        "mask_name": "Cup",
        "foreground_rule": "raw rater TIFF value == 128",
    },
    "riga_disc": {
        "display_name": "RIGA Disc",
        "mask_name": "Disc",
        "foreground_rule": "raw rater TIFF value >= 128",
    },
}
TASKS = tuple(TASK_CONFIG)
ARTIFACT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_STEMS = {
    "riga_cup": "L1D_per_rater_cup",
    "riga_disc": "L1D_per_rater_disc",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _torch_load(torch: Any, path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=True)


def _configure_deterministic_inference(torch: Any, seed: int) -> None:
    """Match the saved-head trainer's deterministic CUDA evaluation protocol."""
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _dice_on_device(torch: Any, predictions: Any, targets: Any) -> Any:
    pred = predictions.flatten(1).to(torch.float32)
    truth = targets.flatten(1).to(torch.float32)
    intersection = (pred * truth).sum(dim=1)
    denominator = pred.sum(dim=1) + truth.sum(dim=1)
    return torch.where(
        denominator > 0,
        2.0 * intersection / denominator,
        torch.ones_like(intersection),
    )


def _foreground_mask(array: np.ndarray, task: str) -> np.ndarray:
    if task not in TASK_CONFIG:
        raise ValueError(f"unsupported task: {task!r}")
    sys.path.insert(0, str(ARTIFACT_ROOT / "code" / "src"))
    from fmpool.datasets.riga import _target_pixels
    return _target_pixels(array, task.removeprefix("riga_"))


def _load_rater_masks(
    case_ids: list[str],
    rater_mask_root: Path,
    target_hw: tuple[int, int],
    task: str = "riga_cup",
    expected_hashes: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    import torch.nn.functional as functional
    from PIL import Image

    if expected_hashes is None:
        raise ValueError("authenticated rater mask hashes are required")
    per_rater: list[list[Any]] = [[] for _ in range(6)]
    raw_shapes: set[tuple[int, int]] = set()
    source_paths: list[str] = []
    consumed_hashes: dict[str, str] = {}
    for case_id in case_ids:
        relative = Path(case_id)
        for rater_index in range(1, 7):
            mask_path = (
                rater_mask_root
                / relative.parent
                / f"{relative.stem}-{rater_index}.tif"
            )
            if not mask_path.is_file():
                raise FileNotFoundError(f"missing rater mask: {mask_path}")
            content = mask_path.read_bytes()
            key = "RIGA-mask/" + str(mask_path.relative_to(rater_mask_root))
            actual_hash = hashlib.sha256(content).hexdigest()
            if expected_hashes.get(key) != actual_hash:
                raise ValueError(f"rater mask hash mismatch: {mask_path}")
            consumed_hashes[key] = actual_hash
            array = np.asarray(
                Image.open(io.BytesIO(content)).convert("L"),
                dtype=np.uint8,
            ).copy()
            raw_shapes.add(tuple(array.shape))
            mask = torch.from_numpy(_foreground_mask(array, task))
            resized = functional.interpolate(
                mask[None, None].to(torch.float32),
                size=target_hw,
                mode="nearest",
            )[0, 0].to(torch.bool)
            per_rater[rater_index - 1].append(resized)
            source_paths.append(str(mask_path))

    stacked = {
        f"rater_{index + 1}": torch.stack(masks)
        for index, masks in enumerate(per_rater)
    }
    votes = torch.stack([stacked[f"rater_{index}"] for index in range(1, 7)])
    stacked["consensus"] = votes.sum(dim=0) >= 4
    diagnostics = {
        "raw_mask_shapes": [list(shape) for shape in sorted(raw_shapes)],
        "n_mask_files": len(source_paths),
        "n_cases": len(case_ids),
        "target_hw": list(target_hw),
        "foreground_rule": TASK_CONFIG[task]["foreground_rule"],
        "majority_rule": "consensus = at least 4 of 6",
        "source_root": str(rater_mask_root),
        "mask_sha256": consumed_hashes,
    }
    return stacked, diagnostics


def run_inference(
    repo_root: Path,
    rater_mask_root: Path,
    output: Path,
    device_name: str,
    batch_size: int,
    bundles: tuple[str, ...] = BUNDLES,
    seeds: tuple[int, ...] = SEEDS,
    task: str = "riga_cup",
    *, cache_root: Path,
) -> dict[str, Any]:
    sys.path.insert(0, str(ARTIFACT_ROOT / "code" / "src"))
    from fmpool.determinism import numerical_failure_record

    with numerical_failure_record(output, task=task, stage="per_rater_replay",
                                  bundles=list(bundles), seeds=list(seeds)):
        return _run_inference(repo_root, rater_mask_root, output, device_name,
                              batch_size, bundles, seeds, task, cache_root=cache_root)


def _run_inference(
    repo_root: Path,
    rater_mask_root: Path,
    output: Path,
    device_name: str,
    batch_size: int,
    bundles: tuple[str, ...] = BUNDLES,
    seeds: tuple[int, ...] = SEEDS,
    task: str = "riga_cup",
    *, cache_root: Path,
) -> dict[str, Any]:
    """Run inference only; never trains or writes under the source results tree."""
    import torch
    sys.path.insert(0, str(ARTIFACT_ROOT / "code" / "src"))
    from fmpool import provenance
    from fmpool.decoder import LinearSegHead
    from fmpool.determinism import require_finite, sha256_file

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if task not in TASK_CONFIG:
        raise ValueError(f"unsupported task: {task!r}")
    device = torch.device(device_name)
    cache_root = cache_root / task
    checkpoint_root = repo_root / "checkpoints" / task
    trace_root = repo_root / "per_case_dice" / task

    if not bundles or not seeds:
        raise ValueError("inference requires at least one bundle and one seed")
    _configure_deterministic_inference(torch, seeds[0])
    first_trace = json.loads((trace_root / bundles[0] / f"seed_{seeds[0]}.json").read_text())
    first_sha = first_trace.get("cache_manifest_sha256")
    if not first_sha:
        raise ValueError("trace must pin its cache manifest")
    _, train_blob, first_blob = provenance.load_feature_cache(cache_root / bundles[0], expected_sha256=first_sha)
    del train_blob
    expected_mask_hashes = json.loads((cache_root / bundles[0] / "provenance.json").read_text())[
        "riga_test_palette"]["mask_sha256"]
    case_ids = [str(case_id) for case_id in first_blob["case_ids"]]
    target_hw = tuple(int(value) for value in first_blob["masks"].shape[-2:])
    ground_truths, mask_diagnostics = _load_rater_masks(
        case_ids,
        rater_mask_root,
        target_hw,
        task,
        expected_mask_hashes,
    )
    cached_consensus = first_blob["masks"].to(torch.bool)
    if cached_consensus.ndim == 4 and cached_consensus.shape[1] == 1:
        cached_consensus = cached_consensus[:, 0]
    disagreement = int(torch.count_nonzero(
        cached_consensus != ground_truths["consensus"]
    ).item())
    if disagreement:
        raise ValueError(
            f"reconstructed 4-of-6 consensus differs from cache at "
            f"{disagreement} pixels"
        )

    members: list[dict[str, Any]] = []
    max_trace_difference = 0.0
    prediction_count = 0
    for bundle in bundles:
        cache_path = cache_root / bundle / "test.pt"
        cache_sha = sha256_file(cache_path.parent / "manifest.json")
        manifest, train_blob, blob = provenance.load_feature_cache(cache_path.parent, expected_sha256=cache_sha)
        del train_blob
        bundle_mask_hashes = json.loads((cache_path.parent / "provenance.json").read_text())[
            "riga_test_palette"]["mask_sha256"]
        if bundle_mask_hashes != expected_mask_hashes:
            raise ValueError(f"{bundle}: rater source identity differs across bundles")
        if manifest["task"] != task or manifest["fm"] != bundle:
            raise ValueError("cache task/bundle mismatch")
        bundle_ids = [str(case_id) for case_id in blob["case_ids"]]
        if bundle_ids != case_ids:
            raise ValueError(f"{cache_path}: case_ids do not match the first cache")
        bundle_masks = blob["masks"].to(torch.bool)
        if bundle_masks.ndim == 4 and bundle_masks.shape[1] == 1:
            bundle_masks = bundle_masks[:, 0]
        bundle_disagreement = int(torch.count_nonzero(
            bundle_masks != cached_consensus
        ).item())
        if bundle_disagreement:
            raise ValueError(
                f"{cache_path}: cached masks differ at {bundle_disagreement} pixels"
            )
        features = blob["feats"]
        require_finite(features, "replay features")
        if features.dtype != torch.float32:
            raise ValueError(f"{cache_path}: v2 per-rater inference requires FP32 features")
        if features.shape[0] != len(case_ids):
            raise ValueError(f"{cache_path}: feature/case count mismatch")

        for seed in seeds:
            checkpoint_path = checkpoint_root / bundle / f"seed_{seed}.pt"
            trace_path = trace_root / bundle / f"seed_{seed}.json"
            trace = json.loads(trace_path.read_text())
            identity = trace.get("run_identity_v2", {})
            unsigned = {k: v for k, v in identity.items() if k != "identity_sha256"}
            if (not identity.get("primary_eligible") or identity.get("cache_identity") != manifest
                    or identity.get("identity_sha256") != provenance.digest(unsigned)
                    or trace.get("cache_manifest_sha256") != cache_sha
                    or trace.get("head_checkpoint_sha256") != sha256_file(checkpoint_path)):
                raise ValueError(f"{trace_path}: checkpoint/cache/run identity mismatch")
            if (trace.get("task"), trace.get("fm"), trace.get("seed")) != (task, bundle, seed):
                raise ValueError(f"{trace_path}: task/bundle/seed mismatch")
            state = _torch_load(torch, checkpoint_path)
            require_finite(state, f"replay checkpoint {checkpoint_path}")
            head = LinearSegHead(int(features.shape[1]), num_classes=1, out_size=target_hw).to(device)
            head.load_state_dict(state, strict=True)
            head.eval()
            scores = {annotation: [] for annotation in ANNOTATIONS}
            with torch.no_grad():
                for start in range(0, len(case_ids), batch_size):
                    end = min(start + batch_size, len(case_ids))
                    feature_batch = features[start:end].to(
                        device,
                        non_blocking=True,
                    )
                    logits = head(feature_batch)
                    require_finite(logits, "replay logits")
                    predictions = (torch.sigmoid(logits.float()) > 0.5).squeeze(1)
                    prediction_count += int(predictions.shape[0])
                    for annotation in ANNOTATIONS:
                        truth = ground_truths[annotation][start:end].to(
                            device,
                            non_blocking=True,
                        )
                        batch_scores = _dice_on_device(
                            torch,
                            predictions,
                            truth,
                        )
                        scores[annotation].extend(
                            float(value) for value in batch_scores.cpu().tolist()
                        )

            consensus_scores = np.asarray(scores["consensus"], dtype=float)
            if [str(case_id) for case_id in trace["test_ids"]] != case_ids:
                raise ValueError(f"{trace_path}: test_ids mismatch")
            trace_scores = np.asarray(trace["per_case_dice"], dtype=float)
            if trace_scores.shape != consensus_scores.shape or not np.isfinite(trace_scores).all():
                raise ValueError(f"{trace_path}: invalid reference Dice vector")
            trace_difference = float(np.max(np.abs(
                consensus_scores - trace_scores
            )))
            if trace_difference > 1e-7:
                raise RuntimeError(
                    f"{trace_path}: per-case consensus replay differs by "
                    f"{trace_difference:.12g}, exceeding 1e-7"
                )
            max_trace_difference = max(max_trace_difference, trace_difference)
            members.append(
                {
                    "bundle": bundle,
                    "seed": seed,
                    "case_ids": case_ids,
                    "dice": scores,
                    "mean_dice": {
                        annotation: float(np.mean(values))
                        for annotation, values in scores.items()
                    },
                    "consensus_trace_max_abs_difference": trace_difference,
                    "source_assets": {
                        "feature_cache": str(cache_path),
                        "decoder_checkpoint": str(checkpoint_path),
                        "released_consensus_trace": str(trace_path),
                    },
                }
            )

    payload = {
        "schema_version": "l1d_raw_v2",
        "generated_at": _now(),
        "task": task,
        "mode": "inference_only_from_cached_features_and_saved_decoder_heads",
        "device": str(device),
        "bundles": list(bundles),
        "seeds": list(seeds),
        "annotations": list(ANNOTATIONS),
        "n_bundles": len(bundles),
        "n_members": len(members),
        "n_cases": len(case_ids),
        "n_predictions_formed": prediction_count,
        "same_prediction_reused_across_all_annotations": True,
        "mask_diagnostics": {
            **mask_diagnostics,
            "cached_consensus_pixel_disagreement": disagreement,
            "cached_consensus_matches_reconstructed_4_of_6": True,
        },
        "released_consensus_trace_check": {
            "max_abs_per_case_dice_difference": max_trace_difference,
            "exact_match": max_trace_difference <= 1e-7,
            "status": "PASS",
            "note": (
                "All saved per-case consensus traces were reproduced within "
                "the 1e-7 acceptance tolerance."
            ),
        },
        "members": members,
    }
    _write_json(output, payload)
    return payload


def _member_map(
    raw: dict[str, Any],
    annotation: str,
) -> dict[tuple[str, int], np.ndarray]:
    output: dict[tuple[str, int], np.ndarray] = {}
    for member in raw["members"]:
        key = (str(member["bundle"]), int(member["seed"]))
        if key in output:
            raise ValueError(f"duplicate member: {key}")
        values = np.asarray(member["dice"][annotation], dtype=float)
        if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise ValueError(f"{key}/{annotation}: invalid Dice vector")
        output[key] = values
    lengths = {len(values) for values in output.values()}
    if len(lengths) != 1:
        raise ValueError(f"{annotation}: unaligned member lengths {lengths}")
    return output


def functional_floor_filter(
    per_case: dict[tuple[str, int], np.ndarray],
    floor: float = FUNCTIONAL_FLOOR,
) -> dict[tuple[str, int], np.ndarray]:
    """Paper convention: retain an entire FM by its cross-seed mean Dice."""
    means: dict[str, list[float]] = {}
    for (bundle, _seed), values in per_case.items():
        means.setdefault(bundle, []).append(float(np.mean(values)))
    kept = {
        bundle
        for bundle, seed_means in means.items()
        if float(np.mean(seed_means)) >= floor
    }
    return {key: values for key, values in per_case.items() if key[0] in kept}


def _pair_masks(
    keys: list[tuple[str, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pair_i, pair_j = np.triu_indices(len(keys), k=1)
    within = np.asarray(
        [
            keys[left][0] == keys[right][0]
            or frozenset({keys[left][0], keys[right][0]}) == DINOV2_PAIR
            for left, right in zip(pair_i, pair_j, strict=True)
        ],
        dtype=bool,
    )
    return pair_i, pair_j, within


def within_cross(
    per_case: dict[tuple[str, int], np.ndarray],
) -> dict[str, float | int]:
    keys = sorted(per_case)
    matrix = np.stack([per_case[key] for key in keys])
    correlation = np.corrcoef(matrix)
    pair_i, pair_j, within_mask = _pair_masks(keys)
    values = correlation[pair_i, pair_j]
    finite = np.isfinite(values)
    within_values = values[within_mask & finite]
    cross_values = values[~within_mask & finite]
    if not len(within_values) or not len(cross_values):
        raise ValueError("within/cross estimator has an empty stratum")
    within_mean = float(np.mean(within_values))
    cross_mean = float(np.mean(cross_values))
    return {
        "within_rbar": within_mean,
        "cross_rbar": cross_mean,
        "gap": within_mean - cross_mean,
        "n_within_pairs": int(len(within_values)),
        "n_cross_pairs": int(len(cross_values)),
    }


def case_bootstrap(
    per_case: dict[tuple[str, int], np.ndarray],
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    keys = sorted(per_case)
    matrix = np.stack([per_case[key] for key in keys])
    pair_i, pair_j, within_mask = _pair_masks(keys)
    bootstrap_values = {"within_rbar": [], "cross_rbar": [], "gap": []}
    for indices in bootstrap_indices:
        correlation = np.corrcoef(matrix[:, indices])
        values = correlation[pair_i, pair_j]
        finite = np.isfinite(values)
        within_values = values[within_mask & finite]
        cross_values = values[~within_mask & finite]
        if not len(within_values) or not len(cross_values):
            continue
        within_mean = float(np.mean(within_values))
        cross_mean = float(np.mean(cross_values))
        bootstrap_values["within_rbar"].append(within_mean)
        bootstrap_values["cross_rbar"].append(cross_mean)
        bootstrap_values["gap"].append(within_mean - cross_mean)

    intervals = {
        key: [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ]
        for key, values in bootstrap_values.items()
    }
    return {
        "B_requested": int(len(bootstrap_indices)),
        "B_valid": int(len(bootstrap_values["gap"])),
        "random_seed": RANDOM_SEED,
        "ci95": intervals,
    }


def analyze_annotation(
    per_case: dict[tuple[str, int], np.ndarray],
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    point = within_cross(per_case)
    bootstrap = case_bootstrap(per_case, bootstrap_indices)
    bundles = sorted({bundle for bundle, _seed in per_case})
    return {
        **point,
        "gap_case_bootstrap_95ci": bootstrap["ci95"]["gap"],
        "within_case_bootstrap_95ci": bootstrap["ci95"]["within_rbar"],
        "cross_case_bootstrap_95ci": bootstrap["ci95"]["cross_rbar"],
        "bootstrap": bootstrap,
        "functional_bundles": bundles,
        "n_bundles": len(bundles),
        "n_members": len(per_case),
    }


def _validation(
    consensus: dict[str, Any],
    task: str = "riga_cup",
) -> dict[str, Any]:
    if task not in TASK_CONFIG:
        raise ValueError(f"unsupported task: {task!r}")
    within = float(consensus["within_rbar"])
    cross = float(consensus["cross_rbar"])
    gap = float(consensus["gap"])
    valid = (all(math.isfinite(x) for x in (within, cross, gap))
             and -1.0 <= within <= 1.0 and -1.0 <= cross <= 1.0
             and -2.0 <= gap <= 2.0 and abs(gap - (within - cross)) <= 1e-12)
    return {"status": "PASS" if valid else "FAIL", "task": task,
            "target": "bounded valid correlation gap; exact source trace replay is checked separately"}


def _headline(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    consensus_gap = float(rows["consensus"]["gap"])
    rater_gaps = [float(rows[f"rater_{index}"]["gap"]) for index in range(1, 7)]
    lower = min(rater_gaps)
    upper = max(rater_gaps)
    inside = lower <= consensus_gap <= upper
    return {
        "consensus_gap": consensus_gap,
        "single_rater_gap_range": [lower, upper],
        "consensus_inside_single_rater_range": inside,
        "consensus_inflated_relative_to_all_single_raters": consensus_gap > upper,
        "amount_above_single_rater_max": max(0.0, consensus_gap - upper),
        "interpretation": (
            "The consensus-GT gap falls inside the range of the six "
            "single-rater gaps."
            if inside
            else (
                "The consensus-GT gap is inflated relative to all six "
                "single-rater gaps."
                if consensus_gap > upper
                else "The consensus-GT gap is below all six single-rater gaps."
            )
        ),
    }


def build_payload(
    raw: dict[str, Any],
    task: str | None = None,
) -> dict[str, Any]:
    if raw.get("schema_version") != "l1d_raw_v2":
        raise ValueError("per-rater raw schema l1d_raw_v2 is required")
    raw_task = str(raw.get("task"))
    task = raw_task if task is None else task
    if task not in TASK_CONFIG:
        raise ValueError(f"unsupported task: {task!r}")
    if raw_task != task:
        raise ValueError(f"raw task {raw_task!r} does not match --task {task!r}")
    if raw.get("n_members") != 44 or raw.get("n_bundles") != 11:
        raise ValueError(
            f"raw inference does not contain the complete 11x4 {task} pool"
        )
    if raw.get("n_cases") != 95 or raw.get("n_predictions_formed") != 4180:
        raise ValueError(
            "raw inference must contain 95 cases and 4180 predictions"
        )
    if raw.get("mask_diagnostics", {}).get("n_mask_files") != 570:
        raise ValueError("raw inference must contain exactly 570 rater masks")
    if raw["mask_diagnostics"].get("foreground_rule") != TASK_CONFIG[task]["foreground_rule"]:
        raise ValueError("raw foreground rule does not match the task")
    trace_check = raw.get("released_consensus_trace_check", {})
    trace_difference = float(
        trace_check.get("max_abs_per_case_dice_difference", math.inf)
    )
    if (
        trace_check.get("status") != "PASS"
        or not trace_check.get("exact_match")
        or not math.isfinite(trace_difference)
        or not 0 <= trace_difference <= 1e-7
    ):
        raise RuntimeError(
            "released consensus per-case trace replay gate failed: "
            + json.dumps(trace_check, sort_keys=True)
        )
    expected_members = {(bundle, seed) for bundle in BUNDLES for seed in SEEDS}
    if len(raw.get("members", [])) != len(expected_members):
        raise ValueError("member count does not match the declared complete pool")
    canonical_ids = raw["members"][0].get("case_ids", [])
    if len(canonical_ids) != raw["n_cases"] or len(set(canonical_ids)) != len(canonical_ids):
        raise ValueError("case IDs must be unique and match the declared case count")
    for member in raw["members"]:
        if member.get("case_ids") != canonical_ids:
            raise ValueError("member case IDs are not aligned")
    maps = {annotation: _member_map(raw, annotation) for annotation in ANNOTATIONS}
    if any(set(values) != expected_members for values in maps.values()):
        raise ValueError("member identities do not match the declared complete pool")
    if any(len(vector) != len(canonical_ids) for values in maps.values() for vector in values.values()):
        raise ValueError("Dice vector length does not match case IDs")
    n_cases = len(next(iter(maps["consensus"].values())))
    consensus_per_case = functional_floor_filter(maps["consensus"])
    validation = _validation(within_cross(consensus_per_case), task)
    if validation["status"] != "PASS":
        raise RuntimeError(
            "consensus validation gate failed: "
            + json.dumps(validation, sort_keys=True)
        )

    rng = np.random.default_rng(RANDOM_SEED)
    bootstrap_indices = rng.integers(
        0,
        n_cases,
        size=(BOOTSTRAP_B, n_cases),
    )

    gt_specific_rows = {
        annotation: analyze_annotation(
            functional_floor_filter(maps[annotation]),
            bootstrap_indices,
        )
        for annotation in ANNOTATIONS
    }

    consensus_pool = set(gt_specific_rows["consensus"]["functional_bundles"])
    fixed_pool_rows = {
        annotation: analyze_annotation(
            {
                key: values
                for key, values in maps[annotation].items()
                if key[0] in consensus_pool
            },
            bootstrap_indices,
        )
        for annotation in ANNOTATIONS
    }
    trace_exact = trace_difference <= 1e-7
    return {
        "generated_at": _now(),
        "status": "PASS",
        "task": task,
        "protocol": {
            "prediction_source": (
                "cached test features plus saved 1x1 decoder heads; inference only"
            ),
            "same_prediction_scored_against_all_ground_truths": True,
            "ground_truths": list(ANNOTATIONS),
            "foreground_rule": TASK_CONFIG[task]["foreground_rule"],
            "consensus_rule": "at least 4 of 6 raters",
            "functional_floor": FUNCTIONAL_FLOOR,
            "functional_floor_scope": (
                "per FM, using the cross-seed mean Dice; primary rows apply "
                "the floor separately under each ground truth"
            ),
            "pairing": (
                "all unordered member pairs; same bundle and DINOv2-B/S are "
                "within, all remaining bundle pairs are cross"
            ),
            "bootstrap": (
                "case-only percentile bootstrap with one shared set of case "
                "resamples across annotations"
            ),
            "bootstrap_B": BOOTSTRAP_B,
            "random_seed": RANDOM_SEED,
        },
        "pool": {
            "n_bundles_before_floor": raw["n_bundles"],
            "n_members_before_floor": raw["n_members"],
            "n_cases": raw["n_cases"],
            "bundles_before_floor": raw["bundles"],
            "seeds": raw["seeds"],
        },
        "inference_sanity": {
            "mask_diagnostics": raw["mask_diagnostics"],
            "released_consensus_trace_check": {
                "max_abs_per_case_dice_difference": trace_difference,
                "exact_match": trace_exact,
                "status": "PASS",
                "note": (
                    "All saved per-case consensus traces were reproduced within "
                    "the 1e-7 acceptance tolerance."
                ),
            },
            "same_prediction_reused_across_all_annotations": raw[
                "same_prediction_reused_across_all_annotations"
            ],
        },
        "validation_gate": validation,
        "primary_gt_specific_floor": {
            "headline": _headline(gt_specific_rows),
            "annotations": gt_specific_rows,
        },
        "fixed_consensus_pool_sensitivity": {
            "purpose": (
                "Annotation-source sensitivity with the consensus-selected "
                "functional bundle set held fixed."
            ),
            "functional_bundles": sorted(consensus_pool),
            "headline": _headline(fixed_pool_rows),
            "annotations": fixed_pool_rows,
        },
        "source": {
            "raw_inference_schema": raw["schema_version"],
            "raw_generated_at": raw["generated_at"],
            "raw_device": raw["device"],
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    task = str(payload["task"])
    if task not in TASK_CONFIG:
        raise ValueError(f"unsupported task: {task!r}")
    config = TASK_CONFIG[task]
    primary = payload["primary_gt_specific_floor"]
    fixed = payload["fixed_consensus_pool_sensitivity"]
    lines = [
        f"# {config['display_name']} per-rater ground-truth sensitivity",
        "",
        (
            f"**Headline:** {primary['headline']['interpretation']} "
            f"Consensus Δ = {primary['headline']['consensus_gap']:.3f}; "
            "six-rater range = "
            f"[{primary['headline']['single_rater_gap_range'][0]:.3f}, "
            f"{primary['headline']['single_rater_gap_range'][1]:.3f}]."
        ),
        "",
        (
            "| Ground truth | GT-specific functional pool | Within | Cross | "
            "Gap [95% case CI] | Gap on fixed consensus pool [95% case CI] |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        **{f"rater_{index}": f"Rater {index}" for index in range(1, 7)},
        "consensus": "4-of-6 consensus",
    }
    for annotation in ANNOTATIONS:
        row = primary["annotations"][annotation]
        fixed_row = fixed["annotations"][annotation]
        ci = row["gap_case_bootstrap_95ci"]
        fixed_ci = fixed_row["gap_case_bootstrap_95ci"]
        lines.append(
            f"| {labels[annotation]} | {row['n_bundles']} bundles / "
            f"{row['n_members']} members | {row['within_rbar']:.3f} | "
            f"{row['cross_rbar']:.3f} | {row['gap']:.3f} "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] | {fixed_row['gap']:.3f} "
            f"[{fixed_ci[0]:.3f}, {fixed_ci[1]:.3f}] |"
        )
    fixed_headline = fixed["headline"]
    lines.extend(
        [
            "",
            (
                "The fixed-pool sensitivity is: "
                f"{fixed_headline['interpretation']} Consensus Δ = "
                f"{fixed_headline['consensus_gap']:.3f}; single-rater range = "
                f"[{fixed_headline['single_rater_gap_range'][0]:.3f}, "
                f"{fixed_headline['single_rater_gap_range'][1]:.3f}]."
            ),
            "",
            (
                "Protocol: identical saved predictions scored against six "
                f"individual {config['mask_name']} masks and their 4-of-6 "
                f"consensus; {payload['protocol']['foreground_rule']}; "
                "Dice ≥ 0.30 "
                "per-FM floor; symmetric four-seed pairing; case bootstrap "
                f"B={BOOTSTRAP_B}."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inference = subparsers.add_parser("inference")
    inference.add_argument("--task", choices=TASKS, required=True)
    inference.add_argument("--repo-root", type=Path, required=True, help="validated run root containing checkpoints/ and per_case_dice/")
    inference.add_argument("--cache-root", type=Path, required=True)
    inference.add_argument("--rater-mask-root", type=Path, required=True)
    inference.add_argument("--output", type=Path, required=True)
    inference.add_argument("--device", default="cuda:0")
    inference.add_argument("--batch-size", type=int, default=16)
    inference.add_argument(
        "--bundles",
        nargs="+",
        choices=BUNDLES,
        default=list(BUNDLES),
    )
    inference.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--task", choices=TASKS, required=True)
    analyze.add_argument("--raw-input", type=Path, required=True)
    analyze.add_argument("--output-json", type=Path)
    analyze.add_argument("--output-markdown", type=Path)
    return parser.parse_args(argv)


def main() -> int:
    args = _parse_args()
    if args.command == "inference":
        if args.output.exists():
            print(f"STOP: {args.output} exists; outputs are never overwritten", file=sys.stderr)
            return 2
        payload = run_inference(
            args.repo_root,
            args.rater_mask_root,
            args.output,
            args.device,
            args.batch_size,
            tuple(args.bundles),
            tuple(args.seeds),
            args.task,
            cache_root=args.cache_root,
        )
        print(
            f"PASS: {args.output} "
            f"({payload['n_members']} members x {payload['n_cases']} cases)"
        )
        return 0

    raw_input = args.raw_input
    output_root = ARTIFACT_ROOT / "build/stats"
    output_json = args.output_json or output_root / f"{OUTPUT_STEMS[args.task]}.json"
    output_markdown = (
        args.output_markdown or output_root / f"{OUTPUT_STEMS[args.task]}.md"
    )
    for existing in (output_json, output_markdown):
        if existing.exists():
            print(f"STOP: {existing} exists; outputs are never overwritten", file=sys.stderr)
            return 2
    with raw_input.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    try:
        payload = build_payload(raw, args.task)
    except RuntimeError as error:
        print(f"STOP: {error}", file=sys.stderr)
        return 2
    _write_json(output_json, payload)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(render_markdown(payload), encoding="utf-8")
    print(
        f"PASS: {output_json}; {output_markdown}; "
        f"{payload['primary_gt_specific_floor']['headline']['interpretation']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
