#!/usr/bin/env python3
"""Summarise case-identical nnU-Net held-out traces (RIGA 2D, ACDC 3D).

The input is the output directory produced by ``run_nnunet.sh
case_riga_100ep`` or by ``run_nnunet_case_worker.sh``:

    results/nnunet/nnunet_case_riga_2d_100ep/riga_cup_fold{F}_seed{S}.json
    results/nnunet/nnunet_case_riga_2d_1000ep/riga_cup_fold{F}_seed{S}.json  (1000-epoch companion, not reported)

Each file contains per-case Dice on the same Magrabia held-out cases used by
the primary frozen-encoder RIGA rows. The summary computes same-fold
different-seed and cross-fold pairwise Pearson correlations, with a fixed
case-bootstrap interval for the gap. ACDC traces (``acdc_lv_fold{F}_seed{S}.json``
from ``CASE_TASK=acdc_lv``) are scored on the primary test patients; their
bootstrap resamples patients (ED and ES frames together). The summary also
embeds the M_eff analysis (P33 primary) for the 5-fold ensembles.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger("fmpool.compute_nnunet_case_identical")
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 0
EXPECTED_FOLDS = tuple(range(5))
EXPECTED_SEEDS = (13, 37)
FILE_RE = re.compile(r"^(?P<task>riga_cup|acdc_lv)_fold(?P<fold>[0-4])_seed(?P<seed>13|37)\.json$")
TASK_TEXT = {
    "riga_cup": dict(dataset="RIGA_cup", case_unit="fundus image", test_source="Magrabia",
                     label="RIGA Cup nnU-Net v2 2D", train="BinRushed+MESSIDOR"),
    "acdc_lv": dict(dataset="ACDC_LV", case_unit="cine frame volume (ED/ES)",
                    test_source="held-out ACDC test patients (split recorded in the traces' run identity)",
                    label="ACDC LV nnU-Net v2 3d_fullres", train="the primary training patients"),
}


@dataclass(frozen=True)
class Trace:
    path: Path
    task: str
    fm: str
    fold: int
    seed: int
    ids: tuple[str, ...]
    dice: np.ndarray
    mean_dice: float
    elapsed_s: float | None


def _git_commit_sha(cwd: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or "not_recorded_for_artifact"
    except Exception:  # noqa: BLE001
        return "not_recorded_for_artifact"


def _iso_utc_now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _strict(value):
    if isinstance(value, dict):
        return {k: _strict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(v) for v in value]
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _load(path: Path) -> Trace:
    match = FILE_RE.match(path.name)
    if match is None:
        raise ValueError(
            f"{path}: expected canonical filename "
            "{riga_cup|acdc_lv}_fold{0..4}_seed{13|37}.json"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"task", "fm", "fold", "seed", "n_test", "test_ids", "per_case_dice"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"{path}: missing keys {sorted(missing)}")
    if payload["task"] != match.group("task"):
        raise ValueError(f"{path}: filename task disagrees with payload task {payload['task']!r}")
    fold = int(payload["fold"])
    seed = int(payload["seed"])
    if fold != int(match.group("fold")) or seed != int(match.group("seed")):
        raise ValueError(
            f"{path}: filename fold/seed disagree with payload "
            f"fold={fold} seed={seed}"
        )
    ids = tuple(str(x) for x in payload["test_ids"])
    dice = np.asarray(payload["per_case_dice"], dtype=np.float64)
    if int(payload["n_test"]) != len(ids) or len(ids) != len(dice):
        raise ValueError(
            f"{path}: inconsistent lengths n_test={payload['n_test']} "
            f"len(test_ids)={len(ids)} len(per_case_dice)={len(dice)}"
        )
    if len(dice) == 0 or not np.all(np.isfinite(dice)):
        raise ValueError(f"{path}: per_case_dice must be nonempty and finite")
    return Trace(
        path=path,
        task=str(payload["task"]),
        fm=str(payload["fm"]),
        fold=fold,
        seed=seed,
        ids=ids,
        dice=dice,
        mean_dice=float(payload.get("mean_dice", np.mean(dice))),
        elapsed_s=(
            float(payload["training_elapsed_s"])
            if payload.get("training_elapsed_s") is not None
            else None
        ),
    )


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _point(traces: list[Trace], idx: np.ndarray | None = None) -> dict:
    within: list[float] = []
    cross: list[float] = []
    for i, a in enumerate(traces):
        av = a.dice if idx is None else a.dice[idx]
        for b in traces[i + 1:]:
            bv = b.dice if idx is None else b.dice[idx]
            r = _pearson(av, bv)
            if math.isnan(r):
                continue
            (within if a.fold == b.fold else cross).append(r)
    return {
        "within_rbar": float(np.mean(within)) if within else float("nan"),
        "cross_rbar": float(np.mean(cross)) if cross else float("nan"),
        "gap": (
            float(np.mean(within) - np.mean(cross))
            if within and cross else float("nan")
        ),
        "n_within_pairs": len(within),
        "n_cross_pairs": len(cross),
        "within_pairs": within,
        "cross_pairs": cross,
    }


def _cluster_ids(ids, task: str) -> list[str]:
    """Independent unit: the ACDC patient; every RIGA image is its own cluster."""
    return [cid.split("_frame")[0] for cid in ids] if task == "acdc_lv" else list(ids)


def _cluster_groups(ids, task: str) -> list[list[int]]:
    """Case positions per cluster, in first-appearance order (RIGA: [[0], [1], ...])."""
    groups: dict[str, list[int]] = {}
    for index, cluster in enumerate(_cluster_ids(ids, task)):
        groups.setdefault(cluster, []).append(index)
    return list(groups.values())


def _case_bootstrap_gap(traces: list[Trace], groups: list[list[int]]) -> list[float]:
    # With singleton groups this draws exactly the historical case-bootstrap indices.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    gaps: list[float] = []
    for _ in range(BOOTSTRAP_B):
        picked = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in picked])
        gap = _point(traces, idx=idx)["gap"]
        if math.isfinite(gap):
            gaps.append(float(gap))
    if not gaps:
        return [float("nan"), float("nan")]
    lo, hi = np.percentile(np.asarray(gaps), [2.5, 97.5])
    return [float(lo), float(hi)]


def _protocol_label(input_dir: Path, trainer_epochs: int | None = None) -> str:
    if trainer_epochs is not None:
        return f"{trainer_epochs}-epoch"
    name = input_dir.name
    if "1000ep" in name:
        return "1000-epoch"
    if "100ep" in name:
        return "100-epoch"
    return "seeded"


def _epochs_from_name(input_dir: Path) -> int | None:
    """Fallback when --trainer-epochs is not given: read 100ep/1000ep from the directory name."""
    if "1000ep" in input_dir.name:
        return 1000
    if "100ep" in input_dir.name:
        return 100
    return None


def _expected_fm(input_dir: Path, task: str = "riga_cup", trainer_epochs: int | None = None) -> str:
    epochs = trainer_epochs if trainer_epochs is not None else _epochs_from_name(input_dir)
    prefix = "nnunet_2d" if task == "riga_cup" else "nnunet_3d_fullres"
    if epochs == 1000:
        return "nnunet_2d_1000ep" if task == "riga_cup" else prefix
    if epochs == 100:
        return prefix + "_100ep"
    raise ValueError(
        f"cannot infer expected nnU-Net fm label from input directory {input_dir}; pass --trainer-epochs"
    )


def _effective_size_summary(traces: list[Trace], task: str) -> dict:
    """M_eff (pooled P33 primary) for the five-fold and combined-seed pools."""
    stats_root = Path(__file__).resolve().parents[1] / "stats"
    if str(stats_root) not in sys.path:
        sys.path.insert(0, str(stats_root))
    from dataclasses import replace
    from analysis import effective_size

    collection = effective_size._collection_from_embedded(
        config="nnunet_case_identical", task=task,
        member_map={f"fold{tr.fold}_seed{tr.seed}": tr.dice.tolist() for tr in traces},
        case_ids=traces[0].ids, provenance=traces[0].path.parent,
        evidence_scope="case-identical nnU-Net")
    if task == "acdc_lv":
        collection = replace(collection, cluster_ids=tuple(_cluster_ids(traces[0].ids, task)))
    members = collection.member_ids
    seed_idx = {s: [i for i, m in enumerate(members) if m.endswith(f"seed{s}")] for s in EXPECTED_SEEDS}
    return effective_size.analyze_collection(
        collection, b=BOOTSTRAP_B, seed=BOOTSTRAP_SEED + (1_000_000 if task == "riga_cup" else 1_100_000),
        include_diverse4=False, include_full=False,
        fixed_subsets=(("M5_seed13_primary", seed_idx[13]), ("M5_seed37_sensitivity", seed_idx[37]),
                       ("M10_both_seeds", tuple(range(len(members))))))


def summarise(input_dir: Path, output: Path, repo_root: Path, *, with_meff: bool = True,
              trainer_epochs: int | None = None) -> dict:
    traces = [_load(p) for p in sorted(input_dir.glob("*_fold*_seed*.json"))]
    if not traces:
        raise FileNotFoundError(f"no <task>_fold*_seed*.json under {input_dir}")
    tasks = {tr.task for tr in traces}
    if len(tasks) != 1:
        raise ValueError(f"mixed task traces under {input_dir}: {sorted(tasks)}")
    task = tasks.pop()
    text = TASK_TEXT[task]
    expected_cells = {(fold, seed) for fold in EXPECTED_FOLDS for seed in EXPECTED_SEEDS}
    counts = Counter((tr.fold, tr.seed) for tr in traces)
    duplicates = sorted(cell for cell, count in counts.items() if count != 1)
    if duplicates:
        raise ValueError(
            "duplicate case-identical nnU-Net cells are not allowed: "
            f"{duplicates}"
        )
    observed_cells = set(counts)
    if observed_cells != expected_cells:
        missing = sorted(expected_cells - observed_cells)
        extra = sorted(observed_cells - expected_cells)
        raise ValueError(
            "incomplete or unexpected case-identical nnU-Net grid: "
            f"missing={missing}, extra={extra}, observed={sorted(observed_cells)}"
        )
    expected_fm = _expected_fm(input_dir, task, trainer_epochs)
    bad_fm = sorted((str(tr.path), tr.fm) for tr in traces if tr.fm != expected_fm)
    if bad_fm:
        raise ValueError(f"unexpected fm labels for {input_dir}: {bad_fm}")
    first_ids = traces[0].ids
    for tr in traces:
        if tr.ids != first_ids:
            raise ValueError(f"test_ids mismatch: {tr.path} disagrees with {traces[0].path}")
    point = _point(traces)
    groups = _cluster_groups(first_ids, task)
    protocol_label = _protocol_label(input_dir, trainer_epochs)
    payload = {
        "schema_version": "fmpool_clean_v1",
        "generated_at": _iso_utc_now(),
        "commit_sha": _git_commit_sha(repo_root),
        "protocol": (
            f"Case-identical {text['label']} {protocol_label} scope check; "
            f"train folds use {text['train']} and held-out evaluation uses "
            f"the {text['test_source']} cases."
        ),
        "dataset": text["dataset"],
        "task": task,
        "fm": expected_fm,
        "trainer_epochs": (trainer_epochs if trainer_epochs is not None
                           else 1000 if "1000ep" in input_dir.name else 100),
        "case_unit": text["case_unit"],
        "n_test": len(first_ids),
        "n_test_clusters": len(groups),
        "test_source": text["test_source"],
        "n_models": len(traces),
        "n_models_trained": len(traces),
        "folds": sorted({tr.fold for tr in traces}),
        "seeds": sorted({tr.seed for tr in traces}),
        "within_fold_rbar": point["within_rbar"],
        "cross_fold_rbar": point["cross_rbar"],
        "delta_within_minus_cross": point["gap"],
        "delta_ci95_case_bootstrap": _case_bootstrap_gap(traces, groups),
        "delta_ci95_method": (
            "case bootstrap over held-out cases" if task == "riga_cup" else
            "patient-cluster bootstrap over held-out patients (ED/ES frames resampled together)"),
        "bootstrap_unit": "case" if task == "riga_cup" else "patient cluster",
        "bootstrap_b": BOOTSTRAP_B,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "n_within_pairs": point["n_within_pairs"],
        "n_cross_pairs": point["n_cross_pairs"],
        "mean_dice_all": float(np.mean([tr.mean_dice for tr in traces])),
        "mean_dice_std": float(np.std([tr.mean_dice for tr in traces], ddof=1))
        if len(traces) > 1 else 0.0,
        "mean_dice_per_model": {
            f"fold{tr.fold}_seed{tr.seed}": tr.mean_dice for tr in traces
        },
        "case_ids": list(first_ids),
        "per_case_dice": {
            f"fold{tr.fold}_seed{tr.seed}": [float(x) for x in tr.dice]
            for tr in traces
        },
        "training_elapsed_s_per_model": {
            f"fold{tr.fold}_seed{tr.seed}": tr.elapsed_s for tr in traces
        },
        "within_pairs": point["within_pairs"],
        "cross_pairs": point["cross_pairs"],
        "note": (
            f"Case-identical full-pipeline scope check on the {text['test_source']} "
            f"held-out cases. This is a single-task {text['label']} "
            "diagnostic, not an official nnU-Net benchmark or a replacement "
            "for the frozen-encoder primary evidence."
        ),
        "source_files": [str(p.relative_to(repo_root)) if p.is_relative_to(repo_root) else str(p)
                         for p in sorted(tr.path for tr in traces)],
    }
    if with_meff:
        payload["m_eff_e1"] = _effective_size_summary(traces, task)  # key name read by table_generation
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/nnunet/nnunet_case_riga_2d_100ep"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/nnunet/riga_case_identical_2d_100ep.json"),
    )
    parser.add_argument("--trainer-epochs", type=int, choices=(100, 1000), default=None,
                        help="nnU-Net training length; if omitted, read from a 100ep/1000ep substring of --input-dir")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(),
                        help="root against which source_files are recorded relative")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    payload = summarise(args.input_dir, args.out, args.repo_root, trainer_epochs=args.trainer_epochs)
    LOGGER.info(
        "wrote %s: n=%d models=%d gap=%.4f",
        args.out,
        payload["n_test"],
        payload["n_models"],
        payload["delta_within_minus_cross"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
