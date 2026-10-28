#!/usr/bin/env python3
"""Replay one primary cell (task, fm, 4 seeds) from its FP32 feature cache and saved
LinearSegHead checkpoints; gate on the saved per-case Dice trace; write packed error masks.

Inference only.  Loading/identity checks are copied from
code/stats/analysis/rater_sensitivity.py::_run_inference (same cache loader, same checkpoint/trace
identity checks, same deterministic CUDA settings, same Dice).  GT = the cache's own test masks,
which is what train_head_cached.py scored against.

Per seed writes OUT/<task>__<fm>/seed_<s>.npz (+ .sha256):
  case_ids, dice (replayed), trace_dice, err_packed = packbits(pred XOR gt) [N, 6272] (all cases;
  brats_wt: panel cases only), gt_packed (same rows), panel_idx/panel_case_ids/panel_pred/panel_gt.
Panel list (deterministic): test cases with GT area > 0 sorted by (area, case_id); the middle 8.
The cell aborts (FAIL receipt, no npz) if any seed's replayed Dice differs from its trace by > 1e-7.
usage: evaluate_error_masks.py --source SRC --cell CELL_DIR --cache-root CACHE --task T --fm FM --out OUT
       [--device cuda:0] [--batch-size 16] [--data-root DATA  (optional panel input images)]
"""
import argparse, hashlib, io, json, os, sys, time
from pathlib import Path

import numpy as np

SEEDS = (42, 43, 44, 45)
GATE = 1e-7
N_PANEL = 8
PANEL_ONLY_TASKS = {"brats_wt"}


def panel_indices(areas, case_ids):
    cand = sorted((int(a), str(c), i) for i, (a, c) in enumerate(zip(areas, case_ids)) if a > 0)
    mid = len(cand) // 2
    return [i for _, _, i in cand[max(0, mid - N_PANEL // 2): mid + N_PANEL // 2]]


def write_npz(path, **arrays):
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    data = buf.getvalue()
    with open(path, "xb") as fh:
        fh.write(data)
    sha = hashlib.sha256(data).hexdigest()
    Path(str(path) + ".sha256").write_text(f"{sha}  {path.name}\n")
    return sha


def panel_images(task, case_ids, idx, gt, data_root):
    """Optional: raw test images for the panel cases, resized like extract_features (no norm)."""
    import torch, torch.nn.functional as F
    os.environ["FMPOOL_DATA_ROOT"] = str(data_root)
    from fmpool import datasets as fmpool_datasets
    hw = tuple(gt.shape[-2:])

    def resize(image, mask):
        img = image.float()
        img = F.interpolate(img.unsqueeze(0), size=hw, mode="bilinear", align_corners=False).squeeze(0)
        m = mask.unsqueeze(0) if mask.ndim == 2 else mask
        m = F.interpolate(m.unsqueeze(0).float(), size=hw, mode="nearest").squeeze(0)
        return img, m
    ds = fmpool_datasets.build_dataset(task, "test")
    ds.transform = resize
    out = []
    for k in idx:
        img, m, cid = ds[k]
        if str(cid) != case_ids[k]:
            raise ValueError(f"dataset order differs at {k}: {cid} vs {case_ids[k]}")
        if not np.array_equal(m.squeeze(0).numpy() > 0.5, gt[k]):
            raise ValueError(f"{cid}: dataset mask differs from cached mask")
        a = img.numpy().transpose(1, 2, 0)
        a = (a - a.min()) / max(float(a.max() - a.min()), 1e-12)
        out.append((a * 255).round().astype(np.uint8))
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    for k in ("source", "cell", "cache-root", "task", "fm", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--data-root", default=None)
    a = ap.parse_args()
    src = Path(a.source)
    sys.path.insert(0, str(src / "code" / "src"))
    sys.path.insert(0, str(src / "code" / "stats"))
    import torch
    from fmpool import provenance
    from fmpool.decoder import LinearSegHead
    from fmpool.determinism import require_finite, sha256_file
    from analysis.rater_sensitivity import _configure_deterministic_inference, _dice_on_device, _torch_load

    t0 = time.time()
    out_dir = Path(a.out) / f"{a.task}__{a.fm}"
    out_dir.mkdir(parents=True, exist_ok=False)
    receipt = {"task": a.task, "fm": a.fm, "source": str(src), "cell": a.cell, "cache_root": a.cache_root,
               "device": a.device, "torch": torch.__version__, "gate": GATE, "seeds": {}, "status": "FAIL"}
    try:
        if a.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        device = torch.device(a.device)
        _configure_deterministic_inference(torch, SEEDS[0])
        cache_dir = Path(a.cache_root) / a.task / a.fm
        cache_sha = sha256_file(cache_dir / "manifest.json")
        manifest, train_blob, blob = provenance.load_feature_cache(cache_dir, expected_sha256=cache_sha)
        del train_blob
        if manifest["task"] != a.task or manifest["fm"] != a.fm:
            raise ValueError("cache task/bundle mismatch")
        case_ids = [str(c) for c in blob["case_ids"]]
        gt_t = blob["masks"].to(torch.bool)
        if gt_t.ndim == 4 and gt_t.shape[1] == 1:
            gt_t = gt_t[:, 0]
        feats = blob["feats"]
        require_finite(feats, "replay features")
        if feats.dtype != torch.float32 or feats.shape[0] != len(case_ids):
            raise ValueError("replay requires FP32 features aligned with case_ids")
        gt = gt_t.numpy()
        hw = tuple(int(v) for v in gt.shape[-2:])
        pidx = panel_indices(gt.reshape(len(gt), -1).sum(1), case_ids)
        rows = np.asarray(pidx if a.task in PANEL_ONLY_TASKS else range(len(case_ids)), dtype=np.int64)
        receipt.update(cache_manifest_sha256=cache_sha, n_cases=len(case_ids), target_hw=hw,
                       panel_idx=pidx, panel_case_ids=[case_ids[i] for i in pidx],
                       panel_gt_area=[int(gt[i].sum()) for i in pidx],
                       rows_saved="panel_only" if a.task in PANEL_ONLY_TASKS else "all")
        results = {}
        for seed in SEEDS:
            ckpt = Path(a.cell) / "checkpoints" / a.task / a.fm / f"seed_{seed}.pt"
            trace_path = Path(a.cell) / "per_case_dice" / a.task / a.fm / f"seed_{seed}.json"
            trace = json.loads(trace_path.read_text())
            identity = trace.get("run_identity_v2", {})
            unsigned = {k: v for k, v in identity.items() if k != "identity_sha256"}
            ckpt_sha = sha256_file(ckpt)
            if (not identity.get("primary_eligible") or identity.get("cache_identity") != manifest
                    or identity.get("identity_sha256") != provenance.digest(unsigned)
                    or trace.get("cache_manifest_sha256") != cache_sha
                    or trace.get("head_checkpoint_sha256") != ckpt_sha):
                raise ValueError(f"{trace_path}: checkpoint/cache/run identity mismatch")
            if (trace.get("task"), trace.get("fm"), trace.get("seed")) != (a.task, a.fm, seed):
                raise ValueError(f"{trace_path}: task/bundle/seed mismatch")
            if [str(c) for c in trace["test_ids"]] != case_ids:
                raise ValueError(f"{trace_path}: test_ids mismatch")
            state = _torch_load(torch, ckpt)
            require_finite(state, f"replay checkpoint {ckpt}")
            head = LinearSegHead(int(feats.shape[1]), num_classes=1, out_size=hw).to(device)
            head.load_state_dict(state, strict=True)
            head.eval()
            preds, dice = [], []
            with torch.no_grad():
                for s in range(0, len(case_ids), a.batch_size):
                    logits = head(feats[s:s + a.batch_size].to(device))
                    require_finite(logits, "replay logits")
                    p = (torch.sigmoid(logits.float()) > 0.5).squeeze(1)
                    dice.append(_dice_on_device(torch, p, gt_t[s:s + a.batch_size].to(device)).cpu())
                    preds.append(p.cpu())
            pred = torch.cat(preds).numpy()
            dice = torch.cat(dice).double().numpy()
            tdice = np.asarray(trace["per_case_dice"], dtype=float)
            diff = float(np.max(np.abs(dice - tdice)))
            receipt["seeds"][str(seed)] = {"max_abs_trace_diff": diff, "mean_dice": float(dice.mean()),
                                           "checkpoint": str(ckpt), "checkpoint_sha256": ckpt_sha,
                                           "trace": str(trace_path)}
            if not np.isfinite(diff) or diff > GATE:
                raise RuntimeError(f"seed {seed}: replayed per-case Dice differs from trace by {diff:.3g} > {GATE}")
            results[seed] = (pred, dice, tdice)
        # gate passed for all 4 seeds -> write
        n_px = hw[0] * hw[1]
        gt_rows = gt[rows].reshape(len(rows), n_px)
        for seed, (pred, dice, tdice) in results.items():
            err = np.logical_xor(pred[rows].reshape(len(rows), n_px), gt_rows)
            sha = write_npz(out_dir / f"seed_{seed}.npz", task=a.task, fm=a.fm, seed=seed,
                            case_ids=np.asarray(case_ids), rows=rows, hw=np.asarray(hw),
                            dice=dice, trace_dice=tdice,
                            err_packed=np.packbits(err, axis=1), gt_packed=np.packbits(gt_rows, axis=1),
                            panel_idx=np.asarray(pidx), panel_case_ids=np.asarray([case_ids[i] for i in pidx]),
                            panel_pred=pred[pidx], panel_gt=gt[pidx])
            receipt["seeds"][str(seed)]["npz_sha256"] = sha
        receipt["status"] = "PASS"
        if a.data_root:
            try:
                imgs = panel_images(a.task, case_ids, pidx, gt, a.data_root)
                receipt["panel_images_sha256"] = write_npz(out_dir / "panel_images.npz", panel_idx=np.asarray(pidx),
                                                           panel_case_ids=np.asarray([case_ids[i] for i in pidx]),
                                                           image=imgs)
            except Exception as exc:  # images are optional; gate/masks unaffected
                receipt["panel_images_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    receipt["elapsed_s"] = round(time.time() - t0, 1)
    receipt["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (out_dir / "cell_receipt.json").write_text(json.dumps(receipt, indent=1))
    print(f"{a.task}__{a.fm} {receipt['status']} " + (receipt.get("error") or
          "max diff %.3g" % max(v["max_abs_trace_diff"] for v in receipt["seeds"].values())))
    return 0 if receipt["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
