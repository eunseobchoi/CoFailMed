"""Versioned data, cache and run contracts for reproducible training."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import re
import tempfile
from importlib.metadata import distributions, version as distribution_version
from pathlib import Path

import torch

from fmpool.determinism import sha256_file

CACHE_SCHEMA = "feat_cache_v2"
PRIMARY_RECIPE = {
    "version": "primary_fp32_v2", "precision": "fp32", "tf32": False,
    "input_hw": [224, 224], "optimizer": "Adam", "lr": 1e-3,
    "epochs": 30, "effective_batch_size": 16, "seeds": [42, 43, 44, 45],
    "shuffle": "dedicated_cpu_generator_randperm_v2",
}
IDENTITY_KEYS = (
    "schema_version", "task", "fm", "precision", "split_manifest_sha256",
    "ground_truth_identity", "preprocess_identity", "code_sha256",
    "encoder_checkpoint_sha256", "input_manifest_sha256",
    "runtime_sha256", "encoder_source_sha256",
)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def runtime_identity() -> dict:
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "packages": {name: distribution_version(name) for name in sorted(
                {d.metadata["Name"] for d in distributions() if d.metadata["Name"]})}}


def source_snapshot() -> dict:
    root = Path(__file__).resolve().parents[2]
    paths = list((root / "src" / "fmpool").rglob("*.py"))
    paths += list((root / "scripts").glob("train_*.py"))
    paths += [root / "scripts" / name for name in
              ("extract_features.py", "prepare_nnunet_data.py")]
    return {str(p.relative_to(root)): p.read_text() for p in sorted(paths)}


def state_sha256(model) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted((model if isinstance(model, dict) else model.state_dict()).items()):
        value = tensor.detach().cpu().contiguous()
        h.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def encoder_source(model, declared: dict) -> dict:
    sources = {}
    for module in model.modules():
        path = inspect.getsourcefile(type(module))
        if path and Path(path).is_file():
            sources[path] = Path(path).read_text()
    return {"declared": declared, "loaded_module_sources": sources}


def ground_truth_identity(task: str) -> str:
    if task.startswith("riga_"):
        from fmpool.datasets.riga import GROUND_TRUTH_IDENTITY
        return GROUND_TRUTH_IDENTITY + ":" + task
    return task + "_gt_v1"


def validate_source_split(task: str, split: dict, *, split_contract="primary_2d") -> None:
    if split_contract not in {"primary_2d", "nnunet_export"}:
        raise ValueError("unknown split contract")
    roles = [split[role] for role in ("train", "val", "test")]
    all_ids = [cid for ids in roles for cid in ids]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("input split contains duplicate/overlapping case IDs")
    if task.startswith("brats_"):
        from fmpool.case_ids import validate_brats_split
        validate_brats_split(split)
    if task == "isic2018":
        from fmpool.datasets.isic2018 import OFFICIAL_SCHEME, validate_isic_official_split
        if split.get("split_scheme") == OFFICIAL_SCHEME:
            validate_isic_official_split(split)
    if task.startswith("riga_"):
        if any(Path(cid).is_absolute() or ".." in Path(cid).parts or str(Path(cid)) != cid for cid in all_ids):
            raise ValueError("noncanonical RIGA case path")
        if split.get("_image_paths") != {cid: "RIGA/" + cid for cid in all_ids}:
            if "_image_paths" in split or split_contract != "nnunet_export":
                raise ValueError("RIGA image mapping differs from canonical case IDs")
    if task == "acdc_lv":
        unit = split.get("case_unit", "slice")
        if unit == "slice":
            from fmpool.datasets.acdc import validate_any_acdc_split
            validate_any_acdc_split(split)  # 5-fold CV manifest, or the official 100/50 split by split_scheme
        elif unit == "frame_volume" and split_contract == "nnunet_export":
            if any(re.fullmatch(r"patient\d{3}_frame\d{2}", cid) is None for cid in all_ids):
                raise ValueError("ACDC export requires canonical frame IDs")
            folds = split.get("folds", {})
            if not folds or len(folds) % 2:
                raise ValueError("ACDC export requires paired declared folds")
            expected_keys = {f"fold_{k}_{role}" for k in range(len(folds)//2) for role in ("train", "val")}
            if set(folds) != expected_keys:
                raise ValueError("ACDC export fold keys are incomplete")
            # Two layouts: legacy CV (folds partition train+test, roles = fold zero) or a fixed holdout
            # (folds partition ``train`` only; ``test`` is a disjoint official/held-out patient set).
            fold_frames = set(folds["fold_0_train"] + folds["fold_0_val"])
            fixed_holdout = fold_frames == set(split["train"]) and fold_frames != set(all_ids)
            pool = set(split["train"]) if fixed_holdout else set(all_ids)
            for k in range(len(folds)//2):
                tr, val = folds[f"fold_{k}_train"], folds[f"fold_{k}_val"]
                if len(tr + val) != len(set(tr + val)) or set(tr + val) != pool:
                    raise ValueError("ACDC export fold does not partition declared frames")
                if {cid.split("_frame")[0] for cid in tr} & {cid.split("_frame")[0] for cid in val}:
                    raise ValueError("ACDC export fold leaks patients")
            if not fixed_holdout and (split["train"] != folds["fold_0_train"] or split["test"] != folds["fold_0_val"]):
                raise ValueError("ACDC export primary roles differ from declared fold zero")
        else:
            raise ValueError("unsupported ACDC case unit for this split contract")
    if task.startswith("brats_") or task == "acdc_lv":
        from fmpool.case_ids import brats_patient_id
        units = [{brats_patient_id(cid) if task.startswith("brats_") else cid.split("_frame")[0]
                  for cid in ids} for ids in roles]
        if any(units[i] & units[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("input split leaks patient/subject IDs")


def read_input_manifest(path: Path, task: str, *, split_contract="primary_2d") -> dict:
    path = Path(path).resolve()
    manifest_bytes = path.read_bytes()
    manifest_text = manifest_bytes.decode("utf-8")
    manifest = json.loads(manifest_text)
    if manifest.get("schema_version") != "input_manifest_v2" or manifest.get("task") != task:
        raise ValueError("input manifest schema/task mismatch")
    for key in ("raw_files", "split_manifest"):
        if not manifest.get(key):
            raise ValueError(f"input manifest requires {key}")
    for name, expected in manifest["raw_files"].items():
        raw = Path(name)
        if not raw.is_absolute() or sha256_file(raw) != expected:
            raise ValueError(f"input file hash mismatch or non-absolute path: {name}")
    split_path = Path(manifest["split_manifest"])
    if not split_path.is_absolute():
        raise ValueError("split_manifest must be an absolute path")
    if sha256_file(split_path) != manifest.get("split_manifest_sha256"):
        raise ValueError("split manifest hash mismatch")
    split = json.loads(split_path.read_text())
    validate_source_split(task, split, split_contract=split_contract)
    source = manifest.get("encoder_source", {})
    if source.get("repository") == "facebookresearch/dinov2":
        os.environ["FMPOOL_DINOV2_REVISION"] = source.get("revision", "")
    return {"manifest": manifest, "manifest_text": manifest_text,
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(), "split": split}


def verify_dataset_inputs(inputs: dict, train_ds, test_ds) -> None:
    task = inputs["manifest"]["task"]
    from fmpool.datasets import splits_path
    active_split = splits_path(task)
    if (sha256_file(active_split) != inputs["manifest"]["split_manifest_sha256"]
            or json.loads(active_split.read_text()) != inputs["split"]):
        raise ValueError("active committed split differs from pinned input split")
    validate_source_split(task, inputs["split"])
    for role, ds in (("train", train_ds), ("test", test_ds)):
        ids = list(getattr(ds, "case_ids", getattr(ds, "subject_ids", [])))
        expected = inputs["split"].get(role)
        if hasattr(ds, "subject_ids") and not hasattr(ds, "case_ids"):
            expected = sorted(session for session, assignment in inputs["split"]["_session_splits"].items()
                              if assignment == role)
        if ids != expected:
            raise ValueError(f"input manifest {role} IDs differ from loaded dataset")
        if task.startswith("brats_"):
            from fmpool.case_ids import validate_brats_split
            entries = ([(cid, session, axi) for cid, (session, axi) in ds._slice_index.items()]
                       if hasattr(ds, "_slice_index") else None)
            sessions = ds.subject_dirs if hasattr(ds, "subject_dirs") else ds._dirs
            validate_brats_split(inputs["split"], sessions=sessions, case_entries=entries)
        elif task == "acdc_lv":
            from fmpool.datasets.acdc import validate_any_acdc_split
            validate_any_acdc_split(inputs["split"], ds._case_index)
        elif task.startswith("riga_"):
            if ds._image_paths != inputs["split"]["_image_paths"]:
                raise ValueError("loaded RIGA image mapping differs from pinned split")
            for cid in ids:
                image = ds.root / ds._image_paths[cid]
                consumed = [image, *ds._rater_mask_paths(ds._image_paths[cid])]
                if any(str(path.resolve()) not in inputs["manifest"]["raw_files"] for path in consumed):
                    raise ValueError("RIGA consumed image/rater path is not declared")
    # Require complete root coverage, not just a convenient sample of masks.
    declared = set(inputs["manifest"]["raw_files"])
    for root in {Path(train_ds.root).resolve(), Path(test_ds.root).resolve()}:
        paths = list(root.rglob("*"))
        if any(p.is_symlink() for p in paths):
            raise ValueError("unsupported symlink inside dataset root; only root aliases are supported")
        actual = {str(p.resolve()) for p in paths if p.is_file()}
        if not actual <= declared:
            raise ValueError("input manifest does not cover all dataset-root files")


def cache_identity_matches(actual: dict, expected: dict) -> bool:
    return (expected.get("schema_version") == CACHE_SCHEMA
            and all(k in actual and k in expected and actual[k] == expected[k]
                    for k in IDENTITY_KEYS))


def preprocess_identity(input_hw, mean, std) -> str:
    """Digest of the cache preprocessing (mask mode from FMPOOL_MASK_RESIZE; default reproduces the legacy digest)."""
    from fmpool.datasets import mask_resize_mode
    return digest({"input_hw": input_hw, "image": "bilinear_align_corners_false", "mask": mask_resize_mode(),
                   "mean": mean, "std": std})


def load_feature_cache(cache_dir: Path, *, allow_legacy=False, expected_sha256=None):
    path = Path(cache_dir) / "manifest.json"
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError("cache manifest hash differs from requested artifact")
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != CACHE_SCHEMA:
        if not allow_legacy or manifest.get("schema_version") != "feat_cache_v1":
            raise ValueError("legacy/unknown cache rejected; use an explicit historical opt-in")
        manifest["primary_eligible"] = False
    else:
        if any(not manifest.get(k) for k in IDENTITY_KEYS):
            raise ValueError("incomplete cache identity")
        if manifest["precision"] != "fp32" or manifest.get("use_amp_bf16") is not False:
            raise ValueError("v2 cache requires FP32 computation")
        if manifest["identity_sha256"] != digest({k: manifest[k] for k in IDENTITY_KEYS}):
            raise ValueError("cache identity hash mismatch")
        if manifest["code_sha256"] != digest(source_snapshot()):
            raise ValueError("cache code snapshot differs from current source")
        if manifest["ground_truth_identity"] != ground_truth_identity(manifest["task"]):
            raise ValueError("cache ground truth mismatch")
        from fmpool.datasets import splits_path
        split_file = splits_path(manifest["task"])
        if manifest["split_manifest_sha256"] != sha256_file(split_file):
            raise ValueError("cache split differs from current committed split")
        frozen_split = json.loads(split_file.read_text())
        validate_source_split(manifest["task"], frozen_split)
        if manifest["input_hw"] != PRIMARY_RECIPE["input_hw"]:
            raise ValueError("cache input size differs from the primary recipe")
        if preprocess_identity(manifest["input_hw"], manifest["norm_mean"], manifest["norm_std"]) != manifest["preprocess_identity"]:
            raise ValueError("cache preprocessing metadata mismatch")
        for filename, expected in manifest["artifact_sha256"].items():
            if Path(filename).name != filename or sha256_file(cache_dir / filename) != expected:
                raise ValueError(f"cache artifact hash mismatch: {filename}")
        for required in ("train.pt", "test.pt", "encoder_state.pt", "provenance.json"):
            if required not in manifest["artifact_sha256"]:
                raise ValueError(f"missing artifact hash: {required}")
        evidence = json.loads((cache_dir / "provenance.json").read_text())
        for field, value in {
            "code_sha256": evidence["source_snapshot"],
            "runtime_sha256": evidence["runtime"],
            "encoder_source_sha256": evidence["encoder_source"],
        }.items():
            if digest(value) != manifest[field]:
                raise ValueError(f"cache provenance mismatch: {field}")
        pinned_text = evidence["inputs"].get("manifest_text", "")
        if (hashlib.sha256(pinned_text.encode("utf-8")).hexdigest() != manifest["input_manifest_sha256"]
                or json.loads(pinned_text) != evidence["inputs"]["manifest"]):
            raise ValueError("cache embedded input manifest bytes mismatch")
        declared_input = evidence["inputs"]["manifest"]
        if (declared_input.get("schema_version") != "input_manifest_v2"
                or declared_input.get("task") != manifest["task"]
                or evidence["encoder_source"]["declared"] != declared_input.get("encoder_source")):
            raise ValueError("cache task/encoder declaration mismatch")
        if evidence["inputs"]["sha256"] != manifest["input_manifest_sha256"]:
            raise ValueError("cache input provenance mismatch")
        if (evidence["inputs"]["split"] != frozen_split
                or evidence["inputs"]["manifest"]["split_manifest_sha256"] != manifest["split_manifest_sha256"]):
            raise ValueError("cache embedded split differs from the frozen split")
        encoder_state = torch.load(cache_dir / "encoder_state.pt", map_location="cpu", weights_only=True)
        if any(not torch.isfinite(value).all() for value in encoder_state.values()):
            raise ValueError("cache encoder state contains nonfinite tensors")
        actual_state_hash = state_sha256(encoder_state)
        if (actual_state_hash != manifest["encoder_checkpoint_sha256"]
                or evidence["inputs"]["manifest"]["encoder_source"]["state_sha256"] != actual_state_hash):
            raise ValueError("cache encoder tensors differ from the pinned state")
        del encoder_state
        if manifest["task"].startswith("riga_"):
            from fmpool.datasets.riga import require_training_palette_verification
            require_training_palette_verification("train", evidence.get("riga_training_palette"))
            if not evidence.get("riga_test_palette"):
                raise ValueError("RIGA test-palette audit missing")
    blobs = [torch.load(cache_dir / f"{role}.pt", map_location="cpu", weights_only=True)
             for role in ("train", "test")]
    for role, blob in zip(("train", "test"), blobs):
        ids, feats, masks = blob["case_ids"], blob["feats"], blob["masks"]
        if not ids or len(set(ids)) != len(ids) or len(ids) != len(feats) or len(ids) != len(masks):
            raise ValueError(f"invalid cache {role} row alignment")
        if manifest.get("schema_version") == CACHE_SCHEMA:
            if ids != frozen_split[role] or manifest[f"n_{role}"] != len(ids):
                raise ValueError(f"cache {role} IDs/count differ from the frozen split")
            if digest(ids) != manifest[f"{role}_case_ids_sha256"]:
                raise ValueError(f"cache {role} ID hash mismatch")
            if tuple(feats.shape[1:]) != (manifest["feature_dim"], *manifest["feature_hw"]):
                raise ValueError("cache feature shape mismatch")
            if tuple(masks.shape[1:]) != tuple(manifest["input_hw"]) or not torch.isin(masks, torch.tensor([0, 1], dtype=masks.dtype)).all():
                raise ValueError("cache binary mask shape/palette mismatch")
            if feats.dtype != torch.float32 or not torch.isfinite(feats).all():
                raise ValueError("v2 feature bank must contain finite FP32 tensors")
            if masks.flatten(1).sum(1).tolist() != manifest[f"{role}_foreground_pixels"]:
                raise ValueError(f"cache {role} foreground counts mismatch")
            if manifest["task"].startswith("riga_"):
                audit = evidence[f"riga_{'training' if role == 'train' else 'test'}_palette"]
                expected_masks = {str(Path("RIGA-mask") / Path(cid).parent /
                                      f"{Path(cid).stem}-{rater}.tif")
                                  for cid in ids for rater in range(1, 7)}
                if (audit.get("training_case_ids") != ids or audit.get("split") != role
                        or set(audit.get("mask_sha256", {})) != expected_masks
                        or set(audit.get("topology", {})) != expected_masks):
                    raise ValueError(f"cache RIGA {role} audit membership mismatch")
                declared_masks = {"RIGA-mask/" + path.split("/RIGA-mask/", 1)[1]: value
                                  for path, value in evidence["inputs"]["manifest"]["raw_files"].items()
                                  if "/RIGA-mask/" in path}
                if any(declared_masks.get(path) != value for path, value in audit["mask_sha256"].items()):
                    raise ValueError(f"cache RIGA {role} audit source hashes mismatch")
    if set(blobs[0]["case_ids"]) & set(blobs[1]["case_ids"]):
        raise ValueError("cache train/test overlap")
    return manifest, *blobs


def add_recipe_args(parser, *, cached=False, primary=False):
    parser.add_argument("--precision", choices=("fp32",), default="fp32")
    if primary:
        parser.add_argument("--non-primary", action="store_true",
                            help="Explicitly mark a smoke or different recipe as non-primary")
    if cached:
        parser.add_argument("--cache-manifest-sha256")
        parser.add_argument("--allow-legacy-cache", action="store_true")
    else:
        parser.add_argument("--input-manifest", type=Path, required=True)


def validate_primary_recipe(args):
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    if not args.non_primary and (args.epochs != 30 or args.batch_size != 16
                                or args.seed not in (42, 43, 44, 45)):
        raise ValueError("primary recipe requires 30 epochs, batch 16, seed 42–45")


def run_identity(args, script, *, cache=None, inputs=None, initial_state_sha256=None,
                 data_identity=None, recipe=None) -> dict:
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    runtime = runtime_identity()
    source = source_snapshot()
    record = {"schema_version": "run_identity_v2", "entrypoint": Path(script).name,
              "config": config, "recipe": recipe, "recipe_sha256": digest(recipe), "runtime": runtime,
              "source_snapshot": source, "code_sha256": digest(source),
              "runtime_sha256": digest(runtime),
              "tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32,
                       "cudnn": torch.backends.cudnn.allow_tf32},
              "cache_identity": cache, "input_identity": inputs,
              "initial_state_sha256": initial_state_sha256,
              "data_identity": data_identity,
              "primary_eligible": bool(recipe == PRIMARY_RECIPE and
                  not config.get("non_primary", False) and
                  not config.get("allow_legacy_cache", False))}
    from fmpool.datasets import mask_resize_mode
    if mask_resize_mode() != "nearest":  # sensitivity masks: recorded, never primary
        record["mask_resize"] = mask_resize_mode()
        record["primary_eligible"] = False
    record["identity_sha256"] = digest(record)
    return record


def _publish_initial_artifact(path, write):
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(tmp, path)  # Publish complete bytes without replacing a concurrent winner.
        except FileExistsError:
            pass
    finally:
        os.unlink(tmp)


def preserve_initial_state(model, out: Path, inputs: dict) -> str:
    """Content-address actual tensors; retain source, not a mutable hub alias."""
    sha = state_sha256(model)
    declared = inputs["manifest"].get("encoder_source", {})
    folder = out / "initial_states" / sha
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "state.pt"
    if not path.exists():
        _publish_initial_artifact(path, lambda stream: torch.save(model.state_dict(), stream))
    if state_sha256(torch.load(path, map_location="cpu", weights_only=True)) != sha:
        raise ValueError("existing initial-state artifact is corrupt")
    source = encoder_source(model, declared)
    source_path = folder / "source.json"
    if not source_path.exists():
        data = json.dumps(source, sort_keys=True, allow_nan=False).encode("utf-8")
        _publish_initial_artifact(source_path, lambda stream: stream.write(data))
    if json.loads(source_path.read_text(encoding="utf-8")) != source:
        raise ValueError("existing initial-state source differs from the current encoder source")
    return sha


def verify_encoder_pin(model, inputs, *, required=False):
    source = inputs["manifest"].get("encoder_source", {})
    if required and any(not source.get(k) for k in ("repository", "revision", "state_sha256")):
        raise ValueError("primary encoder requires pinned repository/revision/state_sha256")
    expected = source.get("state_sha256")
    if expected and state_sha256(model) != expected:
        raise ValueError("loaded encoder state differs from pinned input manifest")
