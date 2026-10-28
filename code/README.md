# Study implementation

This directory holds the code used for the paper's preprocessing, feature extraction, decoder training, fine-tuning, nnU-Net runs, inference and statistics. It needs the upstream datasets and model weights, which are not redistributed. To recompute the paper's statistics from the supplied scores, use the top-level commands in [REPRODUCING.md](../REPRODUCING.md); they do not need this directory.

| Location | Contents |
|---|---|
| `src/fmpool/` | Dataset loaders, case IDs, encoder wrappers and multi-scale taps, decoders (including the UNet-skip readout), input manifests and provenance, the seeded nnU-Net trainer shim, determinism and estimators |
| `data/splits/`, `configs/` | Train/test partitions, including the official and content-group Kvasir splits; data paths; encoder sources (`configs/encoder_sources.json`) |
| `scripts/` | Manifests, feature extraction, training, nnU-Net, pixel-overlap analysis and trace-level statistics |
| `stats/` | Failure metrics, effective size, pool selection, controls and AMOS scorers |
| `medenc_overlay/` | Encoder wrappers for RETFound, Phikon, RAD-DINO and BiomedParse |
| `table_generation/`, `scripts/prepare_workspace.py` | Paper table rendering from the original result files, which are not included (see [Table generators](#table-generators)) |

Run commands from the repository root with Python 3.10. Obtain raw datasets and model assets separately under their upstream terms.

## Environment

| Purpose | Install | Notes |
|---|---|---|
| Released analyses (top level) | `python3 -m pip install -r requirements.txt` | NumPy 1.26.4 and SciPy 1.15.3; see the [quick start](../README.md#quick-start) |
| Training, feature extraction and inference | `python3 -m pip install -r code/requirements-training.txt`, then `python3 -m pip install -r code/requirements-pinned.txt` | The pinned file lists the direct numerical and model dependencies of the Python 3.10 / CUDA 12.4 training environment; it is not a full lockfile |
| nnU-Net | `python3 -m pip install -r code/requirements-3d-nnunet.txt` | Training requirements plus MONAI and nnunetv2==2.6.2 |
| AMOS evaluation | `python3 -m pip install -r code/requirements-amos-evaluation.txt` in a separate environment | Pins differ from the training environment (for example torch 2.5.1 and MONAI 1.5.0) |
| CPU tests of this directory | `python3 -m pip install -r code/requirements-analysis.txt -r code/requirements-training.txt pytest SimpleITK==2.5.6` | Run `python3 -m pytest -q code/stats code/medenc_overlay` after setting `PYTHONPATH` |

`requirements-repro.txt` (numerical core), `requirements-stats.txt` (adds pandas) and `requirements-analysis.txt` (adds NiBabel) are include layers for the files above, not separate environments. Tests that need torch are skipped when it is not installed. The released analyses have their own tests: `python3 -m unittest discover -s tests`.

Set `export PYTHONPATH=code/src:code/stats:code/scripts` in the same shell. Replace uppercase/path placeholders below before running; abbreviated Python script names mean `python3 code/scripts/NAME.py` from the repository root. The output folders used in the commands below (`runs/`, `cache/`, `inputs/`, `nnunet/`, `receipts/`, `overlap/`, `results/`) and `build/`, the default output folder of `stats/analysis`, are ignored by git.

## Data preparation

Obtain datasets from their distributors; set `FMPOOL_DATA_ROOT` (or `code/configs/paths.yaml`). Sub-directories: `Kvasir-SEG`, `ACDC`, `brats2024_gli`, `RIGA/RIGAPlus`, `ISIC2018_official`, `Task04_Hippocampus`, `Task09_Spleen`, `Task02_Heart`.

The loaders default to the paper's primary splits (official Kvasir, ACDC and ISIC-2018 splits). Set `FMPOOL_KVASIR_SPLIT=grouped_v2` for the content-group Kvasir sensitivity split.

- **Kvasir-SEG** (datasets.simula.no/kvasir-seg; `Kvasir-SEG/{images,masks}/<id>.jpg`): the primary split is the official 880/120 split of Jha et al. 2021 (default); `FMPOOL_KVASIR_SPLIT=grouped_v2` selects the content-group-disjoint sensitivity split. `kvasir_official_split.json` records the SHA-256 of every image and mask. `make_kvasir_official_split.py train.txt val.txt code/data/splits/kvasir_grouped_split.json OUT_JSON RETRIEVAL_UTC` rebuilds the committed file.
- **ACDC** (creatis.insa-lyon.fr/Challenge/acdc): flatten the official `training/` and `testing/` folders into `ACDC/Images/patientNNN_frameYY.nii.gz` and `ACDC/Masks/patientNNN_frameYY_gt.nii.gz` for all 150 patients. The target is the LV label (3), and every slice with LV foreground is a case. The default split, `acdc_official_split.json`, uses the official train patients 001–100 and test patients 101–150. `make_acdc_official_split.py ACDC_ROOT OUT_JSON RETRIEVAL_UTC` rebuilds the slice split; it validates the ED/ES frames against the committed `acdc_lv_nnunet_official_split.json`, which nnU-Net step 5 derives from the slice split.
- **BraTS 2024 GLI** (synapse.org/brats2024; `brats2024_gli/training/training_data1_v2/<subject>/<subject>-{t1c,t1n,t2f,t2w,seg}.nii.gz` and `brats2024_gli/additional/training_data_additional/<subject>/...`): 2D FLAIR, target `seg>0`. `brats_patient_split.json` is a patient-disjoint 80/20 split: all sessions of a five-digit patient stay together, `random.Random(42)` shuffles the sorted patients and 20% (rounded) form the test set. Each session contributes up to ten native-third-axis slices with the most foreground (at least 64 pixels each).
- **RIGA+** (zenodo.org/records/6325549; `RIGA/RIGAPlus/{BinRushed,Magrabia,MESSIDOR_Base1,MESSIDOR_Base2,MESSIDOR_Base3}_{train,test}.csv`, `RIGA/RIGAPlus/RIGA/<source>/<subdir>/imageN.tif` and `RIGA/RIGAPlus/RIGA-mask/<source>/<subdir>/imageN-{1..6}.tif`): train = BinRushed + MESSIDOR Base1–3, test = Magrabia. Each of the six rater masks has values {0, 128, 255}; a cup pixel has rater value 128 and a disc pixel has rater value ≥ 128. A pixel is foreground when at least 4 of the 6 raters mark it.
- **ISIC-2018 Task 1**: official 2,594/1,000 training/test split, `FMPOOL_ISIC_SPLIT=official`, `isic2018_official_split.json`; `ISIC2018_official/` holds `ISIC2018_Task1-2_{Training,Test}_Input` and `ISIC2018_Task1_{Training,Test}_GroundTruth`. Rebuild: `make_isic_official_split.py ISIC_ROOT OUT_JSON RETRIEVAL_UTC`.
- **MSD Hippocampus/Spleen/Heart**: committed volume-level splits, `imagesTr/` and `labelsTr/`.

The ID lists and folds in the committed split files are identical to those used for the runs. For release, nine of the eleven files were edited outside their ID lists (`kvasir_official_split.json` and `brats_patient_split.json` are unchanged):

- free-text `provenance` notes were corrected (the MSD splits are volume-level; internal protocol references were removed);
- internal download records were removed from the provenance blocks of `acdc_official_split.json` (`girder_receipt`, `girder_files`, `compare`, `share_copy_verdict`, `provenance_edit`) and `isic2018_official_split.json` (`malformed_receipt_lines`, `training_archives_from`, `raw_manifest`, and the per-archive `expected_bytes_head` and `curl_rc`);
- `acdc_lv_nnunet_official_split.json` now records the SHA-256 of the released `acdc_official_split.json` in `provenance.source_split`.

These files' SHA-256 values therefore differ from the ones in the run records. Two facts from the removed records: the study's copy of the ACDC NIfTI files was byte-identical to the 600 official Girder NIfTI files, and the two ISIC-2018 training archives were checked against SHA-256 values beginning `80f98572` and `99f8b2bb`. In `kvasir_grouped_split.json`, `near_duplicate_groups` lists one pair twice: the declared near-duplicate pair is also found by the content rule, and the loader regenerates this list and checks it against the file.

ISIC is an image-level benchmark here. Partial official patient IDs reveal train/test overlap; calibration/evaluation splits can also place images of a known patient in both halves. The paper's Appendix I.1 details the evaluation units.

For ACDC and BraTS, the loaders read native NIfTI arrays with NiBabel and slice along the third axis without affine reorientation, cropping, padding or intensity windowing. Each float32 slice is min–max scaled over all pixels, including background, to [0, 255] and truncated to uint8; constant slices become zero. After replication into three channels, bilinear resizing to 224×224 uses `align_corners=False` without antialiasing, followed by clamping and truncation to uint8. Encoder conversion divides by 255 before bundle-specific normalisation. Kvasir-SEG and ISIC-2018 images are returned at native resolution and resized in float32 inside `extract_features.py` (bilinear, `align_corners=False`, no antialiasing) without an intermediate uint8 step; RIGA, ACDC, BraTS and MSD resize in their loaders and store uint8.

**Manifests.** Raw-data trainers and feature extraction take `--input-manifest`; cached-head trainers take `--cache-manifest-sha256`. The frozen-cell wrapper supplies both. Create a manifest pinning raw files, splits and encoder weights with `make_input_manifest.py --task riga_cup --fm dinov2_vitb14 --repository facebookresearch/dinov2 --revision 7764ea0f912e53c92e82eb78a2a1631e92725fc8 --out inputs/riga_cup/dinov2_vitb14.json`. Manifests and feature caches bind the dataset and encoder source hashes. Keep them with the code checkout that produced them; a different checkout may require a new manifest and cache even when the preprocessing numerics are unchanged.

**Encoder sources.** `configs/encoder_sources.json` lists, for the 11 primary bundles, the weight source and revision recorded in the runs and the SHA-256 of the loaded FP32 encoder state. DINOv2 bundles load torch.hub `facebookresearch/dinov2` at a pinned commit: `export FMPOOL_DINOV2_REVISION=7764ea0f912e53c92e82eb78a2a1631e92725fc8`. Pass the listed repository and revision to `make_input_manifest.py`; the manifest's `encoder_source.state_sha256` equals `fp32_state_sha256` when the same weights load. The open_clip, timm and Hugging Face loaders fetch the current hub files, so a mismatch reveals changed upstream weights. The two CLIP entries name the Hugging Face repositories (`timm/vit_*_clip_224.openai`) from which open_clip 3.3.0 downloads the OpenAI weights for the `quickgelu` configurations of Table B.1. The six added checkpoints and the four medical encoders are not listed. DINOv2-L uses the same pinned hub revision; the other added checkpoints load the identifiers of the paper's Table B.1 from the current hub files; the medical encoders read weight files placed under `FMPOOL_MEDENC_ASSET_ROOT` (step 2), without a recorded revision. RAD-DINO builds its backbone through the DINOv2 hub code, so step 2 also needs `FMPOOL_DINOV2_REVISION` exported.

## Training and inference

Fixed recipe: FP32 (no TF32/AMP), 224×224, Adam lr 1e-3, 30 epochs, batch 16, seeds 42–45. `train_head_cached.py` and `train_frozen_fm.py` refuse other settings without `--non-primary`.

1. **Primary analysis (frozen encoder + 1×1 conv head).**
   - Eleven encoder bundles (`biomedclip clip_vitb16 clip_vitl14 convnext_tiny deit_vitb16 dinov2_vitb14 dinov2_vits14 efficientnet_b0 mae_vitb16 resnet18 resnet50`) × tasks `kvasir acdc_lv brats_wt riga_cup riga_disc`.
   - One cell extracts features once (`extract_features.py`) and trains four head seeds (`train_head_cached.py`; `train_frozen_fm.py` is the equivalent online trainer), writing `runs/<cell>/per_case_dice/<task>/<fm>/seed_<S>.json`: `FMPOOL_CACHE_ROOT=cache bash code/scripts/run_frozen_cell.sh riga_cup dinov2_vitb14 inputs/riga_cup/dinov2_vitb14.json runs/riga_cup__dinov2_vitb14`.
   - The same command runs the distinct same-family checkpoints (`resnet34 resnet101 convnext_small convnext_base dinov2_vitl14 clip_vitb32`) and the held-out tasks (`isic2018 msd_hippocampus msd_spleen msd_heart`).
   - `run_frozen_cell.sh` uses `--device cuda`. For CPU, run its two commands with `--device cpu`: `extract_features.py --task T --fm F --input-manifest M --cache-root cache --device cpu`, then, for each seed, `train_head_cached.py --task T --fm F --seed S --device cpu --cache-root cache --cache-manifest-sha256 SHA --out RUN/per_case_dice/T/F --checkpoint-dir RUN/checkpoints/T/F --epochs 30 --batch-size 16`, where `SHA` is the SHA-256 of `cache/T/F/manifest.json`.
   - A CPU replay of the Kvasir ResNet-18 and DeiT-B cells reproduced the released traces: 949 of 960 per-case Dice values were identical, and the largest difference was 3.5e-4.
   - For the mask-alignment sensitivity, rerun these cells on `riga_cup riga_disc acdc_lv kvasir` with `FMPOOL_MASK_RESIZE=nearest-exact` and a new cache root. The switch applies to feature extraction and the 1×1 trainers; the fine-tuning and UNet-skip trainers resize native-resolution masks with `nearest`. `compute_paired_gap_difference.py --root-a EXACT --root-b PRIMARY --tasks riga_cup riga_disc acdc_lv kvasir --out OUT.json` then gives the Table B.2 gaps and their paired difference.
2. **Medical encoders** (`retfound_vitl16 phikon rad_dino biomedparse` on `riga_cup riga_disc kvasir`): copy the package, replace `code/src/fmpool/encoders.py` by `code/medenc_overlay/encoders.py`, set `FMPOOL_MEDENC_ASSET_ROOT` (weights under `<root>/models/<fm>/`, BiomedParse-v1 source checkout under `<root>/upstream/BiomedParse-v1`; detectron2 is not needed), then run step 1. The overlay is a snapshot of `src/fmpool/encoders.py` with the medical encoders added, taken for these runs; it is not kept in sync with that file, so a change to one is not applied to the other.
3. **UNet-skip readout**: create a separate manifest with `make_input_manifest.py --task T --fm F --repository REPO --revision REV --multiscale --out inputs/T/F_multiscale.json`, using the actual pinned repository/revision. Then run `train_head_unet_skip.py --task T --fm F --seed S --input-manifest inputs/T/F_multiscale.json --out runs/unet/per_case_dice --checkpoint-dir runs/unet/checkpoints/T/F --epochs 30 --batch-size 16`. The default manifest uses the single-scale wrapper for primary/FT trainers; UNet requires multiscale tensor names.
4. **End-to-end fine-tuning and random initialisation**, one folder per encoder-lr arm:
   - `train_adapted.py --regime full_ft --encoder-lr 1e-4 --task riga_cup --fm F --seed S --input-manifest M --epochs 30 --batch-size 16 --out runs/ft/lr1e-4`, and the same with `1e-5`. Checkpoints go under `OUT/checkpoints/` unless `--checkpoint-dir` is given; the other trainers require `--checkpoint-dir`.
   - The paper's official-split Kvasir rows use the default split with `--task kvasir --out runs/ft/kvasir_official/lr<ARM>`. `runs/ft/lr<ARM>` also holds the content-group Kvasir sensitivity rows: run `--task kvasir --out runs/ft/lr<ARM>` with `FMPOOL_KVASIR_SPLIT=grouped_v2` set.
   - Then run `compute_ft_table.py --root runs/ft --frozen-root <root> [--frozen-root-official <root>] --out runs/ft/table`, where `--frozen-root` holds frozen traces for RIGA Cup and the content-group Kvasir split and `--frozen-root-official` those for the official split.
   - The random-initialisation control uses `--regime rand_init` (no `--encoder-lr`) for `convnext_tiny` and `resnet50` on `acdc_lv` and `riga_cup`, seeds 42–49.
5. **nnU-Net v2, case-identical** (RIGA Cup 2D as dataset 501, ACDC 3d_fullres as 502 with `CASE_TASK=acdc_lv ACDC_CASE_DATASET_ID=502`; 5 folds × seeds 13 and 37; reported `TRAINER_EPOCHS=100`): ACDC export takes `--split-json code/data/splits/acdc_lv_nnunet_official_split.json` (without it, the same folds are derived from the committed slice split); use this file as the canonical split for its manifest. `make_acdc_nnunet_official_split.py` rebuilds it from the slice split and a frame listing (`--help`). Adapt the steps below to `--task acdc_lv` and ID 502. RIGA Cup is exported from the same 224×224 resized images and consensus masks as the primary protocol.

   ```bash
   export nnUNet_raw=$PWD/nnunet/raw nnUNet_preprocessed=$PWD/nnunet/preprocessed nnUNet_results=$PWD/nnunet/results
   python3 code/scripts/install_nnunet_seeded_trainers.py            # once: seeded trainer shim
   python3 code/scripts/prepare_nnunet_data.py --task riga_cup --dataset-id 501          # export
   python3 code/scripts/prepare_nnunet_data.py --task riga_cup --dataset-id 501 --preprocess \
     --preprocessing-evidence receipts/riga_cup.json
   python3 code/scripts/make_nnunet_input_manifest.py --task riga_cup \
     --raw-export $nnUNet_raw/Dataset501_RIGACup --receipt receipts/riga_cup.json \
     --canonical-split code/data/splits/riga_cup_split.json --input-manifest inputs/nnunet_riga_cup.json
   CASE_TASK=riga_cup RIGA_CASE_DATASET_ID=501 RIGA_CUP_INPUT_MANIFEST=inputs/nnunet_riga_cup.json \
     RIGA_CUP_PREPROCESSING_EVIDENCE=receipts/riga_cup.json WORKER_ID=0 N_WORKERS=1 TRAINER_EPOCHS=100 \
     bash code/scripts/run_nnunet_case_worker.sh worker        # writes results/nnunet/nnunet_case_riga_2d_100ep/
   python3 code/scripts/compute_nnunet_case_identical.py --input-dir results/nnunet/nnunet_case_riga_2d_100ep \
     --trainer-epochs 100 --out results/nnunet/riga_case_identical_2d_100ep.json
   ```

   For the paper's unsmoothed primary Dice, rescore collected holdout predictions before summarising them. Keep each original trace as `TRACES/TASK_foldF_seedS.json` with its run identity, and its predictions in `CELLS/TASK_foldF_seedS.cell/holdout_predictions/`. `LABELS` is the matching exported `labelsTs/`, and `PRIMARY_TRACES` is a new directory.

   ```bash
   python3 code/scripts/rescore_nnunet_primary_dice.py --traces-dir TRACES --cells-dir CELLS \
     --labels-dir LABELS --out-dir PRIMARY_TRACES
   python3 code/scripts/compute_nnunet_case_identical.py --input-dir PRIMARY_TRACES --trainer-epochs 100 \
     --out PRIMARY_SUMMARY.json
   ```

   NIfTI geometry follows nnU-Net's `SimpleITKIO` reader: size, spacing, origin and direction must agree. New exports preserve voxel arrays and write each image and mask with its own validated geometry. Create a fresh preprocessing receipt and input manifest for these new files. Holdout scoring requires an explicit nonempty holdout; it cannot fall back to fold validation.

6. **Pixel-level error overlap**: per cell, `evaluate_error_masks.py --source . --cell runs/<cell> --cache-root cache --task T --fm F --out overlap/raw/local/out`; then `analyze_pixel_overlap.py --raw overlap/raw --traces <root>/per_case_dice --primary-wc <root>/within_cross --out overlap --tasks riga_cup riga_disc kvasir acdc_lv --suffix _run`. The full task set without `--suffix` also draws example panels, which require BraTS; a restricted task subset writes tables only.
7. **Per-rater RIGA rescoring**: `python3 -m analysis.rater_sensitivity inference --task riga_cup --repo-root RUN --cache-root cache --rater-mask-root $FMPOOL_DATA_ROOT/RIGA/RIGAPlus/RIGA-mask --output raw_cup.json`, then `python3 -m analysis.rater_sensitivity analyze --task riga_cup --raw-input raw_cup.json --output-json rater_cup.json`; repeat for `riga_disc` with `raw_disc.json` and `rater_disc.json`. Output files are never overwritten, so each run needs new names.
8. **Native-3D AMOS comparison**: use the AMOS environment and a clone of [MRSegmentator](https://github.com/hhaentze/MRSegmentator). Set `COFAILMED_AMOS_EVALUATOR` to its `evaluation/` directory; `eval_data.py` and `mappings.py` must match the pinned hashes (unchanged since commit `67e6713`).
   - Score each system with `python3 code/stats/amos/evaluate_masks.py --labels LABELS --segmentation SEG --system {mrseg_fold0,mrseg_fold1,totalseg} --evaluator $COFAILMED_AMOS_EVALUATOR --case-table CASES.csv --out EVAL` (add `--totalseg-organs --images IMAGES` for TotalSegmentator).
   - Merge organs into a new directory with `python3 code/stats/amos/finalize_organs.py --evaluation-dir EVAL --out FINAL`.
   - Compute the statistics with `python3 code/stats/recompute_amos.py --input-csv FINAL/comparison_per_case.csv --out amos_statistics.json`. With `--reference-metrics METRICS_JSON`, it also compares every numeric field with an earlier `metrics.json`. The study's `metrics.json` is not included; its per-case and per-organ scores are released as `amos/ct` and `amos/mri` in `data/extensions.npz`, from which `extensions.py --analysis amos` recomputes Tables G.7 and G.8.

## Statistics from your own traces

Collect traces into `<root>/per_case_dice/<task>/<fm>/seed_<S>.json` (held-out: `<heldout>/per_case_dice/`; pool selection also reads the isic2018 and msd_hippocampus traces from `<root>/per_case_dice_heldout_11fm/`):

```bash
python3 code/scripts/compute_all.py --results-root <root>          # writes <root>/within_cross, m_eff, ...
python3 -m analysis.effective_size --traces-root <root> --stats-root <root>   # pooled-P33 effective sizes
python3 code/scripts/heldout_stats.py <heldout> <task>             # held-out tasks -> <heldout>/stats
python3 -m analysis.pool_selection --results-root <root> --split-count 200 --output ps.json --table ps.md
python3 -m analysis.pool_low_meff --selection ps.json --results-root <root> --output low_meff.json
```

- **Functional floor**: a bundle enters a task's pool only if its mean Dice over seeds is ≥ 0.30.
- **Groups**: a pair of (bundle, seed) traces is a *same/release* pair if both use the same frozen checkpoint (seed pairs of one bundle) or are DINOv2-B × DINOv2-S (one release and recipe); all other pairs form the *cross-bundle* reference.
- **Estimate**: r̄ = mean Pearson correlation of per-case Dice over the pairs of a group; gap = r̄_same − r̄_cross (`fmpool.estimators.within_cross_rbar`).
- **Uncertainty for primary Pearson gaps**: percentile 95% intervals with B = 2000; random seeds and resampling settings are recorded in each analysis output. Kvasir and RIGA use case bootstrap; ACDC and BraTS use patient-cluster bootstrap (all slices of a resampled patient). Hierarchical resampling of bundle/release groups and their traces (DINOv2-B/S merged) is a descriptive sensitivity. The alternative κ, Q, disagreement and double-fault intervals use case resampling on every task, with the observed threshold and retained bundles fixed. The φ and joint-fail ratio intervals re-estimate the percentile on each draw.
- **Effective pool size**: M_eff = M / (1 + (M − 1) ρ_fail), a variance-equivalent summary for standardised failure indicators, where ρ_fail is the mean pairwise correlation of the failure indicators 1[Dice < threshold]. Every member needs ≥ 5 failures and ≥ 5 passes. Nonpositive or numerically singular denominators are undefined. `compute_all.py` uses threshold 0.85; `analysis.effective_size` uses pooled-Dice percentiles (P33 primary). Its P33 intervals fix the observed threshold and retained bundles, while rechecking failure/pass support and numerical eligibility on each draw.
- **Robustness and diagnostics**: `compute_audit_sensitivity.py` (floor sweep, ICC(A,1), leave-one-bundle-out, permutation), `compute_{failure_event_summary,item_difficulty_robustness,tau_sweep,vit_arch_diagnostic}.py`, `compute_{cross_checkpoint_family,medenc_pool,heldout_11fm_summary}.py`, `analysis.failure_metrics` and `analysis.rater_sensitivity`.

### Which copy of a statistic to use

Some statistics exist in more than one file. Edit and run the study implementation; the scorer copies are snapshots that the controls and AMOS scorers import from their own directories, and they are hash-pinned (next section).

| Statistic | Released analysis (score packs) | Study implementation (trace folders) | Scorer copies (do not edit) |
|---|---|---|---|
| P33 effective size | `effective_size.py` | `stats/analysis/effective_size.py` | `stats/controls/analysis/effective_size.py`, `stats/amos/analysis/effective_size.py` |
| Fixed-threshold effective size | `effective_size.py --analysis fixed` | `src/fmpool/estimators.py` (`compute_m_eff`, called by `compute_all.py`) | — |
| κ, Q, disagreement and double fault | `diagnostics.py` | `stats/analysis/failure_metrics.py` | `stats/{controls,amos}/analysis/failure_metrics.py` and `failure_comparison.py` |
| Trace loading and bundle floor | `scores.py` | `stats/analysis/trace_validation.py` | `stats/controls/analysis/trace_validation.py` |

### Provenance replays and hash-pinned files

`stats/recompute_controls.py`, `scripts/prepare_workspace.py` and `scripts/compute_checkpoint_statistics.py --trace-tree` replay the study's own calculations and need its result folders, which are not included. `recompute_controls.py --root ROOT --out OUT` runs the controls scorer with `ROOT/results/primary` (per-case traces, `stats/m_eff/` and `postproc/e1_meff_configs/out/metrics.json`), `ROOT/results/finetuning` (traces and `FT_TABLE.json`), `ROOT/results/controls/metric_points.json` and, if present, `ROOT/SOURCE_MAP.json` written by `prepare_workspace.py`. Do not run `stats/controls/compute_controls.py` directly; it refuses to start.

`SCORER_SOURCE_MAP.json` records SHA-256 hashes for 29 files (the AMOS and controls scorers and the table generators) and for the study file each was derived from (`original/...` names refer to the study workspace). Most were adapted for paths and I/O. `retained_definitions` lists the functions and constants whose code is unchanged from the study file (comments excepted). `prepare_workspace.py` re-checks the nine table-generator hashes. Other files in this directory, including `stats/analysis/`, are not listed in the map.

### Table generators

`table_generation/` writes a macro file (`corrected_numbers.tex`), `verification_macros.tex`, `panel_amos.tex` and `table_*.tex` bodies from the study's result files. The display panels (`consolidated_numbers.PANELS`) are written as `panel_*.tex`. The camera-ready `generated/` folder renames the macro file to `numbers.tex` and keeps the generator names of the 26 display panels, `panel_amos.tex`, `verification_macros.tex`, `table_adaptation_meff.tex`, `table_mask_align.tex`, `table_pixel_empty.tex` and `table_primary_quality.tex`, but most panel bodies were re-laid-out by hand. Regenerated files therefore do not reproduce the camera-ready files byte for byte:

- After generation, the camera-ready table bodies were re-laid-out by hand: duplicated columns were removed, some rows moved between tables (for example, the medical-encoder rows to Table H.1(b) and the pair-quality fit to Table D.1(c)), and spacing and labels changed. No displayed value changed.
- The camera-ready macro file adds three display macros that these generators do not emit: `\MeffFourPoolKvasirPThirtyThreeTwoDp` and `\MeffFourPoolDiscPThirtyThreeTwoDp` (the two-decimal range 1.56–2.92) and `\KvasirOfficialNearDupGroupsWord`.
- Macro and file names, minus signs, digit grouping and range dashes were also finalised in the manuscript sources.
- The held-out MSD split wording in `heldout_numbers.py` matches the camera-ready text.

Comments in `table_generation/` use the manuscript's table numbering at generation time.

## Reproduction scope

| Quantity | Inputs required |
|---|---|
| Case-Dice correlations, failure metrics, effective pool size, selection | Aligned per-case Dice traces and split IDs |
| Per-rater statistics | Per-rater Dice traces |
| Pixel-error overlap statistics | Pairwise error-mask Dice, released as `data/pixel_overlap.npz` (`extensions.py --analysis pixel_overlap`) |
| Qualitative examples | Ground truth and predictions, or raw data plus trained models for re-inference |
| Compute totals | Recorded run intervals and training times |

Scalar Dice traces do not reconstruct pixel masks or run timing. Raw data and model assets must be obtained under upstream terms. Frozen-head replication was checked on sampled cells under a matched runtime. nnU-Net augmentation is stochastic; its seeds identify replicate runs, so fresh training is not promised to be bit-identical. Saved-trace scoring is deterministic under the recorded software environment. Software and checkpoint identities are recorded with runs.

Pool selection groups upstream encoder families (several may be ViTs). Calibration means weight patients equally for ACDC and BraTS. A bundle passes the 0.30 floor on the mean of its four heads; all heads of a passing bundle remain candidates. The rule selects each family's best individual head, then the four highest-ranked family representatives, with lexicographic tie-breaking. Its P33 is taken from the best functional head's calibration case scores (slice scores for ACDC/BraTS), not patient-averaged scores, and is fixed on evaluation. Primary failure-event analysis instead uses pooled model–case P33.

## Internal identifiers

Some file names, output keys and table macros keep the study's internal labels, because the table generators and recorded outputs read them by name. Other output names were made descriptive for release and differ from the study's recorded outputs: `fig1a_inputs/` and the `fig1a` key of `PIXEL_OVERLAP_TABLE.json` are now `panel_inputs/` and `panels`; `E1_MEFF_TABLE.md` is `effective_size_table.md` (without its evidence column); `gate0_reproduction.json` is `trace_validation.json`; `omitted_locked_tasks` is `omitted_tasks`; the per-rater `remote_*` source keys are `raw_*`; `recompute_amos.py` records `scorer_source_sha256` and `helper_source_sha256` (keys `effective_size` and `failure_comparison`) instead of `original_aggregate_source_sha256` and `original_phase2_source_sha256`; and the nnU-Net summary drops the duplicate `per_model_mean_dice` (use `mean_dice_per_model`).

| Label | Meaning |
|---|---|
| `fm`, `fmpool`, `FMPOOL_*` | Encoder bundle; the study package and its environment variables |
| `brats_wt` | The paper's BraTS FG task: all annotated foreground (`seg > 0`), including resection cavities, which differs from the BraTS whole-tumour region |
| `grouped_v2` | The content-group Kvasir sensitivity split |
| `architecture_diversity` | The encoder-family pool-selection rule |
| `E1`, `e1_meff_configs` | Effective-size scorer and its outputs |
| `E4`, `e4_nonpearson_configs` | Failure-event metric comparison (κ, Q, disagreement, double fault) across settings |
| `L1A` | κ, Q, disagreement and double-fault metrics |
| `l1b_gate` | Distinct-checkpoint check in the checkpoint statistics |
| `L1D` | Per-rater RIGA rescoring outputs |
| `L2PS` | Pool-selection low-effective-size output |
| `gate0` | Study name of the trace-validation module |
| `phase2_secondary` | AMOS secondary statistics |
| `corrected` | The final trace set used in the paper |
| `MEASURED`, `DERIVED`, `CITED` | Output provenance: estimated from traces, computed from estimated values, or read from an earlier output |
| `results/_merged` | The study's merged results folder; pass its equivalent as `--results-root` |
| `v3_legacy_nearest`, `nearest_exact` | Keys of the mask-resize summary read by `mask_align_numbers.py`: the primary nearest mask convention and the nearest-exact sensitivity |
| `primary_locked_880_120_no_selection` | `split_scheme` of the content-group Kvasir split (`kvasir_grouped_split.json`); "locked" means fixed before the analysis. Run records and feature-cache manifests carry this value |
| `gray64_corr>=0.995_and_mask64_dice>=0.95_plus_audit_pair` | `group_rule` of that split: the content rule plus the declared near-duplicate pair |
| "locked Pearson bootstrap" | Error message in `stats/amos/aggregate.py`: the fixed 2,000-draw Pearson bootstrap of the AMOS scorer |

## Licences

Datasets and model weights retain their upstream terms; see [data/README.md](../data/README.md#sources-and-terms) and the paper's Appendix J. Phikon outputs are not included and remain subject to Owkin's licence. No Owkin file is redistributed; the Phikon unit test builds an untrained ViT-B/16 from hyperparameters written in the test.
