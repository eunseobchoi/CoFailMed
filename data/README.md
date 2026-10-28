# Analysis inputs

These files contain the aligned per-case Dice scores behind the paper's statistics: **832 traces** (457,274 scores) for the primary study and controls, and **1,028 traces** (135,036 scores) plus the AMOS22 volume scores for the appendix extensions and the held-out MSD Spleen and Heart tasks; `pixel_overlap.npz` adds the per-case error-mask overlap of head pairs. Scores are stored as float64 without rounding, averaging, filtering or reordering. Original case IDs, encoder identities and training seeds are retained. The [reproduction guide](../REPRODUCING.md) maps these inputs to paper results and links recomputed reference CSVs.

| File | Contents |
|---|---|
| `scores.npz` | Five primary panels, six added checkpoints per primary task, and ISIC/Hippocampus panels for pool selection |
| `controls.npz` | UNet-skip, two fine-tuning learning rates, random-initialisation and nnU-Net controls |
| `extensions.npz` | Mask alignment, per-rater RIGA, content-group Kvasir, medical encoders, AMOS22 and the held-out MSD Spleen and Heart tasks ([extension panels](#extension-panels)) |
| `pixel_overlap.npz` | Error-mask overlap (Dice) of every pair of floor-retained heads on each case, for Kvasir, ACDC LV, RIGA Cup, RIGA Disc and the content-group Kvasir split ([error-mask overlap](#error-mask-overlap)); no masks or pixel content |
| `metadata.json` | Array schema, one entry per panel (counts and resampling unit), protocol, family definitions and exclusions |
| `splits.json.gz` | Compressed JSON with the train/test identifier lists (and nnU-Net folds) of eight committed split files; its lists equal those in [`code/data/splits/`](../code/data/splits/). The top-level `val` lists are empty because no model selection uses a validation set; the nnU-Net folds are nnU-Net's own five training/validation folds over the training patients |

[`MANIFEST.sha256`](../MANIFEST.sha256) lists the SHA-256 of every file here and in `reference_results/`; check them with `sha256sum -c MANIFEST.sha256` from the repository root. [`croissant.json`](../croissant.json) describes the files in Croissant 1.1.

Load a panel from the repository root:

```python
from scores import load_amos, load_panel, load_pixel_overlap
panel = load_panel("primary/kvasir")
dice = panel["dice"]  # [model/readout member, case]
ct = load_amos("ct")  # AMOS22: dice [model, case] and organ_dice [model, case, organ]
overlap = load_pixel_overlap("kvasir")  # err_dice, fp_dice, fn_dice [head pair, case]
```

NPZ keys are `<panel>/<field>` and load with `allow_pickle=False`. Member fields are `bundle`, `seed`, `upstream_family` and `selection_family`; case fields are `case_ids` and `cluster_ids`. nnU-Net also retains fold labels and, where supplied, original scored-export IDs. `expanded_panel(task)` combines the 11 primary and six additional encoders after checking case alignment (`grouped=True` does the same on the content-group Kvasir split). Split lists are identifiers, not image-loader manifests. Task keys are `kvasir`, `acdc_lv`, `brats_wt`, `riga_cup` and `riga_disc`; `brats_wt` is the paper's BraTS FG task (all annotated foreground, `seg > 0`, including resection cavities).

## Extension panels

All panels in `extensions.npz` except AMOS use the member/case schema above, with seeds 42–45 and the case columns of the matching primary panel unless stated otherwise.

| Panels | Members × cases | Contents | Paper |
|---|---|---|---|
| `mask_align/<task>` for `kvasir`, `acdc_lv`, `riga_cup`, `riga_disc` | 44 × primary cases | The 11 primary bundles' 1×1 heads retrained with `nearest-exact` mask resizing; compare with `primary/<task>` (default `nearest`) | Table B.2 |
| `per_rater/rater_<k>/<task>`, k = 1–6, for `riga_cup`, `riga_disc` | 44 × 95 | The primary RIGA heads' predictions scored against rater k's mask. The consensus rows use `primary/<task>`, which scores the same predictions against the 4-of-6 consensus | Table C.2 |
| `kvasir_grouped/primary`, `kvasir_grouped/cross_checkpoint`, `kvasir_grouped/unet_skip`, `kvasir_grouped/ft/lr1e-4`, `kvasir_grouped/ft/lr1e-5` | 44 (24 for `cross_checkpoint`) × 120 | Frozen 1×1, six added checkpoints, UNet-skip and fine-tuned heads on the content-group Kvasir split (`code/data/splits/kvasir_grouped_split.json`); 12 of its 120 test images are also in the official test set | Tables C.5, G.2 and G.3 |
| `medical/<task>` for `kvasir`, `riga_cup`, `riga_disc` | 12 × primary cases | RETFound (`retfound_vitl16`), RAD-DINO (`rad_dino`) and BiomedParse v1 (`biomedparse`) frozen 1×1 heads; family labels equal the bundle IDs. Phikon heads are not included | Table H.2 |
| `amos/ct`, `amos/mri` | 3 × 60 | Per-volume Dice of MRSegmentator folds 0 and 1 and TotalSegmentator, with the 13 organ scores they average | Tables G.7 and G.8 |
| `heldout/msd_spleen`, `heldout/msd_heart` | 44 × 8 and 44 × 4 | Frozen 1×1 heads of the 11 primary bundles on the central-slice MSD Spleen and Heart test volumes (`code/data/splits/msd_{spleen,heart}_split.json`) | Table I.1(a), Appendix I.1 |

The AMOS panels have their own fields: `dice` (float64 [model, case], the mean of the case's finite organ scores), `organ_dice` (float64 [model, case, organ]; NaN where the reference organ is empty and Dice is undefined), `organs` (the 13 common organs), `case_ids` (AMOS22 file names), `model` (`mrsegmentator_fold0`, `mrsegmentator_fold1`, then `totalsegmentator_total` for CT or `totalsegmentator_total_mr` for MRI) and `system` (`mrsegmentator` for both folds, `totalsegmentator`). The three models share the same missing cells; `load_amos` checks this and that `dice` equals the mean of the finite organ scores within 1e-12. Images, masks and predictions are not included.

`heldout/msd_spleen` also has two boolean fields recorded on its 224×224 central slices: `target_empty` [case] marks reference masks without foreground and `prediction_empty` [member, case] marks predictions without foreground. They support the paper's empty-target counts and agree with the Dice convention (empty prediction and empty target score 1; exactly one empty scores 0), which `load_panel` checks.

## Error-mask overlap

`pixel_overlap.npz` holds one panel per task: `pixel_overlap/kvasir` and `pixel_overlap/acdc_lv` (official splits), `pixel_overlap/riga_cup`, `pixel_overlap/riga_disc` and `pixel_overlap/kvasir_grouped` (content-group split). Each panel lists the floor-retained heads of the matching score panel (`bundle`, `seed`), every pair of them once (`pair_a` < `pair_b`, indices into the head list) and the test cases (`case_ids`, `cluster_ids`). For each pair and case, `err_dice` is the Dice between the two heads' error masks (prediction XOR ground truth at 224×224), and `fp_dice` and `fn_dice` restrict both masks to false-positive or false-negative pixels. NaN marks a comparison in which both masks are empty: the paper's main convention scores it 1, and the conditional analysis of Table C.4 excludes it. Values are float32, the precision in which the study computed them, so they reproduce its results exactly. They were derived from the study's stored error and ground-truth masks, which are not released.

## Protocol

- **Correlations:** retain every head of each bundle whose mean Dice across heads and cases is at least 0.30. Average finite pairwise Pearson correlations; traces with standard deviation below `1e-8` have undefined correlations. Bootstrap gap intervals omit draws without finite means in both strata. Do not correlate bundle-averaged scores. Same/release pairs include the DINOv2-B/S pair. Expanded checkpoint tiers use upstream families; nnU-Net comparisons use folds.
- **Effective size:** set the threshold to P33 of pooled scores after the bundle floor, separately for each recipe. Enumerate every choice of four distinct bundles and one head per bundle. For mean pairwise failure correlation `rho`, compute `4 / (1 + 3*rho)` per eligible pool, then average those sizes. Each member needs at least five failures and five passes; the denominator must exceed `1e-12`. Outputs retain total and supported pool counts and Eq. (3) exact-covariance point means over pools with both ratios defined; this is the same supported set for all supplied P33 panels. `--bootstrap 2000` holds the observed floor and thresholds fixed, rechecks support on each draw, and reports valid draw counts and paired fine-tuning changes.
- **Fixed-threshold effective size:** `effective_size.py --analysis fixed` uses 0.85 for all five primary tasks, with separate Eq. (2)/Eq. (3) eligibility gates. It returns point estimates, support counts and the size of the intersection of eligible sets.
- **Robustness checks:** `sensitivity.py` varies the bundle floor, reports ICC and leave-out diagnostics, fits the original pair-quality regression, and evaluates pooled P25/P33/P50 failure events. These outputs are point estimates. `control_comparisons.py` adds paired FT correlation-gap intervals and the seed-13 five-fold nnU-Net effective-size calculation; its correlation summaries use all ten fold/seed members.
- **Pool selection:** use 200 half-splits with seeds 0–199. Calibration data alone determine the floor, the best head per family, the top four families and P33 of the best functional head. ACDC/BraTS calibration means weight patients equally; P33 uses that head's slice scores. Fix the threshold and selected pool before evaluating the remaining cases. Failures use strict `<`; ties pass. Unsupported splits remain in the total denominator. Clopper–Pearson intervals describe random-split frequencies conditional on each fixed cohort.
- **Extensions:** `extensions.py` applies the 0.30 floor separately to each panel. The mask-alignment difference (exact minus nearest) uses one shared set of draws for both conventions. Per-rater rows apply the floor under each reference (GT-specific pool) or keep the bundles retained under the consensus (fixed pool); all references share the image draws. The medical-encoder correlation averages each encoder's four heads against every retained primary head; Δ_med subtracts the primary cross-bundle mean. AMOS compares the two MRSegmentator folds (same system) with each fold against TotalSegmentator (cross system); its Pearson intervals use the study's Python `random.Random(20260726)` stream (CT, then MRI), and its effective size and failure metrics fix the pooled P33 and recheck support on every draw. All other extension intervals use 2,000 NumPy draws with seed 0, as in the primary analysis.

On ACDC/BraTS, the primary Pearson gaps, the φ and joint-failure intervals, the effective-size intervals, the item-difficulty, mask-alignment, checkpoint-tier and control gaps resample patients with all associated cases (slices for the primary panels, ED/ES frame volumes for nnU-Net ACDC). The five-ViT comparison, the calibration-split and pair-quality β_same intervals (Table D.1(c)), the κ/Q/disagreement/double-fault intervals (Table D.2(a)) and the error-mask overlap levels use case/slice resampling; their patient-level sensitivities are listed in REPRODUCING.md and Appendix D. Kvasir and RIGA use images; the two RIGA tasks share images. AMOS resamples volumes. ISIC uses the official image-level benchmark and has partial patient overlap. Hippocampus scores one central 2D slice per volume and uses volumes for splitting and resampling, without a subject mapping; MSD Spleen and Heart follow the same slice and volume rules. Neither ISIC nor Hippocampus establishes patient-disjoint generalisation. The seven-task selection analysis excludes MSD Spleen (degenerate empty-target behaviour) and Heart (no functional pool), as detailed in the paper; their traces are in `extensions.npz` for Table I.1(a).

## What the files cannot reproduce

- **Phikon.** Owkin's Phikon licence permits sharing only for non-commercial use by non-profit organisations and excludes transfer to for-profit organisations, which a public repository cannot ensure. Phikon heads are therefore not included. Values that pool Phikon with the other medical encoders cannot be recomputed exactly from this release: the Phikon and pooled-medical rows of Table H.2, all of Table H.1 (mutual ρ, pooled Δ_med and the 15-bundle pool), the two medical-encoder rows of Table C.5, and the pooled and extended-pool values in Section 6 and Appendix G.3. No substitute values without Phikon are provided, because the paper does not report them.
- **Mask-, prediction- and log-derived values.** The mask-centroid offset of Appendix B.3, the spatial figure panels and the GPU hours of Table B.3 need masks, images or run logs, which are not redistributed. [Reproducing the paper](../REPRODUCING.md#values-outside-this-release) lists every such value.

## Sources and terms

The score packs contain no images, masks, pixel predictions, feature caches or model weights. The project software licence does not relicense upstream assets or assign a blanket licence to derived scores. Dataset/model terms and required citations remain applicable; the paper gives checkpoint identities and full bibliographic references.

- **Kvasir-SEG:** Jha et al.; [Simula dataset and terms](https://datasets.simula.no/kvasir-seg/). Research and educational use only; commercial use requires prior written permission from Simula.
- **ACDC:** Bernard et al.; [challenge data](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html), CC BY-NC-SA 4.0, for non-commercial scientific research only (licence terms distributed with the dataset).
- **BraTS 2024 GLI:** [challenge data and terms](https://www.synapse.org/Synapse:syn53708249/wiki/627759), post-challenge CC BY-NC terms. “Data used in this publication were obtained as part of the Brain Tumor Segmentation (BraTS) Challenge project through Synapse ID: syn53708249.” Retain the consortium, source-dataset and MedPerf citations specified by the acquired version.
- **RIGA+:** [Zenodo record](https://doi.org/10.5281/zenodo.6325549), CC BY 4.0; the upstream terms also apply: RIGA ([Deep Blue Data](https://doi.org/10.7302/Z23R0R29), CC BY-NC 4.0) and MESSIDOR (research and educational use only; copying and redistribution of the images are prohibited). “Kindly provided by the Messidor program partners”; [MESSIDOR attribution and terms](https://www.adcis.net/en/third-party/messidor/).
- **ISIC-2018 Task 1:** Codella et al.; [challenge distribution](https://challenge.isic-archive.com/data/), CC0. Assets obtained through the current ISIC Archive retain their individual terms.
- **Medical Segmentation Decathlon:** Simpson et al.; [dataset](http://medicaldecathlon.com/), CC BY-SA 4.0. ShareAlike applies to the derived Hippocampus, Spleen and Heart scores where they are adaptations of the dataset: share them under the same licence, with attribution.

Encoder sources are [DINOv2](https://github.com/facebookresearch/dinov2), [OpenAI CLIP](https://github.com/openai/CLIP), [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224), [MAE](https://github.com/facebookresearch/mae), [DeiT](https://github.com/facebookresearch/deit), and torchvision/timm pretrained backbones; the segmentation control uses [nnU-Net](https://github.com/MIC-DKFZ/nnUNet). In particular, MAE retains its CC BY-NC 4.0 terms.

### Terms of the extension panels

| Panels | Scored data | Models | Terms that apply |
|---|---|---|---|
| `mask_align/*` | Kvasir-SEG, ACDC, RIGA+ test cases | The 11 primary bundles | As for the primary panels: Kvasir-SEG research/educational use; ACDC CC BY-NC-SA 4.0 non-commercial research; RIGA+ CC BY 4.0 with RIGA CC BY-NC 4.0; MAE CC BY-NC 4.0 |
| `per_rater/*` | RIGA+ Magrabia test images and the six RIGA rater masks | The 11 primary bundles | RIGA+ CC BY 4.0 and RIGA CC BY-NC 4.0 (the rater masks are RIGA annotations); MESSIDOR images are used only for training and no MESSIDOR scores are included; MAE CC BY-NC 4.0 |
| `kvasir_grouped/*` | Kvasir-SEG test images of the content-group split | The 11 primary bundles and six added checkpoints | Kvasir-SEG research/educational use, commercial use only with Simula's written permission; MAE CC BY-NC 4.0 |
| `medical/*` | Kvasir-SEG and RIGA+ test cases | [RETFound](https://huggingface.co/YukunZhou/RETFound_mae_natureCFP) weights, CC BY-NC 4.0; [RAD-DINO](https://huggingface.co/microsoft/rad-dino), MIT, shared for research purposes only; [BiomedParse v1](https://huggingface.co/microsoft/BiomedParse) (`biomedparse_v1.pt`) weights, CC BY-NC-SA 4.0, research and development use only | The dataset terms above plus these model terms: use the scores for non-commercial research only, and credit the model sources. Phikon (Owkin non-commercial licence) is excluded |
| `heldout/msd_spleen`, `heldout/msd_heart` | Medical Segmentation Decathlon Task09 Spleen and Task02 Heart test volumes | The 11 primary bundles | MSD CC BY-SA 4.0, with ShareAlike applying where the scores are adaptations; MAE CC BY-NC 4.0 |
| `pixel_overlap/*` | Kvasir-SEG, ACDC and RIGA+ test cases | The 11 primary bundles | As for the primary panels; derived from the heads' predictions and the ground-truth masks, neither of which is released |
| `amos/*` | AMOS22 ([Zenodo 7155725](https://doi.org/10.5281/zenodo.7155725)), first 60 labelled CT and all 60 labelled MRI volumes | [MRSegmentator](https://github.com/hhaentze/MRSegmentator) v1.2.0 folds 0 and 1, Apache-2.0; [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) 2.4.0 `total` and `total_mr`, Apache-2.0, openly available for any usage | AMOS22 CC BY 4.0 (cite Ji et al. 2022); only per-volume and per-organ scores are released |

## Intended use

The traces support recomputing case-level dependence statistics, inspecting the aggregate evidence and reusing the protocol for evaluation research. They are not intended for individual profiling, re-identification or clinical diagnosis; stable alignment identifiers do not guarantee anonymity. The statistics are not validated for clinical deployment decisions. Training, image-level analyses and analyses not listed in the [reproduction guide](../REPRODUCING.md) require inputs outside the release, including the upstream datasets and models (paper Appendix J).

The files are fixed to the paper's version. Please report errors or problems through the repository's [GitHub issues](https://github.com/eunseobchoi/CoFailMed/issues).
