# Added CattleFever Temperature Pipeline

> Current handoff note: the top-level `README.md` is now the main setup and
> model-status document. This folder README keeps older command history for the
> temperature experiments. The current strongest baseline is the 4-seed
> CNN + article-Otsu ROI fusion ensemble, not the older
> `detected_roi_filtered_80_v1` Random Forest run described later in this file.

This folder is intentionally separate from the original Detectron2 keypoint
code. The original repo is the facial keypoint detector. This added module
extends it toward the article-level CattleFever workflow:

```
raw thermal TIFFs
  -> thermal keypoint detector
  -> facial ROI temperature features
  -> Random Forest rectal-temperature regression
```

## Commands

Run from the repo root.

```powershell
$py = "C:\Users\n.ghehalot\AppData\Local\miniconda3\envs\d2.cattle\python.exe"
```

Audit available temperature labels:

```powershell
& $py temperature_model\audit_temperature_data.py
```

Create synchronized paired RGB/thermal splits. A sample is included only when
RGB and thermal both exist for the same `folder + frame_id`, and both modalities
are assigned to the same split:

```powershell
& $py temperature_model\make_paired_rgb_thermal_split.py
```

Create grouped 5-fold RGB/thermal manifests for fold-by-fold training. Each fold
has its own `train`, `val`, and `test` pair CSVs plus RGB and thermal COCO JSONs.
By default this does not duplicate image files; the manifests point back to the
original image paths:

```powershell
& $py temperature_model\make_paired_rgb_thermal_folds.py
```

Audit/evaluate the 5-fold manifests. This trains one paired RGB/thermal
image-stat baseline per fold only when that fold has enough labeled train and
test rows; otherwise it writes the reason the fold was skipped:

```powershell
& $py temperature_model\run_paired_rgb_thermal_folds.py
```

Create consolidated performance and output-temperature tables from the paired
fold audit and the valid locked ROI evaluations:

```powershell
& $py temperature_model\summarize_temperature_performance.py
```

Create a focused table listing the eight missing labeled videos and every
paired/sequence/cow/date fold with its held-out groups and metrics:

```powershell
& $py temperature_model\create_fold_results_tables.py
```

Visualize paired RGB/thermal samples side by side for manual checking:

```powershell
& $py temperature_model\visualize_paired_split.py --split demo --limit 30 --output-dir data\outtest\paired_rgb_thermal_demo_check
```

Resolve which processed thermal folders can be linked back to raw TIFF date /
sequence folders, then create RGB + thermal + raw-temperature triples:

```powershell
& $py temperature_model\resolve_processed_raw_mapping.py --samples 12
& $py temperature_model\make_rgb_thermal_temperature_triples.py --min-score 0.15
```

Visualize raw thermal ROI extraction:

```powershell
& $py temperature_model\visualize_detected_roi_features.py --max-sequences 5 --frames-per-sequence 2 --output-dir data\outtest\temperature_roi_check
```

This writes:

```
datasets/keypoints/paired_rgb_thermal/
  annotations/
    pairs_train.csv
    pairs_val.csv
    pairs_test.csv
    pairs_demo.csv
    rgb_train.json
    thermal_train.json
    ...
  rgb/
    train_imgs/
    val_imgs/
    test_imgs/
    demo_imgs/
  thermal/
    train_imgs/
    val_imgs/
    test_imgs/
    demo_imgs/
```

Train a simple raw-pixel baseline:

```powershell
& $py temperature_model\train_raw_baseline.py --max-frames 80 --output-dir data\temperature_outputs\raw_baseline_v1
```

Extract detected keypoint ROI features from raw TIFF frames:

```powershell
& $py temperature_model\extract_detected_roi_features.py --max-frames 30 --output-dir data\temperature_outputs\detected_roi_v1
```

Extract only higher-quality/frontal ROI frames:

```powershell
& $py temperature_model\extract_detected_roi_features.py --max-frames 30 --output-dir data\temperature_outputs\detected_roi_filtered_v1 --quality-filter --include-quality-features --min-frontal-score 0.25 --max-eye-y-diff 0.20 --max-nostril-y-diff 0.20 --max-muzzle-center-offset 0.35 --max-muzzle-symmetry 0.55
& $py temperature_model\train_temperature_regressor.py --features data\temperature_outputs\detected_roi_filtered_v1\features.csv --output-dir data\temperature_outputs\detected_roi_filtered_v1
```

Run the same cleaned ROI pipeline with more sampled frames per sequence:

```powershell
& $py temperature_model\extract_detected_roi_features.py --max-frames 80 --output-dir data\temperature_outputs\detected_roi_filtered_80_v1 --quality-filter --include-quality-features --min-frontal-score 0.25 --max-eye-y-diff 0.20 --max-nostril-y-diff 0.20 --max-muzzle-center-offset 0.35 --max-muzzle-symmetry 0.55
& $py temperature_model\train_temperature_regressor.py --features data\temperature_outputs\detected_roi_filtered_80_v1\features.csv --output-dir data\temperature_outputs\detected_roi_filtered_80_v1
```

Train the ROI Random Forest regressor:

```powershell
& $py temperature_model\train_temperature_regressor.py --features data\temperature_outputs\detected_roi_v1\features.csv --output-dir data\temperature_outputs\detected_roi_v1
```

Combine raw full-frame features with detected ROI features, then train:

```powershell
& $py temperature_model\combine_features.py --left data\temperature_outputs\raw_baseline_v1\features.csv --right data\temperature_outputs\detected_roi_v1\features.csv --output data\temperature_outputs\combined_v1\features.csv
& $py temperature_model\train_temperature_regressor.py --features data\temperature_outputs\combined_v1\features.csv --output-dir data\temperature_outputs\combined_v1
```

Compare saved temperature runs:

```powershell
& $py temperature_model\compare_temperature_runs.py
```

Sweep feature-selection sizes for a feature table:

```powershell
& $py temperature_model\sweep_feature_selection.py --features data\temperature_outputs\detected_roi_filtered_80_v1\features.csv --output-dir data\temperature_outputs\detected_roi_filtered_80_select_sweep_v1 --ks 10 20 40 80 160 all
```

Compare several regressors on selected features:

```powershell
& $py temperature_model\compare_temperature_regressors.py --features data\temperature_outputs\detected_roi_filtered_80_v1\features.csv --output-dir data\temperature_outputs\detected_roi_filtered_80_sequence_model_compare_v1 --select-k 20
& $py temperature_model\compare_temperature_regressors.py --features data\temperature_outputs\detected_roi_filtered_80_v1\frame_features.csv --output-dir data\temperature_outputs\detected_roi_filtered_80_frame_model_compare_v1 --select-k 40
```

## Notes

- Raw TIFF pixels are used as temperature values in Celsius.
- Ground-truth labels are rectal temperature in Fahrenheit.
- Only rows with both `temperature_f` and raw TIFF frames are usable.
- The ROI model runs the trained thermal keypoint detector on sampled raw TIFF
  frames, scales predicted keypoints back to the raw thermal resolution, and
  extracts stats from face, nostril, muzzle, mouth, eye, and lower-face regions.

## Current Run

Using the available local data:

```
usable temperature sequences: 21
paired RGB/thermal samples: 1286
paired split: 900 train, 192 val, 128 test, 66 demo
paired 5-fold manifests: 1286 samples across 5 grouped folds
RGB/thermal/raw-temperature triples: 50 frame rows
detected ROI sampled frames: 609
unreadable raw TIFF samples skipped: 21
no-detection sampled frames: 0
```

Current Random Forest results:

```
raw_baseline_v1:
  test MAE  = 1.813 F
  test RMSE = 1.951 F
  5-fold MAE mean  = 1.128 F
  5-fold RMSE mean = 1.379 F

detected_roi_v1:
  test MAE  = 1.616 F
  test RMSE = 1.736 F
  5-fold MAE mean  = 1.152 F
  5-fold RMSE mean = 1.348 F

combined_v1:
  test MAE  = 1.609 F
  test RMSE = 1.731 F
  5-fold MAE mean  = 1.143 F
  5-fold RMSE mean = 1.337 F

detected_roi_filtered_v1:
  kept frames = 287
  quality-filtered frames = 322
  test MAE  = 1.520 F
  test RMSE = 1.630 F
  5-fold MAE mean  = 1.158 F
  5-fold RMSE mean = 1.332 F

combined_filtered_v1:
  test MAE  = 1.540 F
  test RMSE = 1.662 F
  5-fold MAE mean  = 1.164 F
  5-fold RMSE mean = 1.335 F

detected_roi_filtered_80_v1:
  kept frames = 805
  quality-filtered frames = 854
  test MAE  = 1.433 F
  test RMSE = 1.637 F
  5-fold MAE mean  = 1.075 F
  5-fold RMSE mean = 1.285 F

combined_filtered_80_v1:
  test MAE  = 1.463 F
  test RMSE = 1.679 F
  5-fold MAE mean  = 1.089 F
  5-fold RMSE mean = 1.299 F
```

The best current local temperature model is `detected_roi_filtered_80_v1`.
It uses only higher-quality/frontal detected ROI frames and more sampled frames
per temperature sequence. Combining raw full-frame thermal features with the ROI
features did not improve the result in this run.

Feature selection improves the sequence-level Random Forest result:

```
detected_roi_filtered_80_select_sweep_v1/top_20:
  selected features = 20
  test MAE  = 1.094 F
  test RMSE = 1.342 F
  5-fold MAE mean  = 1.088 F
  5-fold RMSE mean = 1.305 F
```

Comparing regressors on the same selected sequence-level features gives the
best holdout result with SVR:

```
detected_roi_filtered_80_sequence_model_compare_v1:
  best model = svr_rbf
  selected features = 20
  test MAE  = 0.952 F
  test MSE  = 1.318
  test RMSE = 1.148 F
```

An article-style frame-level experiment treats each accepted ROI frame as a
training sample with the sequence rectal temperature label. This is closer to
image-level evaluation and gives much lower error, but frames from the same
cattle sequence can appear in both train and test:

```
detected_roi_filtered_80_frame_model_compare_v1:
  samples = 805 accepted ROI frames
  best model = extra_trees
  selected features = 40
  test MAE  = 0.358 F
  test MSE  = 0.384
  test RMSE = 0.620 F
```

These are working models and a complete local pipeline, but the reported numbers
do not match the article-level error yet. The likely reasons are the small number
of usable labeled temperature sequences in this local copy and possible
differences from the paper's exact ROI/feature-selection setup.

The strict three-way synchronized set is much smaller than the individual
pipelines: only 50 frame-level rows link RGB, processed thermal, raw TIFF, and a
rectal temperature label in this local copy, and they currently come from one
temperature sequence. That is useful for manual verification, but not enough for
training a meaningful three-modal temperature regressor.

The grouped 5-fold RGB/thermal manifests are written to
`datasets/keypoints/paired_rgb_thermal_5fold`. They are ready for keypoint or
paired-modality experiments, but only 209 paired rows currently have a usable
rectal-temperature label, all from folder `1`. More labeled RGB/thermal folders
are needed before the 5-fold paired set can support meaningful temperature
accuracy comparisons.
