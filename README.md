# Cattle Fever Temperature Model

This repository started from the cattle facial keypoint Detectron2 project. I
extended it into a CattleFever-style temperature pipeline that uses thermal
frames, detected face landmarks, article-style facial ROIs, and rectal
temperature labels.

The current working model is not a Transformer model. The best result so far is
a 4-seed ensemble that combines:

```text
raw thermal frames
detected cattle face/keypoints
article-style Otsu thermal ROI features
selected thermal balance features
small CNN + ROI feature fusion
```

The UI/API can show predicted internal temperature, fever risk, selected
signals, validation metrics, and a quality/confidence check for the input
frames.

## Current Status

The strongest stable baseline is:

```text
4-seed CNN + article-Otsu ROI feature fusion
```

Grouped cross-validation results:

| Grouped split | MAE F | MSE | RMSE F |
|---|---:|---:|---:|
| Sequence grouped | 0.705 | 0.619 | 0.781 |
| Cow grouped | 0.708 | 0.646 | 0.797 |
| Date grouped | 0.724 | 0.661 | 0.798 |

These are honest grouped-CV numbers, so the model is not being judged by random
frame leakage. The result is working, but it is still above the article-level
MSE target that we were trying to approach.

## What Is Included

Important folders and scripts:

```text
configs/                         Detectron2 cattle keypoint configs
temperature_model/               Temperature modeling pipeline
temperature_model/serve_temperature_ui.py
                                  Local API and browser UI
temperature_model/extract_article_otsu_roi_features.py
                                  Article-style Otsu ROI feature extraction
temperature_model/merge_roi_feature_sets.py
                                  Merge detected ROI + article ROI features
temperature_model/run_grouped_thermal_feature_fusion_cnn.py
                                  Grouped CV for CNN + ROI feature fusion
temperature_model/train_thermal_feature_fusion_cnn.py
                                  Train/deploy the fusion CNN
temperature_model/build_quality_gate_v2.py
                                  Frame quality/confidence scoring
```

Large raw data, TIFF folders, ZIP files, and most generated outputs are kept out
of git by `.gitignore`. The small locked model artifacts can be tracked, but the
raw thermal ZIP is about 4.2 GB and must be copied separately or stored in a
release/LFS/OneDrive location.

## Data Expected Locally

The temperature pipeline expects these local files/folders:

```text
data/annotations/metadata.csv
data/thermal_raw.zip
data/temperature_outputs/article_otsu_roi_v1/
data/temperature_outputs/detected_roi_filtered_80_v1/
data/temperature_outputs/detected_article_otsu_fusion_v1/
data/temperature_outputs/detected_article_otsu_fusion_ridge_k20_v1/
data/temperature_outputs/deployment_fusion_cnn_article_otsu_top10_qualityframes_full_v1/
data/temperature_outputs/deployment_fusion_cnn_article_otsu_top10_qualityframes_seed42_full_v1/
data/temperature_outputs/deployment_fusion_cnn_article_otsu_top10_qualityframes_seed2026_full_v1/
data/temperature_outputs/deployment_fusion_cnn_article_otsu_top10_qualityframes_seed7_full_v1/
data/temperature_outputs/quality_gate_v2_lenient_all/
```

If these folders are missing after cloning, copy them from the local project
backup or regenerate them using the commands below.

## Environment Setup

The project was run with Python 3.8 in a Conda environment named `d2.cattle`.

```powershell
conda create -n d2.cattle python=3.8 -y
conda activate d2.cattle

conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3 -c pytorch
pip install pycocotools
pip install opencv-python
pip install setuptools==59.5.0

python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```

Run all commands from the repository root.

## Start The API/UI

This starts the current local UI/API using the locked 4-seed ensemble if the
deployment folders are present:

```powershell
conda activate d2.cattle
python temperature_model\serve_temperature_ui.py --port 8770
```

Open:

```text
http://127.0.0.1:8770
```

Example API call:

```text
http://127.0.0.1:8770/api/predict?date=02_01&sequence_num=0001&threshold_f=103.5
```

The output includes:

```text
prediction_f
fever_flag
fever_probability_research
expected_error_rmse_f
ambient_proxy_c
internal_hot_proxy_c
quality_gate_v2
selected_features
```

`quality_gate_v2` is not the temperature model. It is a frame quality/confidence
check. It tells us whether the available cow-face frames look good enough to
trust the prediction.

## Rebuild The Main Temperature Features

Audit the temperature labels and raw thermal coverage:

```powershell
python temperature_model\audit_temperature_data.py
```

Extract article-style Otsu ROI features:

```powershell
python temperature_model\extract_article_otsu_roi_features.py `
  --max-frames 80 `
  --output-dir data\temperature_outputs\article_otsu_roi_v1
```

Merge detected ROI and article-Otsu ROI features:

```powershell
python temperature_model\merge_roi_feature_sets.py `
  --detected-features data\temperature_outputs\detected_roi_filtered_80_v1\features.csv `
  --article-features data\temperature_outputs\article_otsu_roi_v1\features.csv `
  --output-dir data\temperature_outputs\detected_article_otsu_fusion_v1
```

Lock the small classical ROI model used by the UI:

```powershell
python temperature_model\lock_best_roi_model.py `
  --features data\temperature_outputs\detected_article_otsu_fusion_v1\features.csv `
  --output-dir data\temperature_outputs\detected_article_otsu_fusion_ridge_k20_v1 `
  --select-k 20 `
  --model ridge `
  --split-metrics data\temperature_outputs\thermal_cnn_absolute_quick_lr1e3_v1\metrics.json
```

## Run Grouped CV For The Current CNN Fusion Baseline

Single-seed grouped CV:

```powershell
python temperature_model\run_grouped_thermal_feature_fusion_cnn.py `
  --features data\temperature_outputs\detected_article_otsu_fusion_v1\features.csv `
  --selected-features data\temperature_outputs\detected_article_otsu_fusion_ridge_k20_v1\selected_features_full.csv `
  --feature-limit 10 `
  --frame-filter-csv data\temperature_outputs\article_otsu_roi_v1\frame_detections.csv `
  --frame-score-column frontal_score `
  --output-dir data\temperature_outputs\thermal_feature_fusion_cnn_article_otsu_top10_qualityframes_grouped_v1 `
  --max-frames 8 `
  --epochs 250 `
  --batch-size 2 `
  --lr 0.0005 `
  --weight-decay 0.001 `
  --dropout 0.15 `
  --log-period 125
```

The current best result is the 4-seed ensemble of this model family. The
deployment folders are:

```text
deployment_fusion_cnn_article_otsu_top10_qualityframes_full_v1
deployment_fusion_cnn_article_otsu_top10_qualityframes_seed42_full_v1
deployment_fusion_cnn_article_otsu_top10_qualityframes_seed2026_full_v1
deployment_fusion_cnn_article_otsu_top10_qualityframes_seed7_full_v1
```

## Quality Gate V2

Quality gate v2 scores frames by face/keypoint geometry, frontal quality,
detection confidence, ROI size, and Otsu mask usability.

Build the current quality summary:

```powershell
python temperature_model\build_quality_gate_v2.py `
  --output-dir data\temperature_outputs\quality_gate_v2_lenient_all `
  --min-score 0.36 `
  --min-frontal-score 0.20 `
  --face-area-min 0.08 `
  --face-area-max 0.52 `
  --muzzle-symmetry-max 0.65
```

We tested using this gate for hard filtering and weighted frame selection. It
did not beat the locked 4-seed ensemble, so it should be used as confidence/QA,
not as a replacement model.

Best tested weighted-sampling candidate:

| Candidate | Sequence MSE | Cow MSE | Date MSE |
|---|---:|---:|---:|
| Frontal score-diverse sampling | 0.686 | 0.652 | 0.713 |
| Locked 4-seed ensemble | 0.619 | 0.646 | 0.661 |

## What We Tried

Tested approaches:

```text
raw thermal statistical baseline
detected ROI classical regressors
article-style Otsu ROI features
detected ROI + article ROI feature fusion
CNN + ROI feature fusion
4-seed CNN ensemble
multi-ROI branch CNN
quality-gated frame filtering
quality-weighted frame sampling
```

The CNN + article-Otsu ROI fusion ensemble is still the best stable model.
The multi-ROI branch and quality-gated frame filtering were useful experiments,
but they did not improve the grouped-CV result enough to replace the baseline.

We have not trained a Transformer model yet. With only 21 usable labeled raw
thermal sequences, a large Transformer is likely to overfit unless it is used as
a mostly frozen pretrained feature extractor or trained with extra unlabeled
data first.

## Where We Left Off

The next real improvement step should be:

```text
residual calibration under grouped CV
```

That means checking whether prediction errors are related to ambient proxy,
date, frame quality, hot-face proxy, or ROI confidence, and then testing a
leakage-safe residual correction model.

The next production step should be:

```text
new input -> keypoint detection -> article Otsu ROI extraction -> features -> 4-seed ensemble -> API/UI result
```

Right now the UI works best for sequences that already have precomputed feature
rows. For completely new uploaded images/videos, the preprocessing pipeline
still needs to be wrapped behind the API.

## Notes For GitHub Upload

The repository intentionally does not commit large raw data artifacts:

```text
data/
thermal_raw/
*.zip
vendor/
```

This keeps GitHub usable. The minimum model outputs are small enough to track,
but a fresh clone still needs `data/thermal_raw.zip` or equivalent raw TIFFs for
the CNN ensemble to run predictions.
