from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "python38"
SKLEARN_COMPAT = Path(__file__).resolve().parent / "_sklearn_compat"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))
if SKLEARN_COMPAT.exists():
    sys.path.insert(0, str(SKLEARN_COMPAT))

import joblib
import numpy as np


ID_COLUMNS = {"date", "sequence_num", "cow_tag", "temperature_f"}
DEFAULT_MODEL_DIR = Path("data/temperature_outputs/detected_article_otsu_fusion_ridge_k20_v1")
DEFAULT_FEATURES = Path("data/temperature_outputs/detected_article_otsu_fusion_v1/features.csv")
DEFAULT_RAW_ZIP = Path("data/thermal_raw.zip")
DEFAULT_FUSION_MODEL_DIR = Path("data/temperature_outputs/deployment_fusion_cnn_article_otsu_top10_qualityframes_full_v1")
DEFAULT_FUSION_MODEL_DIRS = [
    DEFAULT_FUSION_MODEL_DIR,
    Path("data/temperature_outputs/deployment_fusion_cnn_article_otsu_top10_qualityframes_seed42_full_v1"),
    Path("data/temperature_outputs/deployment_fusion_cnn_article_otsu_top10_qualityframes_seed2026_full_v1"),
    Path("data/temperature_outputs/deployment_fusion_cnn_article_otsu_top10_qualityframes_seed7_full_v1"),
]
DEFAULT_FUSION_METRICS_DIR = Path(
    "data/temperature_outputs/thermal_feature_fusion_cnn_article_otsu_top10_qualityframes_seed_ensemble_equal_v1"
)
DEFAULT_MULTI_ROI_MODEL_DIR = Path("data/temperature_outputs/deployment_multi_roi_quality_cnn_reg80_full_v1")


def parse_value(name: str, value: str) -> object:
    if name in {"date", "sequence_num", "cow_tag"}:
        return value
    if value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return value


def read_feature_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({name: parse_value(name, value) for name, value in row.items()})
    return rows


def read_csv_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def finite_float(value: object) -> float | None:
    if isinstance(value, (float, int)) and math.isfinite(float(value)):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


class TemperatureService:
    def __init__(
        self,
        model_dir: Path,
        features_csv: Path,
        model_name: str = "model_full.joblib",
        raw_zip: Path = DEFAULT_RAW_ZIP,
        fusion_model_dir: Path | None = None,
        fusion_model_dirs: list[Path] | None = None,
        fusion_weights: list[float] | None = None,
        multi_roi_model_dir: Path | None = None,
        hybrid_weights: list[float] | None = None,
        fusion_metrics_dir: Path | None = None,
    ):
        self.model_dir = model_dir
        self.features_csv = features_csv
        self.raw_zip = raw_zip
        self.model = joblib.load(model_dir / model_name)
        with (model_dir / "feature_schema.json").open("r", encoding="utf-8") as f:
            self.schema = json.load(f)
        with (model_dir / "metrics.json").open("r", encoding="utf-8") as f:
            self.metrics = json.load(f)
        self.feature_names = list(self.schema["feature_names"])
        self.rows = read_feature_rows(features_csv)
        self.rows_by_key = {(str(row["date"]), str(row["sequence_num"])): row for row in self.rows}
        self.selected_features = read_csv_rows(model_dir / "selected_features_full.csv")
        candidate_fusion_dirs = list(fusion_model_dirs or [])
        if not candidate_fusion_dirs and fusion_model_dir:
            candidate_fusion_dirs = [fusion_model_dir]
        self.fusion_model_dirs = [model_dir for model_dir in candidate_fusion_dirs if model_dir.exists()]
        self.fusion_model_dir = self.fusion_model_dirs[0] if self.fusion_model_dirs else None
        if fusion_weights and self.fusion_model_dirs:
            weights = np.asarray(fusion_weights, dtype=np.float32)
            if len(weights) != len(self.fusion_model_dirs):
                raise RuntimeError("--fusion-weights must match --fusion-model-dirs count.")
            weight_sum = float(weights.sum())
            if weight_sum <= 0:
                raise RuntimeError("--fusion-weights must sum to a positive value.")
            self.fusion_weights = (weights / weight_sum).astype(np.float32)
        elif self.fusion_model_dirs:
            self.fusion_weights = np.full((len(self.fusion_model_dirs),), 1.0 / len(self.fusion_model_dirs), dtype=np.float32)
        else:
            self.fusion_weights = np.asarray([], dtype=np.float32)
        self.multi_roi_model_dir = multi_roi_model_dir if multi_roi_model_dir and multi_roi_model_dir.exists() else None
        if hybrid_weights:
            weights = np.asarray(hybrid_weights, dtype=np.float32)
            if len(weights) != 2:
                raise RuntimeError("--hybrid-weights must contain two values: fusion ensemble, multi-ROI.")
            weight_sum = float(weights.sum())
            if weight_sum <= 0:
                raise RuntimeError("--hybrid-weights must sum to a positive value.")
            self.hybrid_weights = (weights / weight_sum).astype(np.float32)
        else:
            self.hybrid_weights = np.asarray([0.7, 0.3], dtype=np.float32)
        self.fusion_metrics_dir = fusion_metrics_dir if fusion_metrics_dir and fusion_metrics_dir.exists() else None
        self.fusion_metrics = None
        self.fusion_feature_names: list[str] = []
        if self.fusion_metrics_dir:
            with (self.fusion_metrics_dir / "metrics.json").open("r", encoding="utf-8") as f:
                self.fusion_metrics = json.load(f)
        if self.fusion_model_dirs:
            import torch

            checkpoint = torch.load(
                self.fusion_model_dirs[0] / "thermal_feature_fusion_cnn.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.fusion_feature_names = list(checkpoint["state"]["feature_names"])

    def primary_metrics(self) -> dict[str, object]:
        return self.fusion_metrics or self.metrics

    def primary_holdout_metrics(self) -> dict[str, object]:
        metrics = self.primary_metrics()
        holdout = metrics.get("holdout", {})
        if isinstance(holdout, dict) and "metrics" in holdout:
            return holdout["metrics"]
        return holdout if isinstance(holdout, dict) else {}

    def model_info(self) -> dict[str, object]:
        metrics = self.primary_metrics()
        validation = {
            row["grouping"]: {
                "mae": row["mae_mean"],
                "rmse": row["rmse_mean"],
            }
            for row in metrics.get("validation", [])
        }
        holdout = self.primary_holdout_metrics()
        selected_count = len(self.fusion_feature_names) if self.fusion_model_dirs else self.metrics.get("selected_feature_count")
        if self.multi_roi_model_dir and self.fusion_model_dirs:
            model_name = "hybrid_fusion_cnn_multi_roi_quality"
        elif self.multi_roi_model_dir:
            model_name = "multi_roi_quality_fusion_cnn"
        elif len(self.fusion_model_dirs) > 1:
            model_name = "fusion_cnn_article_otsu_top10_qualityframes_seed_ensemble"
        elif self.fusion_model_dirs:
            model_name = "fusion_cnn_article_otsu_top10_qualityframes"
        else:
            model_name = self.metrics.get("model", self.schema.get("model_name", ""))
        return {
            "model_dir": str(self.fusion_model_dir or self.model_dir),
            "fusion_model_dirs": [str(model_dir) for model_dir in self.fusion_model_dirs],
            "fusion_weights": [float(weight) for weight in self.fusion_weights],
            "multi_roi_model_dir": str(self.multi_roi_model_dir) if self.multi_roi_model_dir else None,
            "hybrid_weights": [float(weight) for weight in self.hybrid_weights] if self.multi_roi_model_dir else None,
            "classical_model_dir": str(self.model_dir),
            "features_csv": str(self.features_csv),
            "model": model_name,
            "selected_feature_count": selected_count,
            "feature_count": metrics.get("feature_count", selected_count),
            "sample_count": metrics.get("usable_labeled_videos", len(self.rows)),
            "cv_prediction_count": metrics.get("sample_count"),
            "holdout": holdout,
            "validation": validation,
            "holdout_test_videos": metrics.get("holdout_test_videos", metrics.get("holdout", {}).get("test_videos", [])),
        }

    def sequences(self) -> list[dict[str, object]]:
        output = []
        for row in self.rows:
            output.append(
                {
                    "date": row["date"],
                    "sequence_num": row["sequence_num"],
                    "cow_tag": row.get("cow_tag", ""),
                    "temperature_f": row.get("temperature_f"),
                    "detected_frame_count": row.get("detected_frame_count"),
                    "fusion_ambient_proxy_c": row.get("fusion_ambient_proxy_c"),
                    "fusion_internal_hot_proxy_c": row.get("fusion_internal_hot_proxy_c"),
                }
            )
        return output

    def predict(self, date: str, sequence_num: str, threshold_f: float) -> dict[str, object]:
        key = (date, sequence_num)
        if key not in self.rows_by_key:
            raise KeyError(f"No feature row found for {date}/{sequence_num}")
        row = self.rows_by_key[key]
        x = np.asarray([[row.get(name, np.nan) for name in self.feature_names]], dtype=np.float32)
        classical_prediction = float(self.model.predict(x)[0])
        prediction = classical_prediction
        prediction_source = "classical_roi"
        comparison_predictions = [
            {
                "model": self.metrics.get("model", "classical_roi"),
                "prediction_f": classical_prediction,
            }
        ]
        if self.fusion_model_dirs:
            from predict_temperature_system import fusion_prediction

            fusion_predictions = []
            for weight, model_dir in zip(self.fusion_weights, self.fusion_model_dirs):
                model_prediction = float(fusion_prediction(model_dir, self.raw_zip, date, sequence_num, row))
                fusion_predictions.append(model_prediction)
                comparison_predictions.append(
                    {
                        "model": model_dir.name,
                        "prediction_f": model_prediction,
                        "ensemble_weight": float(weight),
                    }
                )
            prediction = float(np.dot(self.fusion_weights, np.asarray(fusion_predictions, dtype=np.float32)))
            prediction_source = (
                "fusion_cnn_article_otsu_top10_qualityframes_seed_ensemble"
                if len(self.fusion_model_dirs) > 1
                else "fusion_cnn_article_otsu_top10_qualityframes"
            )
            if len(self.fusion_model_dirs) > 1:
                comparison_predictions.append(
                    {
                        "model": prediction_source,
                        "prediction_f": prediction,
                    }
                )
        if self.multi_roi_model_dir:
            from predict_temperature_system import multi_roi_quality_prediction

            multi_roi_prediction = float(multi_roi_quality_prediction(self.multi_roi_model_dir, self.raw_zip, date, sequence_num, row))
            comparison_predictions.append(
                {
                    "model": self.multi_roi_model_dir.name,
                    "prediction_f": multi_roi_prediction,
                    "hybrid_weight": float(self.hybrid_weights[1]),
                }
            )
            if self.fusion_model_dirs:
                prediction = float(
                    np.dot(
                        self.hybrid_weights,
                        np.asarray([prediction, multi_roi_prediction], dtype=np.float32),
                    )
                )
                prediction_source = "hybrid_fusion_cnn_multi_roi_quality"
                comparison_predictions.append(
                    {
                        "model": prediction_source,
                        "prediction_f": prediction,
                    }
                )
            else:
                prediction = multi_roi_prediction
                prediction_source = "multi_roi_quality_fusion_cnn"

        holdout_rmse = finite_float(self.primary_holdout_metrics().get("rmse"))
        if holdout_rmse is None:
            sequence_validation = self.model_info().get("validation", {}).get("sequence", {})
            holdout_rmse = finite_float(sequence_validation.get("rmse")) or 1.0
        z = (prediction - threshold_f) / max(holdout_rmse, 1e-6)
        probability = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        if probability >= 0.67:
            flag = "elevated"
        elif probability >= 0.33:
            flag = "watch"
        else:
            flag = "below_threshold"

        truth = finite_float(row.get("temperature_f"))
        selected = []
        if self.fusion_feature_names:
            for rank, feature in enumerate(self.fusion_feature_names[:30], start=1):
                selected.append(
                    {
                        "rank": rank,
                        "feature": feature,
                        "value": row.get(feature),
                        "f_score": None,
                        "model_importance": None,
                    }
                )
        else:
            for item in self.selected_features[:30]:
                feature = item.get("feature", "")
                selected.append(
                    {
                        "rank": int(float(item.get("rank", 0) or 0)),
                        "feature": feature,
                        "value": row.get(feature),
                        "f_score": finite_float(item.get("f_score")),
                        "model_importance": finite_float(item.get("model_importance")),
                    }
                )

        result = {
            "date": date,
            "sequence_num": sequence_num,
            "cow_tag": row.get("cow_tag", ""),
            "prediction_f": prediction,
            "prediction_source": prediction_source,
            "comparison_predictions": comparison_predictions,
            "threshold_f": threshold_f,
            "expected_error_rmse_f": holdout_rmse,
            "fever_probability_research": probability,
            "fever_flag": flag,
            "ambient_proxy_c": row.get("fusion_ambient_proxy_c"),
            "internal_hot_proxy_c": row.get("fusion_internal_hot_proxy_c"),
            "selected_features": selected,
        }
        if truth is not None:
            result["temperature_f"] = truth
            result["error_f"] = prediction - truth
        return result


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Cattle Temperature Workbench</title>
  <style>
    :root {
      --bg: #f5f7f7;
      --panel: #ffffff;
      --ink: #172124;
      --muted: #5f6f72;
      --line: #d8e0df;
      --teal: #147a75;
      --amber: #b36b00;
      --red: #af3030;
      --blue: #2e5f9e;
      --shadow: 0 8px 24px rgba(23, 33, 36, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Segoe UI, Arial, sans-serif;
      line-height: 1.35;
    }
    header {
      background: #ffffff;
      border-bottom: 1px solid var(--line);
      padding: 14px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 650;
      letter-spacing: 0;
    }
    .status {
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
    main {
      max-width: 1180px;
      margin: 0 auto;
      padding: 18px;
    }
    .toolbar {
      display: grid;
      grid-template-columns: repeat(5, minmax(130px, 1fr));
      gap: 12px;
      align-items: end;
      margin-bottom: 16px;
    }
    label {
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }
    select, input, button {
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 10px;
      font: inherit;
      background: #fff;
      color: var(--ink);
    }
    button {
      background: var(--teal);
      color: #fff;
      border-color: var(--teal);
      font-weight: 650;
      cursor: pointer;
    }
    button:disabled {
      opacity: 0.55;
      cursor: default;
    }
    .grid {
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 16px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 16px;
      min-width: 0;
    }
    .panel h2 {
      margin: 0 0 12px;
      font-size: 15px;
      letter-spacing: 0;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 12px;
      min-height: 80px;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }
    .metric strong {
      display: block;
      font-size: 24px;
      line-height: 1.05;
      overflow-wrap: anywhere;
    }
    .flag {
      display: inline-flex;
      align-items: center;
      height: 26px;
      padding: 0 9px;
      border-radius: 6px;
      background: #e9f5f3;
      color: var(--teal);
      font-weight: 700;
      font-size: 12px;
      text-transform: uppercase;
    }
    .flag.watch { background: #fff3dd; color: var(--amber); }
    .flag.elevated { background: #ffe8e8; color: var(--red); }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 13px;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 8px 6px;
      text-align: left;
      vertical-align: middle;
      overflow-wrap: anywhere;
    }
    th {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0;
    }
    .bar {
      width: 100%;
      height: 8px;
      background: #e8eeee;
      border-radius: 4px;
      overflow: hidden;
    }
    .bar div {
      height: 100%;
      background: var(--blue);
      width: 0%;
    }
    .muted { color: var(--muted); }
    .split {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
    }
    .mini {
      border-left: 3px solid var(--teal);
      padding: 8px 10px;
      background: #f8fbfb;
      min-height: 62px;
    }
    .mini span {
      display: block;
      color: var(--muted);
      font-size: 12px;
    }
    .mini strong { font-size: 18px; }
    @media (max-width: 900px) {
      .toolbar, .grid, .metrics, .split {
        grid-template-columns: 1fr;
      }
      header {
        align-items: flex-start;
        flex-direction: column;
      }
      .status { white-space: normal; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Cattle Temperature Workbench</h1>
    <div class="status" id="modelStatus">Loading model</div>
  </header>
  <main>
    <section class="toolbar">
      <label>Date<select id="dateSelect"></select></label>
      <label>Sequence<select id="sequenceSelect"></select></label>
      <label>Threshold F<input id="thresholdInput" type="number" step="0.1" value="103.5" /></label>
      <label>Model<select id="modelSelect"><option>loading</option></select></label>
      <button id="predictButton">Predict</button>
    </section>

    <section class="grid">
      <div class="panel">
        <h2>Prediction</h2>
        <div class="metrics">
          <div class="metric"><span>Predicted F</span><strong id="predicted">--</strong></div>
          <div class="metric"><span>Truth F</span><strong id="truth">--</strong></div>
          <div class="metric"><span>Error F</span><strong id="error">--</strong></div>
          <div class="metric"><span>Flag</span><strong><span class="flag" id="flag">--</span></strong></div>
        </div>
        <div class="split" style="margin-top: 14px;">
          <div class="mini"><span>Risk</span><strong id="risk">--</strong></div>
          <div class="mini"><span>Ambient C</span><strong id="ambient">--</strong></div>
          <div class="mini"><span>Hot Proxy C</span><strong id="hotProxy">--</strong></div>
        </div>
      </div>
      <div class="panel">
        <h2>Validation</h2>
        <table>
          <thead><tr><th>Split</th><th>MAE</th><th>RMSE</th></tr></thead>
          <tbody id="validationBody"></tbody>
        </table>
      </div>
    </section>

    <section class="panel" style="margin-top: 16px;">
      <h2>Selected Signals</h2>
      <table>
        <thead><tr><th style="width: 52px;">Rank</th><th>Feature</th><th style="width: 120px;">Value</th><th style="width: 130px;">Weight</th></tr></thead>
        <tbody id="featureBody"></tbody>
      </table>
    </section>
  </main>
  <script>
    const state = { sequences: [], latest: null };
    const fmt = (value, digits = 2) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "--";

    async function api(path, options) {
      const response = await fetch(path, options);
      if (!response.ok) throw new Error(await response.text());
      return await response.json();
    }

    function unique(items) {
      return [...new Set(items)];
    }

    function fillDates() {
      const dates = unique(state.sequences.map(item => item.date));
      dateSelect.innerHTML = dates.map(date => `<option value="${date}">${date}</option>`).join("");
      fillSequences();
    }

    function fillSequences() {
      const date = dateSelect.value;
      const items = state.sequences.filter(item => item.date === date);
      sequenceSelect.innerHTML = items.map(item => {
        const truth = Number.isFinite(Number(item.temperature_f)) ? ` | ${Number(item.temperature_f).toFixed(1)} F` : "";
        return `<option value="${item.sequence_num}">${item.sequence_num} | cow ${item.cow_tag}${truth}</option>`;
      }).join("");
    }

    function renderModel(info) {
      modelStatus.textContent = `${info.model} | ${info.sample_count} sequences | ${info.selected_feature_count}/${info.feature_count} features`;
      modelSelect.innerHTML = `<option>${info.model}</option>`;
      const rows = [
        ["holdout", info.holdout?.mae, info.holdout?.rmse],
        ["sequence", info.validation?.sequence?.mae, info.validation?.sequence?.rmse],
        ["cow", info.validation?.cow?.mae, info.validation?.cow?.rmse],
        ["date", info.validation?.date?.mae, info.validation?.date?.rmse],
      ];
      validationBody.innerHTML = rows.map(row => `<tr><td>${row[0]}</td><td>${fmt(row[1], 3)}</td><td>${fmt(row[2], 3)}</td></tr>`).join("");
    }

    function renderPrediction(result) {
      state.latest = result;
      predicted.textContent = fmt(result.prediction_f, 2);
      truth.textContent = fmt(result.temperature_f, 2);
      error.textContent = fmt(result.error_f, 2);
      risk.textContent = `${fmt(100 * result.fever_probability_research, 0)}%`;
      ambient.textContent = fmt(result.ambient_proxy_c, 2);
      hotProxy.textContent = fmt(result.internal_hot_proxy_c, 2);
      flag.textContent = result.fever_flag.replace("_", " ");
      flag.className = `flag ${result.fever_flag}`;

      const maxWeight = Math.max(...result.selected_features.map(item => Math.abs(Number(item.model_importance) || 0)), 1e-6);
      featureBody.innerHTML = result.selected_features.slice(0, 14).map(item => {
        const weight = Math.abs(Number(item.model_importance) || 0);
        const width = Math.max(2, Math.min(100, 100 * weight / maxWeight));
        return `<tr>
          <td>${item.rank}</td>
          <td>${item.feature}</td>
          <td>${fmt(item.value, 3)}</td>
          <td><div class="bar"><div style="width:${width}%"></div></div></td>
        </tr>`;
      }).join("");
    }

    async function predict() {
      predictButton.disabled = true;
      try {
        const payload = {
          date: dateSelect.value,
          sequence_num: sequenceSelect.value,
          threshold_f: Number(thresholdInput.value)
        };
        const result = await api("/api/predict", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload)
        });
        renderPrediction(result);
      } finally {
        predictButton.disabled = false;
      }
    }

    async function boot() {
      const [info, sequences] = await Promise.all([api("/api/model"), api("/api/sequences")]);
      state.sequences = sequences;
      renderModel(info);
      fillDates();
      await predict();
    }

    dateSelect.addEventListener("change", fillSequences);
    predictButton.addEventListener("click", predict);
    boot().catch(error => {
      modelStatus.textContent = error.message;
    });
  </script>
</body>
</html>
"""


class TemperatureHandler(BaseHTTPRequestHandler):
    service: TemperatureService

    def log_message(self, format: str, *args: object) -> None:
        return

    def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(json_safe(payload), indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self) -> None:
        body = HTML.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json_body(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        body = self.rfile.read(length).decode("utf-8")
        return json.loads(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_html()
            return
        if parsed.path == "/api/health":
            self.send_json({"ok": True})
            return
        if parsed.path == "/api/model":
            self.send_json(self.service.model_info())
            return
        if parsed.path == "/api/sequences":
            self.send_json(self.service.sequences())
            return
        if parsed.path == "/api/predict":
            params = parse_qs(parsed.query)
            try:
                result = self.service.predict(
                    date=params.get("date", [""])[0],
                    sequence_num=params.get("sequence_num", [""])[0],
                    threshold_f=float(params.get("threshold_f", ["103.5"])[0]),
                )
                self.send_json(result)
            except Exception as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/predict":
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            payload = self.read_json_body()
            result = self.service.predict(
                date=str(payload.get("date", "")),
                sequence_num=str(payload.get("sequence_num", "")),
                threshold_f=float(payload.get("threshold_f", 103.5)),
            )
            self.send_json(result)
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the cattle temperature fusion model API and UI.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, type=Path)
    parser.add_argument("--features-csv", default=DEFAULT_FEATURES, type=Path)
    parser.add_argument("--model-name", default="model_full.joblib")
    parser.add_argument("--raw-zip", default=DEFAULT_RAW_ZIP, type=Path)
    parser.add_argument("--fusion-model-dir", type=Path)
    parser.add_argument("--fusion-model-dirs", nargs="*", type=Path)
    parser.add_argument("--fusion-weights", nargs="*", type=float)
    parser.add_argument("--multi-roi-model-dir", type=Path)
    parser.add_argument("--hybrid-weights", nargs="*", type=float)
    parser.add_argument("--fusion-metrics-dir", default=DEFAULT_FUSION_METRICS_DIR, type=Path)
    parser.add_argument("--no-fusion-model", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args()

    fusion_model_dirs = []
    if not args.no_fusion_model:
        if args.fusion_model_dirs:
            fusion_model_dirs = args.fusion_model_dirs
        elif args.fusion_model_dir:
            fusion_model_dirs = [args.fusion_model_dir]
        else:
            fusion_model_dirs = DEFAULT_FUSION_MODEL_DIRS

    TemperatureHandler.service = TemperatureService(
        args.model_dir,
        args.features_csv,
        args.model_name,
        raw_zip=args.raw_zip,
        fusion_model_dirs=fusion_model_dirs,
        fusion_weights=None if args.no_fusion_model else args.fusion_weights,
        multi_roi_model_dir=None if args.no_fusion_model else args.multi_roi_model_dir,
        hybrid_weights=None if args.no_fusion_model else args.hybrid_weights,
        fusion_metrics_dir=None if args.no_fusion_model else args.fusion_metrics_dir,
    )
    server = ThreadingHTTPServer((args.host, args.port), TemperatureHandler)
    print(f"Serving cattle temperature UI at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
