from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ID_COLUMNS = {
    "date",
    "sequence_num",
    "cow_tag",
    "temperature_f",
    "raw_frame_count",
    "sampled_frame_count",
    "detected_frame_count",
}

ROI_REGIONS = [
    "face_bbox",
    "left_eye",
    "right_eye",
    "muzzle",
    "left_nostril",
    "right_nostril",
    "mouth",
    "nostrils_box",
    "lower_face",
]

HOT_STATS = [
    "max_mean",
    "max_p95",
    "p99_mean",
    "p95_mean",
    "top5_mean_mean",
    "p90_mean",
    "mean_mean",
    "p50_mean",
]

PAIR_STATS = [
    "max_mean",
    "p99_mean",
    "p95_mean",
    "top5_mean_mean",
    "mean_mean",
]


def read_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed = {}
            for key, value in row.items():
                if key in {"date", "sequence_num", "cow_tag"}:
                    parsed[key] = value
                elif value == "":
                    parsed[key] = math.nan
                else:
                    try:
                        parsed[key] = float(value)
                    except ValueError:
                        parsed[key] = value
            rows.append(parsed)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    fields = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def is_number(value: object) -> bool:
    return isinstance(value, (float, int)) and math.isfinite(float(value))


def value(row: dict[str, object], *names: str) -> float | None:
    for name in names:
        candidate = row.get(name)
        if is_number(candidate):
            return float(candidate)
    return None


def roi_name(region: str, stat: str) -> tuple[str, str]:
    return f"roi_{region}_{stat}", f"{region}_{stat}"


def raw_name(region: str, stat: str) -> tuple[str, str]:
    return f"raw_{region}_{stat}", f"{region}_{stat}"


def c_to_f(temp_c: float) -> float:
    return temp_c * 9.0 / 5.0 + 32.0


def kelvin(temp_c: float) -> float:
    return temp_c + 273.15


def radiance_delta(hot_c: float, ambient_c: float) -> float:
    return (kelvin(hot_c) ** 4 - kelvin(ambient_c) ** 4) / 1e8


def add_if_finite(features: dict[str, object], name: str, number: float | None) -> None:
    if number is not None and math.isfinite(float(number)):
        features[name] = float(number)


def average(values: list[float | None]) -> float | None:
    finite = [float(item) for item in values if item is not None and math.isfinite(float(item))]
    if not finite:
        return None
    return sum(finite) / len(finite)


def weighted_average(items: list[tuple[float | None, float]]) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for number, weight in items:
        if number is None or not math.isfinite(float(number)):
            continue
        numerator += float(number) * weight
        denominator += weight
    if denominator == 0:
        return None
    return numerator / denominator


def ratio_delta(hot_c: float, ambient_c: float) -> float:
    return (hot_c - ambient_c) / max(abs(ambient_c), 1e-6)


def row_feature(row: dict[str, object], region: str, stat: str) -> float | None:
    return value(row, *roi_name(region, stat))


def ambient_features(row: dict[str, object]) -> dict[str, float]:
    ambient = {}
    ambient_p50 = value(row, *roi_name("face_surround", "p50_mean"), *raw_name("whole", "p50_mean"))
    scene_mean = value(row, *roi_name("face_surround", "mean_mean"), *raw_name("whole", "mean_mean"))
    scene_min = value(row, *roi_name("face_surround", "min_mean"), *raw_name("whole", "min_mean"))
    scene_cool = value(row, *raw_name("whole", "p50_min"), *raw_name("whole", "min_mean"))
    scene_hot = value(row, *roi_name("face_surround", "p95_mean"), *raw_name("whole", "p95_mean"), *raw_name("whole", "top5_mean_mean"))

    if ambient_p50 is None:
        ambient_p50 = value(row, *roi_name("face_bbox", "min_mean"))
    if scene_mean is None:
        scene_mean = value(row, *roi_name("face_bbox", "mean_mean"))
    if scene_min is None:
        scene_min = value(row, *roi_name("face_bbox", "min_mean"))

    add_if_finite(ambient, "surrounding_ambient_p50_c", ambient_p50)
    add_if_finite(ambient, "surrounding_scene_mean_c", scene_mean)
    add_if_finite(ambient, "surrounding_scene_min_c", scene_min)
    add_if_finite(ambient, "surrounding_scene_cool_c", scene_cool)
    add_if_finite(ambient, "surrounding_scene_hot_c", scene_hot)
    for key, number in list(ambient.items()):
        add_if_finite(ambient, key.replace("_c", "_f"), c_to_f(number))

    if ambient_p50 is not None and scene_hot is not None:
        add_if_finite(ambient, "surrounding_scene_hot_minus_ambient_c", scene_hot - ambient_p50)
    if scene_mean is not None and scene_min is not None:
        add_if_finite(ambient, "surrounding_scene_mean_minus_min_c", scene_mean - scene_min)
    return ambient


def thermal_balance_features(row: dict[str, object]) -> dict[str, object]:
    features: dict[str, object] = {}
    ambient = ambient_features(row)
    features.update(ambient)
    ambient_c = value(features, "surrounding_ambient_p50_c", "surrounding_scene_mean_c")
    scene_mean_c = value(features, "surrounding_scene_mean_c", "surrounding_ambient_p50_c")

    for region in ROI_REGIONS:
        for stat in HOT_STATS:
            observed = row_feature(row, region, stat)
            if observed is None:
                continue
            add_if_finite(features, f"infrared_{region}_{stat}_c", observed)
            add_if_finite(features, f"infrared_{region}_{stat}_f", c_to_f(observed))
            if ambient_c is not None:
                add_if_finite(features, f"balance_{region}_{stat}_minus_ambient_c", observed - ambient_c)
                add_if_finite(features, f"balance_{region}_{stat}_ambient_ratio", ratio_delta(observed, ambient_c))
                add_if_finite(features, f"balance_{region}_{stat}_radiance_delta", radiance_delta(observed, ambient_c))
            if scene_mean_c is not None:
                add_if_finite(features, f"balance_{region}_{stat}_minus_scene_mean_c", observed - scene_mean_c)

    for stat in PAIR_STATS:
        left_eye = row_feature(row, "left_eye", stat)
        right_eye = row_feature(row, "right_eye", stat)
        left_nostril = row_feature(row, "left_nostril", stat)
        right_nostril = row_feature(row, "right_nostril", stat)
        eye_mean = average([left_eye, right_eye])
        nostril_mean = average([left_nostril, right_nostril])
        nostrils_box = row_feature(row, "nostrils_box", stat)
        face = row_feature(row, "face_bbox", stat)
        lower_face = row_feature(row, "lower_face", stat)
        muzzle = row_feature(row, "muzzle", stat)
        mouth = row_feature(row, "mouth", stat)

        if left_eye is not None and right_eye is not None:
            add_if_finite(features, f"symmetry_eye_{stat}_absdiff_c", abs(left_eye - right_eye))
            add_if_finite(features, f"symmetry_eye_{stat}_mean_c", (left_eye + right_eye) / 2.0)
        if left_nostril is not None and right_nostril is not None:
            add_if_finite(features, f"symmetry_nostril_{stat}_absdiff_c", abs(left_nostril - right_nostril))
            add_if_finite(features, f"symmetry_nostril_{stat}_mean_c", (left_nostril + right_nostril) / 2.0)
        if nostril_mean is not None and eye_mean is not None:
            add_if_finite(features, f"gradient_nostril_minus_eye_{stat}_c", nostril_mean - eye_mean)
        if nostrils_box is not None and face is not None:
            add_if_finite(features, f"gradient_nostrils_box_minus_face_{stat}_c", nostrils_box - face)
        if lower_face is not None and face is not None:
            add_if_finite(features, f"gradient_lower_face_minus_face_{stat}_c", lower_face - face)
        if muzzle is not None and mouth is not None:
            add_if_finite(features, f"gradient_muzzle_minus_mouth_{stat}_c", muzzle - mouth)
        if nostril_mean is not None and lower_face is not None:
            add_if_finite(features, f"gradient_nostril_minus_lower_face_{stat}_c", nostril_mean - lower_face)

    nostril_hot = average(
        [
            row_feature(row, "left_nostril", "top5_mean_mean"),
            row_feature(row, "right_nostril", "top5_mean_mean"),
            row_feature(row, "nostrils_box", "top5_mean_mean"),
        ]
    )
    internal_hot = weighted_average(
        [
            (row_feature(row, "nostrils_box", "top5_mean_mean"), 0.32),
            (row_feature(row, "left_nostril", "top5_mean_mean"), 0.14),
            (row_feature(row, "right_nostril", "top5_mean_mean"), 0.14),
            (row_feature(row, "lower_face", "p95_mean"), 0.18),
            (row_feature(row, "muzzle", "p95_mean"), 0.14),
            (row_feature(row, "mouth", "p95_mean"), 0.08),
        ]
    )
    surface_mean = average(
        [
            row_feature(row, "face_bbox", "mean_mean"),
            row_feature(row, "lower_face", "mean_mean"),
            row_feature(row, "muzzle", "mean_mean"),
            row_feature(row, "mouth", "mean_mean"),
        ]
    )

    add_if_finite(features, "internal_nostril_hot_proxy_c", nostril_hot)
    add_if_finite(features, "internal_hot_proxy_c", internal_hot)
    add_if_finite(features, "infrared_surface_proxy_c", surface_mean)
    if internal_hot is not None:
        add_if_finite(features, "internal_hot_proxy_f", c_to_f(internal_hot))
    if surface_mean is not None:
        add_if_finite(features, "infrared_surface_proxy_f", c_to_f(surface_mean))

    if internal_hot is not None and ambient_c is not None:
        delta = internal_hot - ambient_c
        add_if_finite(features, "internal_hot_minus_ambient_c", delta)
        add_if_finite(features, "internal_hot_ambient_ratio", ratio_delta(internal_hot, ambient_c))
        add_if_finite(features, "internal_hot_radiance_delta", radiance_delta(internal_hot, ambient_c))
        for alpha in (0.10, 0.20, 0.35, 0.50):
            corrected = internal_hot + alpha * delta
            tag = str(alpha).replace(".", "p")
            add_if_finite(features, f"internal_newton_corrected_alpha_{tag}_c", corrected)
            add_if_finite(features, f"internal_newton_corrected_alpha_{tag}_f", c_to_f(corrected))

    if internal_hot is not None and surface_mean is not None:
        add_if_finite(features, "internal_hot_minus_surface_c", internal_hot - surface_mean)
    if surface_mean is not None and ambient_c is not None:
        add_if_finite(features, "surface_minus_ambient_c", surface_mean - ambient_c)

    frontal = value(row, "roi_quality_frontal_score_mean", "quality_frontal_score_mean")
    face_area = value(row, "roi_quality_face_area_frac_mean", "quality_face_area_frac_mean")
    center_offset = value(row, "roi_quality_center_offset_x_mean", "quality_center_offset_x_mean")
    if internal_hot is not None and frontal is not None:
        add_if_finite(features, "quality_weighted_internal_hot_c", internal_hot * frontal)
    if ambient_c is not None and internal_hot is not None and face_area is not None:
        add_if_finite(features, "area_scaled_internal_delta", (internal_hot - ambient_c) * face_area)
    if internal_hot is not None and center_offset is not None:
        add_if_finite(features, "center_penalized_internal_hot_c", internal_hot * (1.0 - min(center_offset, 1.0)))

    return features


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build thermal balance features that separate surrounding, infrared, and internal-proxy signals."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--engineered-only", action="store_true")
    args = parser.parse_args()

    rows = read_rows(args.input)
    output_rows = []
    engineered_counts = []
    for row in rows:
        output = {key: row.get(key, "") for key in ID_COLUMNS if key in row}
        if not args.engineered_only:
            for key, item in row.items():
                if key not in output:
                    output[key] = item
        engineered = thermal_balance_features(row)
        output.update(engineered)
        engineered_counts.append(len(engineered))
        output_rows.append(output)

    write_csv(args.output, output_rows)
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "rows": len(output_rows),
        "engineered_only": args.engineered_only,
        "engineered_feature_count_min": min(engineered_counts),
        "engineered_feature_count_max": max(engineered_counts),
        "engineered_feature_count_mean": sum(engineered_counts) / len(engineered_counts),
        "concepts": [
            "surrounding ambient/background thermal estimate",
            "observed infrared ROI temperature",
            "ROI minus ambient thermal deltas",
            "left/right symmetry",
            "anatomical thermal gradients",
            "internal hot proxy",
            "Newton-style ambient correction",
            "radiance delta approximation",
            "quality-weighted thermal proxies",
        ],
    }
    with (args.output.parent / "engineering_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Saved:", args.output)
    print("Rows:", len(output_rows))
    print("Engineered features per row:", int(summary["engineered_feature_count_mean"]))


if __name__ == "__main__":
    main()
