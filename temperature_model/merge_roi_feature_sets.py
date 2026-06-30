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

JOIN_COLUMNS = ("date", "sequence_num")
LABEL_COLUMNS = ("cow_tag", "temperature_f")
COUNT_COLUMNS = ("raw_frame_count", "sampled_frame_count", "detected_frame_count")

DETECTED_REGIONS = [
    "detected_face_bbox",
    "detected_left_eye",
    "detected_right_eye",
    "detected_muzzle",
    "detected_left_nostril",
    "detected_right_nostril",
    "detected_mouth",
    "detected_nostrils_box",
    "detected_lower_face",
]

ARTICLE_REGIONS = [
    "article_eyes",
    "article_nostrils",
    "article_forehead",
    "article_ears",
    "article_all",
]

HOT_STATS = [
    "mean_mean",
    "p50_mean",
    "p75_mean",
    "p90_mean",
    "p95_mean",
    "p99_mean",
    "top5_mean_mean",
    "max_mean",
]

PAIR_STATS = [
    "mean_mean",
    "p90_mean",
    "p95_mean",
    "p99_mean",
    "top5_mean_mean",
    "max_mean",
]


def parse_value(value: str) -> object:
    if value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return value


def read_rows(path: Path) -> dict[tuple[str, str], dict[str, object]]:
    rows: dict[tuple[str, str], dict[str, object]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = tuple(row[name] for name in JOIN_COLUMNS)
            rows[key] = {
                name: value if name in {"date", "sequence_num", "cow_tag"} else parse_value(value)
                for name, value in row.items()
            }
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


def is_finite(value: object) -> bool:
    return isinstance(value, (float, int)) and math.isfinite(float(value))


def get_number(row: dict[str, object], *names: str) -> float | None:
    for name in names:
        value = row.get(name)
        if is_finite(value):
            return float(value)
    return None


def add_if_finite(row: dict[str, object], name: str, value: float | None) -> None:
    if value is not None and math.isfinite(float(value)):
        row[name] = float(value)


def average(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None
    return sum(finite) / len(finite)


def ratio_delta(value: float, baseline: float) -> float:
    return (value - baseline) / max(abs(baseline), 1e-6)


def kelvin(celsius: float) -> float:
    return celsius + 273.15


def radiance_delta(value_c: float, baseline_c: float) -> float:
    return (kelvin(value_c) ** 4 - kelvin(baseline_c) ** 4) / 1e8


def c_to_f(celsius: float) -> float:
    return celsius * 9.0 / 5.0 + 32.0


def feature_name(source: str, name: str) -> str:
    if source == "detected":
        return f"detected_{name}"
    if name.startswith("article_"):
        return name
    return f"article_{name}"


def add_source_features(
    output: dict[str, object],
    source_row: dict[str, object],
    source: str,
    include_count_features: bool,
) -> None:
    for name, value in source_row.items():
        if name in JOIN_COLUMNS or name in LABEL_COLUMNS:
            continue
        if name in COUNT_COLUMNS:
            if include_count_features:
                output[f"{source}_{name}"] = value
            continue
        output[feature_name(source, name)] = value


def check_label_match(key: tuple[str, str], detected: dict[str, object], article: dict[str, object]) -> None:
    for name in LABEL_COLUMNS:
        left = detected.get(name)
        right = article.get(name)
        if str(left) != str(right):
            raise RuntimeError(f"Label mismatch for {key}: {name} {left!r} != {right!r}")


def ambient_value(row: dict[str, object]) -> float | None:
    return get_number(
        row,
        "detected_face_bbox_p50_mean",
        "detected_face_bbox_mean_mean",
        "article_all_p50_mean",
        "article_all_mean_mean",
    )


def scene_low_value(row: dict[str, object]) -> float | None:
    return get_number(
        row,
        "detected_face_bbox_min_mean",
        "article_all_min_mean",
        "detected_face_bbox_p50_min",
        "article_all_p50_min",
    )


def scene_hot_value(row: dict[str, object]) -> float | None:
    return get_number(
        row,
        "detected_face_bbox_p95_mean",
        "article_all_p95_mean",
        "detected_face_bbox_top5_mean_mean",
        "article_all_top5_mean_mean",
    )


def add_region_balance_features(row: dict[str, object]) -> int:
    added = 0
    ambient = ambient_value(row)
    scene_low = scene_low_value(row)
    scene_hot = scene_hot_value(row)
    add_if_finite(row, "fusion_ambient_proxy_c", ambient)
    if ambient is not None:
        add_if_finite(row, "fusion_ambient_proxy_f", c_to_f(ambient))
    add_if_finite(row, "fusion_scene_low_proxy_c", scene_low)
    add_if_finite(row, "fusion_scene_hot_proxy_c", scene_hot)
    if ambient is not None and scene_hot is not None:
        add_if_finite(row, "fusion_scene_hot_minus_ambient_c", scene_hot - ambient)

    for region in DETECTED_REGIONS + ARTICLE_REGIONS:
        for stat in HOT_STATS:
            observed = get_number(row, f"{region}_{stat}")
            if observed is None:
                continue
            if ambient is not None:
                add_if_finite(row, f"fusion_balance_{region}_{stat}_minus_ambient_c", observed - ambient)
                add_if_finite(row, f"fusion_balance_{region}_{stat}_ambient_ratio", ratio_delta(observed, ambient))
                add_if_finite(row, f"fusion_balance_{region}_{stat}_radiance_delta", radiance_delta(observed, ambient))
                added += 3
            if scene_low is not None:
                add_if_finite(row, f"fusion_balance_{region}_{stat}_minus_scene_low_c", observed - scene_low)
                added += 1
    return added


def add_pair_features(row: dict[str, object]) -> int:
    added = 0
    for stat in PAIR_STATS:
        detected_eye = average(
            [
                get_number(row, f"detected_left_eye_{stat}"),
                get_number(row, f"detected_right_eye_{stat}"),
            ]
        )
        detected_nostril = average(
            [
                get_number(row, f"detected_left_nostril_{stat}"),
                get_number(row, f"detected_right_nostril_{stat}"),
                get_number(row, f"detected_nostrils_box_{stat}"),
            ]
        )
        article_eye = get_number(row, f"article_eyes_{stat}")
        article_nostril = get_number(row, f"article_nostrils_{stat}")
        article_forehead = get_number(row, f"article_forehead_{stat}")
        article_ears = get_number(row, f"article_ears_{stat}")
        detected_face = get_number(row, f"detected_face_bbox_{stat}")
        detected_lower = get_number(row, f"detected_lower_face_{stat}")
        detected_muzzle = get_number(row, f"detected_muzzle_{stat}")

        pairs = {
            f"fusion_gradient_detected_nostril_minus_eye_{stat}_c": (
                detected_nostril,
                detected_eye,
            ),
            f"fusion_gradient_article_nostril_minus_eye_{stat}_c": (
                article_nostril,
                article_eye,
            ),
            f"fusion_gradient_article_nostril_minus_forehead_{stat}_c": (
                article_nostril,
                article_forehead,
            ),
            f"fusion_gradient_article_eye_minus_forehead_{stat}_c": (
                article_eye,
                article_forehead,
            ),
            f"fusion_gradient_article_forehead_minus_ears_{stat}_c": (
                article_forehead,
                article_ears,
            ),
            f"fusion_gradient_detected_lower_minus_face_{stat}_c": (
                detected_lower,
                detected_face,
            ),
            f"fusion_gradient_detected_muzzle_minus_face_{stat}_c": (
                detected_muzzle,
                detected_face,
            ),
            f"fusion_cross_article_nostril_minus_detected_nostril_{stat}_c": (
                article_nostril,
                detected_nostril,
            ),
            f"fusion_cross_article_eye_minus_detected_eye_{stat}_c": (
                article_eye,
                detected_eye,
            ),
        }
        for name, (left, right) in pairs.items():
            if left is None or right is None:
                continue
            add_if_finite(row, name, left - right)
            added += 1

    ambient = ambient_value(row)
    hot_proxy = average(
        [
            get_number(row, "article_nostrils_top5_mean_mean"),
            get_number(row, "article_nostrils_p95_mean"),
            get_number(row, "detected_nostrils_box_top5_mean_mean"),
            get_number(row, "detected_left_nostril_top5_mean_mean"),
            get_number(row, "detected_right_nostril_top5_mean_mean"),
        ]
    )
    surface_proxy = average(
        [
            get_number(row, "article_forehead_mean_mean"),
            get_number(row, "article_eyes_mean_mean"),
            get_number(row, "detected_face_bbox_mean_mean"),
            get_number(row, "detected_lower_face_mean_mean"),
        ]
    )
    add_if_finite(row, "fusion_internal_hot_proxy_c", hot_proxy)
    add_if_finite(row, "fusion_surface_proxy_c", surface_proxy)
    if hot_proxy is not None:
        add_if_finite(row, "fusion_internal_hot_proxy_f", c_to_f(hot_proxy))
    if hot_proxy is not None and ambient is not None:
        delta = hot_proxy - ambient
        add_if_finite(row, "fusion_internal_hot_minus_ambient_c", delta)
        add_if_finite(row, "fusion_internal_hot_ambient_ratio", ratio_delta(hot_proxy, ambient))
        add_if_finite(row, "fusion_internal_hot_radiance_delta", radiance_delta(hot_proxy, ambient))
        for alpha in (0.10, 0.20, 0.35, 0.50):
            tag = str(alpha).replace(".", "p")
            corrected = hot_proxy + alpha * delta
            add_if_finite(row, f"fusion_internal_newton_alpha_{tag}_c", corrected)
            add_if_finite(row, f"fusion_internal_newton_alpha_{tag}_f", c_to_f(corrected))
    if hot_proxy is not None and surface_proxy is not None:
        add_if_finite(row, "fusion_internal_hot_minus_surface_c", hot_proxy - surface_proxy)
    return added


def merge_rows(
    detected_rows: dict[tuple[str, str], dict[str, object]],
    article_rows: dict[tuple[str, str], dict[str, object]],
    include_count_features: bool,
    add_balance: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    shared_keys = sorted(set(detected_rows) & set(article_rows))
    if not shared_keys:
        raise RuntimeError("No shared date/sequence rows between detected and article feature files.")

    output_rows = []
    engineered_counts = []
    for key in shared_keys:
        detected = detected_rows[key]
        article = article_rows[key]
        check_label_match(key, detected, article)
        output = {
            "date": detected["date"],
            "sequence_num": detected["sequence_num"],
            "cow_tag": detected.get("cow_tag", ""),
            "temperature_f": detected.get("temperature_f", math.nan),
            "raw_frame_count": detected.get("raw_frame_count", math.nan),
            "sampled_frame_count": detected.get("sampled_frame_count", math.nan),
            "detected_frame_count": detected.get("detected_frame_count", math.nan),
        }
        add_source_features(output, detected, "detected", include_count_features)
        add_source_features(output, article, "article", include_count_features)
        engineered = 0
        if add_balance:
            engineered += add_region_balance_features(output)
            engineered += add_pair_features(output)
        engineered_counts.append(engineered)
        output_rows.append(output)

    summary = {
        "rows": len(output_rows),
        "detected_rows": len(detected_rows),
        "article_rows": len(article_rows),
        "shared_rows": len(shared_keys),
        "detected_only_rows": len(set(detected_rows) - set(article_rows)),
        "article_only_rows": len(set(article_rows) - set(detected_rows)),
        "include_count_features": include_count_features,
        "add_balance": add_balance,
        "engineered_feature_count_min": min(engineered_counts) if engineered_counts else 0,
        "engineered_feature_count_max": max(engineered_counts) if engineered_counts else 0,
        "engineered_feature_count_mean": (
            sum(engineered_counts) / len(engineered_counts) if engineered_counts else 0
        ),
    }
    return output_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge detected ROI and article-Otsu ROI sequence features for fusion modeling."
    )
    parser.add_argument(
        "--detected-features",
        default="data/temperature_outputs/detected_roi_filtered_80_v1/features.csv",
        type=Path,
    )
    parser.add_argument(
        "--article-features",
        default="data/temperature_outputs/article_otsu_roi_v1/features.csv",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        default="data/temperature_outputs/detected_article_otsu_fusion_v1",
        type=Path,
    )
    parser.add_argument("--output-name", default="features.csv")
    parser.add_argument("--include-count-features", action="store_true")
    parser.add_argument("--no-balance", dest="add_balance", action="store_false")
    parser.set_defaults(add_balance=True)
    args = parser.parse_args()

    detected_rows = read_rows(args.detected_features)
    article_rows = read_rows(args.article_features)
    rows, summary = merge_rows(
        detected_rows,
        article_rows,
        include_count_features=args.include_count_features,
        add_balance=args.add_balance,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name
    write_csv(output_path, rows)
    summary.update(
        {
            "detected_features": str(args.detected_features),
            "article_features": str(args.article_features),
            "output": str(output_path),
            "feature_count": len(rows[0]) - len(ID_COLUMNS),
        }
    )
    with (args.output_dir / "merge_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", output_path)
    print("Rows:", summary["rows"])
    print("Feature count:", summary["feature_count"])
    print("Engineered balance features per row:", int(summary["engineered_feature_count_mean"]))


if __name__ == "__main__":
    main()
