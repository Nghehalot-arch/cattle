from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common import aggregate_feature_rows, read_feature_csv, write_csv
from merge_roi_feature_sets import merge_rows, read_rows


ID_KEYS = {"date", "sequence_num", "cow_tag", "temperature_f", "frame_id"}
COUNT_KEYS = {"raw_frame_count", "sampled_frame_count", "detected_frame_count"}


def as_float(value: object, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def high_score(value: float, low: float, high: float) -> float:
    return clamp01((value - low) / max(high - low, 1e-6))


def low_score(value: float, low: float, high: float) -> float:
    return clamp01((high - value) / max(high - low, 1e-6))


def band_score(value: float, outer_low: float, ideal_low: float, ideal_high: float, outer_high: float) -> float:
    if not math.isfinite(value) or value <= outer_low or value >= outer_high:
        return 0.0
    if ideal_low <= value <= ideal_high:
        return 1.0
    if value < ideal_low:
        return high_score(value, outer_low, ideal_low)
    return low_score(value, ideal_high, outer_high)


def get_quality(row: dict[str, object], name: str) -> float:
    return as_float(row.get(f"quality_{name}", row.get(name)))


def get_any(row: dict[str, object], *names: str) -> float:
    for name in names:
        value = as_float(row.get(name))
        if math.isfinite(value):
            return value
    return math.nan


def compute_score(row: dict[str, object], args: argparse.Namespace) -> tuple[float, dict[str, float]]:
    detection = get_quality(row, "detection_score")
    frontal = get_quality(row, "frontal_score")
    face_area = get_quality(row, "face_area_frac")
    face_aspect = get_quality(row, "face_aspect")
    center_x = abs(get_quality(row, "center_offset_x"))
    center_y = abs(get_quality(row, "center_offset_y"))
    eye_y = get_quality(row, "eye_y_diff_norm")
    nostril_y = get_quality(row, "nostril_y_diff_norm")
    muzzle_center = get_quality(row, "muzzle_center_offset")
    muzzle_symmetry = get_quality(row, "muzzle_symmetry")
    lower_order = get_quality(row, "lower_face_order_ok")
    kp_mean = get_quality(row, "required_kp_score_mean")

    eye_fraction = get_any(row, "article_eyes_otsu_fraction")
    nostril_fraction = get_any(row, "article_nostrils_otsu_fraction")
    all_fraction = get_any(row, "article_all_otsu_fraction")
    eye_pixels = get_any(row, "article_eyes_pixel_count", "article_eyes_patch_pixel_count")
    nostril_pixels = get_any(row, "article_nostrils_pixel_count", "article_nostrils_patch_pixel_count")

    components = {
        "component_detection": high_score(detection, args.detection_floor, 0.85),
        "component_frontal": high_score(frontal, args.frontal_floor, args.frontal_good),
        "component_face_area": band_score(face_area, 0.04, args.face_area_min, args.face_area_max, 0.65),
        "component_face_aspect": band_score(face_aspect, 0.70, 1.25, 2.45, 3.40),
        "component_center": 0.5
        * (
            low_score(center_x, args.center_x_good, args.center_x_max)
            + low_score(center_y, args.center_y_good, args.center_y_max)
        ),
        "component_symmetry": (
            low_score(eye_y, 0.03, args.eye_y_max)
            + low_score(nostril_y, 0.04, args.nostril_y_max)
            + low_score(muzzle_center, 0.10, args.muzzle_center_max)
            + low_score(muzzle_symmetry, 0.12, args.muzzle_symmetry_max)
        )
        / 4.0,
        "component_lower_order": 1.0 if lower_order >= 1.0 else 0.0,
        "component_keypoints": high_score(kp_mean, 0.002, 0.012),
        "component_otsu_fraction": (
            band_score(eye_fraction, 0.0, 0.03, 0.75, 0.98)
            + band_score(nostril_fraction, 0.0, 0.03, 0.75, 0.98)
            + band_score(all_fraction, 0.0, 0.03, 0.80, 0.99)
        )
        / 3.0,
        "component_roi_pixels": 0.5
        * (
            high_score(eye_pixels, args.min_eye_pixels, args.min_eye_pixels * 4.0)
            + high_score(nostril_pixels, args.min_nostril_pixels, args.min_nostril_pixels * 4.0)
        ),
    }
    weights = {
        "component_detection": 0.12,
        "component_frontal": 0.22,
        "component_face_area": 0.10,
        "component_face_aspect": 0.06,
        "component_center": 0.10,
        "component_symmetry": 0.18,
        "component_lower_order": 0.06,
        "component_keypoints": 0.04,
        "component_otsu_fraction": 0.08,
        "component_roi_pixels": 0.04,
    }
    score = sum(components[name] * weight for name, weight in weights.items())
    return float(score), components


def rejection_reasons(row: dict[str, object], score: float, args: argparse.Namespace) -> list[str]:
    reasons = []
    checks = [
        ("low_score", score >= args.min_score),
        ("low_detection", get_quality(row, "detection_score") >= args.min_detection_score),
        ("low_frontal", get_quality(row, "frontal_score") >= args.min_frontal_score),
        ("face_area_small", get_quality(row, "face_area_frac") >= args.face_area_min),
        ("face_area_large", get_quality(row, "face_area_frac") <= args.face_area_max),
        ("face_aspect_bad", args.face_aspect_min <= get_quality(row, "face_aspect") <= args.face_aspect_max),
        ("center_x_bad", abs(get_quality(row, "center_offset_x")) <= args.center_x_max),
        ("center_y_bad", abs(get_quality(row, "center_offset_y")) <= args.center_y_max),
        ("eye_y_bad", get_quality(row, "eye_y_diff_norm") <= args.eye_y_max),
        ("nostril_y_bad", get_quality(row, "nostril_y_diff_norm") <= args.nostril_y_max),
        ("muzzle_center_bad", get_quality(row, "muzzle_center_offset") <= args.muzzle_center_max),
        ("muzzle_symmetry_bad", get_quality(row, "muzzle_symmetry") <= args.muzzle_symmetry_max),
        ("eye_roi_too_small", get_any(row, "article_eyes_pixel_count", "article_eyes_patch_pixel_count") >= args.min_eye_pixels),
        (
            "nostril_roi_too_small",
            get_any(row, "article_nostrils_pixel_count", "article_nostrils_patch_pixel_count")
            >= args.min_nostril_pixels,
        ),
        ("eye_otsu_bad", get_any(row, "article_eyes_otsu_fraction") <= args.max_otsu_fraction),
        ("nostril_otsu_bad", get_any(row, "article_nostrils_otsu_fraction") <= args.max_otsu_fraction),
    ]
    if args.require_lower_face_order:
        checks.append(("bad_lower_face_order", get_quality(row, "lower_face_order_ok") >= 1.0))
    for name, passed in checks:
        if not bool(passed):
            reasons.append(name)
    return reasons


def frame_key(row: dict[str, object]) -> tuple[str, str, int]:
    return (str(row["date"]), str(row["sequence_num"]), int(as_float(row["frame_id"])))


def feature_only(row: dict[str, object]) -> dict[str, float]:
    return {
        key: value
        for key, value in row.items()
        if key not in ID_KEYS and key not in COUNT_KEYS and isinstance(value, (float, int, np.floating))
    }


def build_sequence_rows(rows: list[dict[str, object]], prefix: str) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["date"]), str(row["sequence_num"]))].append(row)

    output_rows = []
    for group_rows in grouped.values():
        group_rows = sorted(group_rows, key=lambda item: int(as_float(item["frame_id"])))
        first = group_rows[0]
        output = {
            "date": first["date"],
            "sequence_num": first["sequence_num"],
            "cow_tag": first["cow_tag"],
            "temperature_f": first["temperature_f"],
            "detected_frame_count": len(group_rows),
        }
        output.update(aggregate_feature_rows([feature_only(row) for row in group_rows]))
        output_rows.append(output)
    return sorted(output_rows, key=lambda item: (str(item["date"]), str(item["sequence_num"])))


def parse_frame_rows(path: Path) -> list[dict[str, object]]:
    return read_feature_csv(path)


def index_by_frame(rows: list[dict[str, object]]) -> dict[tuple[str, str, int], dict[str, object]]:
    return {frame_key(row): row for row in rows}


def add_gate_columns(row: dict[str, object], score: float, components: dict[str, float], accepted: bool, reasons: list[str]) -> dict[str, object]:
    output = dict(row)
    output["quality_gate_v2_score"] = score
    output["quality_gate_v2_accept"] = 1.0 if accepted else 0.0
    output["quality_gate_v2_reasons"] = ";".join(reasons)
    for name, value in components.items():
        output[f"quality_gate_v2_{name}"] = value
    return output


def select_accepted_frames(rows: list[dict[str, object]], args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    audited_rows = []
    for row in rows:
        score, components = compute_score(row, args)
        reasons = rejection_reasons(row, score, args)
        audited_rows.append(add_gate_columns(row, score, components, not reasons, reasons))

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    rejected_rows = []
    for row in audited_rows:
        if row["quality_gate_v2_accept"] >= 1.0:
            grouped[(str(row["date"]), str(row["sequence_num"]))].append(row)
        else:
            rejected_rows.append(row)

    accepted_rows = []
    fallback_rows = []
    for key, group_rows in grouped.items():
        group_rows.sort(
            key=lambda item: (
                as_float(item["quality_gate_v2_score"]),
                as_float(item.get("quality_frontal_score", item.get("frontal_score"))),
                as_float(item.get("quality_detection_score", item.get("detection_score"))),
            ),
            reverse=True,
        )
        if args.top_frames_per_sequence:
            group_rows = group_rows[: args.top_frames_per_sequence]
        accepted_rows.extend(group_rows)

    if args.min_frames_per_sequence > 0:
        accepted_group_keys = {(str(row["date"]), str(row["sequence_num"])) for row in accepted_rows}
        all_group_keys = {(str(row["date"]), str(row["sequence_num"])) for row in audited_rows}
        missing_keys = sorted(all_group_keys - accepted_group_keys)
        for key in missing_keys:
            candidates = [
                row for row in audited_rows if (str(row["date"]), str(row["sequence_num"])) == key
            ]
            candidates.sort(
                key=lambda item: (
                    as_float(item["quality_gate_v2_score"]),
                    as_float(item.get("quality_frontal_score", item.get("frontal_score"))),
                    as_float(item.get("quality_detection_score", item.get("detection_score"))),
                ),
                reverse=True,
            )
            fallback = candidates[: args.min_frames_per_sequence]
            for row in fallback:
                row["quality_gate_v2_accept"] = 1.0
                row["quality_gate_v2_reasons"] = "fallback_best_available"
            accepted_rows.extend(fallback)
            fallback_rows.extend(fallback)

    accepted_rows.sort(key=lambda item: (str(item["date"]), str(item["sequence_num"]), int(as_float(item["frame_id"]))))
    return audited_rows, accepted_rows


def read_sequence_rows(path: Path) -> dict[tuple[str, str], dict[str, object]]:
    return read_rows(path)


def write_summary(
    path: Path,
    audited_rows: list[dict[str, object]],
    accepted_rows: list[dict[str, object]],
    merged_rows: list[dict[str, object]],
    args: argparse.Namespace,
) -> None:
    reject_reasons = Counter()
    for row in audited_rows:
        reasons = str(row.get("quality_gate_v2_reasons", ""))
        if not reasons:
            continue
        for reason in reasons.split(";"):
            if reason:
                reject_reasons[reason] += 1

    by_sequence = defaultdict(lambda: {"input": 0, "accepted": 0})
    for row in audited_rows:
        by_sequence[(str(row["date"]), str(row["sequence_num"]))]["input"] += 1
    for row in accepted_rows:
        by_sequence[(str(row["date"]), str(row["sequence_num"]))]["accepted"] += 1

    accepted_scores = [as_float(row.get("quality_gate_v2_score")) for row in accepted_rows]
    accepted_scores = [score for score in accepted_scores if math.isfinite(score)]
    summary = {
        "input_frame_count": len(audited_rows),
        "accepted_frame_count": len(accepted_rows),
        "rejected_frame_count": len(audited_rows) - len(accepted_rows),
        "input_sequence_count": len(by_sequence),
        "accepted_sequence_count": len([item for item in by_sequence.values() if item["accepted"] > 0]),
        "merged_sequence_count": len(merged_rows),
        "accepted_score_mean": float(np.mean(accepted_scores)) if accepted_scores else math.nan,
        "accepted_score_min": float(np.min(accepted_scores)) if accepted_scores else math.nan,
        "accepted_score_max": float(np.max(accepted_scores)) if accepted_scores else math.nan,
        "thresholds": {
            key: value
            for key, value in vars(args).items()
            if isinstance(value, (int, float, str, bool)) or value is None
        },
        "reject_reasons": dict(reject_reasons.most_common()),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    sequence_rows = []
    for (date, sequence_num), counts in sorted(by_sequence.items()):
        sequence_rows.append(
            {
                "date": date,
                "sequence_num": sequence_num,
                "input_frames": counts["input"],
                "accepted_frames": counts["accepted"],
                "accepted_fraction": counts["accepted"] / max(counts["input"], 1),
            }
        )
    write_csv(path.with_name("sequence_frame_counts.csv"), sequence_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build quality_gate_v2 frame filters and rebuilt article/detected fusion features."
    )
    parser.add_argument("--article-frame-features", default="data/temperature_outputs/article_otsu_roi_v1/frame_features.csv", type=Path)
    parser.add_argument("--detected-frame-features", default="data/temperature_outputs/detected_roi_filtered_80_v1/frame_features.csv", type=Path)
    parser.add_argument("--output-dir", default="data/temperature_outputs/quality_gate_v2_default", type=Path)
    parser.add_argument("--min-score", default=0.42, type=float)
    parser.add_argument("--min-detection-score", default=0.50, type=float)
    parser.add_argument("--min-frontal-score", default=0.25, type=float)
    parser.add_argument("--detection-floor", default=0.45, type=float)
    parser.add_argument("--frontal-floor", default=0.18, type=float)
    parser.add_argument("--frontal-good", default=0.55, type=float)
    parser.add_argument("--face-area-min", default=0.10, type=float)
    parser.add_argument("--face-area-max", default=0.45, type=float)
    parser.add_argument("--face-aspect-min", default=0.95, type=float)
    parser.add_argument("--face-aspect-max", default=3.05, type=float)
    parser.add_argument("--center-x-good", default=0.08, type=float)
    parser.add_argument("--center-x-max", default=0.34, type=float)
    parser.add_argument("--center-y-good", default=0.05, type=float)
    parser.add_argument("--center-y-max", default=0.24, type=float)
    parser.add_argument("--eye-y-max", default=0.24, type=float)
    parser.add_argument("--nostril-y-max", default=0.28, type=float)
    parser.add_argument("--muzzle-center-max", default=0.46, type=float)
    parser.add_argument("--muzzle-symmetry-max", default=0.58, type=float)
    parser.add_argument("--min-eye-pixels", default=8.0, type=float)
    parser.add_argument("--min-nostril-pixels", default=8.0, type=float)
    parser.add_argument("--max-otsu-fraction", default=0.98, type=float)
    parser.add_argument("--require-lower-face-order", action="store_true")
    parser.add_argument("--top-frames-per-sequence", type=int)
    parser.add_argument("--min-frames-per-sequence", default=1, type=int)
    args = parser.parse_args()

    article_rows = parse_frame_rows(args.article_frame_features)
    detected_rows = parse_frame_rows(args.detected_frame_features)
    detected_by_key = index_by_frame(detected_rows)

    audited_rows, accepted_article_rows = select_accepted_frames(article_rows, args)
    accepted_keys = {frame_key(row) for row in accepted_article_rows}
    accepted_detected_rows = [
        detected_by_key[key] for key in sorted(accepted_keys) if key in detected_by_key
    ]

    article_sequence_rows = build_sequence_rows(accepted_article_rows, "article")
    detected_sequence_rows = build_sequence_rows(accepted_detected_rows, "detected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "frame_quality_all.csv", audited_rows)
    write_csv(args.output_dir / "frame_quality_accepted.csv", accepted_article_rows)
    rejected_rows = [row for row in audited_rows if row.get("quality_gate_v2_accept", 0.0) < 1.0]
    if rejected_rows:
        write_csv(args.output_dir / "frame_quality_rejected.csv", rejected_rows)
    write_csv(args.output_dir / "article_frame_features.csv", accepted_article_rows)
    write_csv(args.output_dir / "detected_frame_features.csv", accepted_detected_rows)
    write_csv(args.output_dir / "article_features.csv", article_sequence_rows)
    write_csv(args.output_dir / "detected_features.csv", detected_sequence_rows)

    detected_sequence_by_key = read_sequence_rows(args.output_dir / "detected_features.csv")
    article_sequence_by_key = read_sequence_rows(args.output_dir / "article_features.csv")
    merged_rows, merge_summary = merge_rows(
        detected_sequence_by_key,
        article_sequence_by_key,
        include_count_features=False,
        add_balance=True,
    )
    write_csv(args.output_dir / "features.csv", merged_rows)
    with (args.output_dir / "merge_summary.json").open("w", encoding="utf-8") as f:
        json.dump(merge_summary, f, indent=2)
    write_summary(args.output_dir / "quality_gate_summary.json", audited_rows, accepted_article_rows, merged_rows, args)

    print("Saved:", args.output_dir)
    print("Input frames:", len(audited_rows))
    print("Accepted frames:", len(accepted_article_rows))
    print("Merged sequences:", len(merged_rows))


if __name__ == "__main__":
    main()
