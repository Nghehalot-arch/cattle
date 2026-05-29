from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import aggregate_feature_rows, read_feature_csv, write_csv


ID_KEYS = {"date", "sequence_num", "cow_tag", "temperature_f", "frame_id"}


def passes(row: dict[str, object], args: argparse.Namespace) -> bool:
    checks = [
        row.get("quality_frontal_score", 0.0) >= args.min_frontal_score,
        row.get("quality_eye_y_diff_norm", 1.0) <= args.max_eye_y_diff,
        row.get("quality_nostril_y_diff_norm", 1.0) <= args.max_nostril_y_diff,
        row.get("quality_muzzle_center_offset", 1.0) <= args.max_muzzle_center_offset,
        row.get("quality_muzzle_symmetry", 1.0) <= args.max_muzzle_symmetry,
        row.get("quality_required_kp_score_min", 0.0) >= args.min_required_keypoint_score,
        row.get("quality_detection_score", 0.0) >= args.min_detection_score,
    ]
    if args.require_lower_face_order:
        checks.append(row.get("quality_lower_face_order_ok", 0.0) >= 1.0)
    return all(bool(check) for check in checks)


def feature_only(row: dict[str, object]) -> dict[str, float]:
    return {
        key: value
        for key, value in row.items()
        if key not in ID_KEYS and isinstance(value, (float, int, np.floating))
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter saved ROI frame features by quality and rebuild sequence aggregates."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-frontal-score", default=0.25, type=float)
    parser.add_argument("--max-eye-y-diff", default=0.20, type=float)
    parser.add_argument("--max-nostril-y-diff", default=0.20, type=float)
    parser.add_argument("--max-muzzle-center-offset", default=0.35, type=float)
    parser.add_argument("--max-muzzle-symmetry", default=0.55, type=float)
    parser.add_argument("--min-required-keypoint-score", default=0.0, type=float)
    parser.add_argument("--min-detection-score", default=0.0, type=float)
    parser.add_argument("--require-lower-face-order", action="store_true")
    parser.add_argument(
        "--top-frames-per-sequence",
        default=None,
        type=int,
        help="Keep only the highest-frontal-score frames per date/sequence.",
    )
    args = parser.parse_args()

    rows = [row for row in read_feature_csv(args.input) if passes(row, args)]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["date"], row["sequence_num"])].append(row)

    filtered_rows = []
    for group_rows in grouped.values():
        group_rows.sort(
            key=lambda row: (
                row.get("quality_frontal_score", 0.0),
                row.get("quality_detection_score", 0.0),
                -row.get("quality_muzzle_symmetry", 1.0),
            ),
            reverse=True,
        )
        if args.top_frames_per_sequence:
            group_rows = group_rows[: args.top_frames_per_sequence]
        filtered_rows.extend(group_rows)

    sequence_rows = []
    grouped = defaultdict(list)
    for row in filtered_rows:
        grouped[(row["date"], row["sequence_num"])].append(row)

    for group_rows in grouped.values():
        first = group_rows[0]
        record = {
            "date": first["date"],
            "sequence_num": first["sequence_num"],
            "cow_tag": first["cow_tag"],
            "temperature_f": first["temperature_f"],
            "detected_frame_count": len(group_rows),
        }
        record.update(aggregate_feature_rows([feature_only(row) for row in group_rows]))
        sequence_rows.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "frame_features.csv", filtered_rows)
    write_csv(args.output_dir / "features.csv", sequence_rows)
    write_csv(
        args.output_dir / "summary.csv",
        [
            {
                "input_frames": len(read_feature_csv(args.input)),
                "kept_frames": len(filtered_rows),
                "sequence_count": len(sequence_rows),
                "top_frames_per_sequence": args.top_frames_per_sequence or "",
            }
        ],
    )
    print("Saved:", args.output_dir)
    print("Kept frames:", len(filtered_rows))
    print("Sequences:", len(sequence_rows))


if __name__ == "__main__":
    main()
