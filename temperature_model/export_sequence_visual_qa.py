from __future__ import annotations

import argparse
import csv
import math
import sys
import zipfile
from io import BytesIO
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "python38"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROI_COLORS = {
    "left_eye": (60, 220, 90),
    "right_eye": (60, 220, 90),
    "left_nostril": (70, 190, 255),
    "right_nostril": (70, 190, 255),
    "forehead": (255, 215, 70),
    "left_ear": (235, 90, 230),
    "right_ear": (235, 90, 230),
}


def read_detection_rows(path: Path, sequence: str) -> list[dict[str, str]]:
    date, sequence_num = sequence.split("/")
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["date"] == date and row["sequence_num"] == sequence_num:
                rows.append(row)
    rows.sort(key=lambda row: int(float(row["frame_id"])))
    return rows


def zip_name(date: str, sequence_num: str, frame_id: int) -> str:
    return f"thermal_raw/{date}/{sequence_num}_Video_Frame_{frame_id}.tiff"


def read_tiff_array(zf: zipfile.ZipFile, name: str) -> np.ndarray | None:
    with zf.open(name) as f:
        data = f.read()
    try:
        image = Image.open(BytesIO(data))
        array = np.asarray(image, dtype=np.float32)
    except Exception:
        return None
    if array.ndim == 3:
        array = array[:, :, 0]
    return array


def normalize_absolute(array: np.ndarray, thermal_min: float, thermal_max: float) -> np.ndarray:
    return np.clip((array - thermal_min) / max(thermal_max - thermal_min, 1e-6), 0, 1)


def normalize_percentile(array: np.ndarray) -> np.ndarray:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    low, high = np.percentile(finite, [1, 99])
    return np.clip((array - low) / max(high - low, 1e-6), 0, 1)


def heatmap_image(scaled: np.ndarray) -> Image.Image:
    x = np.clip(scaled, 0, 1)
    red = np.clip(3.0 * x - 0.6, 0, 1)
    green = np.clip(3.0 * x - 1.1, 0, 1)
    blue = np.clip(1.5 - 3.0 * x, 0, 1)
    rgb = np.stack([red, green, blue], axis=2)
    return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")


def row_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def roi_rect(row: dict[str, str], roi_name: str) -> tuple[float, float, float, float] | None:
    keys = [
        f"article_{roi_name}_x0",
        f"article_{roi_name}_y0",
        f"article_{roi_name}_x1",
        f"article_{roi_name}_y1",
    ]
    values = [row_float(row, key) for key in keys]
    if not all(math.isfinite(value) for value in values):
        return None
    return tuple(values)


def draw_rois(
    draw: ImageDraw.ImageDraw,
    row: dict[str, str],
    scale_x: float,
    scale_y: float,
    offset_x: int = 0,
    offset_y: int = 0,
) -> None:
    for roi_name, color in ROI_COLORS.items():
        rect = roi_rect(row, roi_name)
        if rect is None:
            continue
        x0, y0, x1, y1 = rect
        box = [
            offset_x + x0 * scale_x,
            offset_y + y0 * scale_y,
            offset_x + x1 * scale_x,
            offset_y + y1 * scale_y,
        ]
        draw.rectangle(box, outline=color, width=2)


def select_rows(rows: list[dict[str, str]], frames: int) -> list[dict[str, str]]:
    if len(rows) <= frames:
        return rows
    half = max(1, frames // 2)
    top = sorted(rows, key=lambda row: row_float(row, "frontal_score"), reverse=True)[:half]
    low = sorted(rows, key=lambda row: row_float(row, "frontal_score"))[: frames - half]
    selected = {int(float(row["frame_id"])): row for row in top + low}
    return [selected[key] for key in sorted(selected)]


def make_tile(array: np.ndarray, row: dict[str, str], tile_size: int, font: ImageFont.ImageFont) -> Image.Image:
    height, width = array.shape
    abs_img = heatmap_image(normalize_absolute(array, 15.0, 45.0)).resize((tile_size, tile_size))
    pct_img = heatmap_image(normalize_percentile(array)).resize((tile_size, tile_size))
    tile = Image.new("RGB", (tile_size * 2, tile_size + 54), (20, 24, 26))
    tile.paste(abs_img, (0, 28))
    tile.paste(pct_img, (tile_size, 28))
    draw = ImageDraw.Draw(tile)
    sx = tile_size / width
    sy = tile_size / height
    draw_rois(draw, row, sx, sy, offset_y=28)
    draw_rois(draw, row, sx, sy, offset_x=tile_size, offset_y=28)
    frame_id = int(float(row["frame_id"]))
    text = (
        f"f={frame_id} det={row_float(row, 'detection_score'):.2f} "
        f"front={row_float(row, 'frontal_score'):.2f} area={row_float(row, 'face_area_frac'):.2f}"
    )
    draw.text((6, 6), text, fill=(245, 245, 245), font=font)
    draw.text((6, tile_size + 32), "absolute 15-45C", fill=(210, 210, 210), font=font)
    draw.text((tile_size + 6, tile_size + 32), "per-frame percentile", fill=(210, 210, 210), font=font)
    return tile


def summarize_rows(rows: list[dict[str, str]], sequence: str) -> dict[str, object]:
    frontals = np.asarray([row_float(row, "frontal_score") for row in rows], dtype=np.float32)
    detections = np.asarray([row_float(row, "detection_score") for row in rows], dtype=np.float32)
    first = rows[0]
    return {
        "sequence": sequence,
        "cow_tag": first.get("cow_tag", ""),
        "temperature_f": first.get("temperature_f", ""),
        "accepted_frames": len(rows),
        "frontal_mean": float(np.nanmean(frontals)),
        "frontal_min": float(np.nanmin(frontals)),
        "frontal_max": float(np.nanmax(frontals)),
        "detection_mean": float(np.nanmean(detections)),
        "first_frame": int(float(rows[0]["frame_id"])),
        "last_frame": int(float(rows[-1]["frame_id"])),
    }


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export visual QA contact sheets for accepted article-Otsu frames.")
    parser.add_argument("--raw-zip", default="data/thermal_raw.zip", type=Path)
    parser.add_argument("--frame-detections", default="data/temperature_outputs/article_otsu_roi_v1/frame_detections.csv", type=Path)
    parser.add_argument("--sequence", action="append", required=True)
    parser.add_argument("--output-dir", default="data/temperature_outputs/visual_sequence_qa_v1", type=Path)
    parser.add_argument("--frames", default=24, type=int)
    parser.add_argument("--tile-size", default=170, type=int)
    parser.add_argument("--columns", default=3, type=int)
    args = parser.parse_args()

    font = ImageFont.load_default()
    summaries = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.raw_zip) as zf:
        for sequence in args.sequence:
            rows = read_detection_rows(args.frame_detections, sequence)
            if not rows:
                raise RuntimeError(f"No detection rows found for {sequence}")
            summaries.append(summarize_rows(rows, sequence))
            selected = select_rows(rows, args.frames)
            tiles = []
            for row in selected:
                frame_id = int(float(row["frame_id"]))
                name = zip_name(row["date"], row["sequence_num"], frame_id)
                array = read_tiff_array(zf, name)
                if array is None:
                    continue
                tiles.append(make_tile(array, row, args.tile_size, font))
            if not tiles:
                raise RuntimeError(f"No readable selected frames for {sequence}")

            tile_w, tile_h = tiles[0].size
            columns = min(args.columns, len(tiles))
            rows_count = math.ceil(len(tiles) / columns)
            sheet = Image.new("RGB", (columns * tile_w, rows_count * tile_h + 34), (14, 18, 20))
            draw = ImageDraw.Draw(sheet)
            title = f"{sequence} cow={summaries[-1]['cow_tag']} temp={summaries[-1]['temperature_f']}F"
            draw.text((8, 10), title, fill=(245, 245, 245), font=font)
            for index, tile in enumerate(tiles):
                col = index % columns
                row_index = index // columns
                sheet.paste(tile, (col * tile_w, row_index * tile_h + 34))
            out_name = sequence.replace("/", "_") + "_visual_qa.png"
            sheet.save(args.output_dir / out_name)
            print("Saved:", args.output_dir / out_name)

    write_summary(args.output_dir / "sequence_visual_qa_summary.csv", summaries)
    print("Saved:", args.output_dir / "sequence_visual_qa_summary.csv")


if __name__ == "__main__":
    main()
