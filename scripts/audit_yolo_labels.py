from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit YOLO labels and create box preview images.")
    parser.add_argument(
        "--bruise-root",
        type=Path,
        default=Path(r"R:\Dataset2\Bruisess.yolov8"),
        help="YOLO folder for bruise images.",
    )
    parser.add_argument(
        "--non-bruise-root",
        type=Path,
        default=Path(r"R:\Dataset2\Non-Bruises.yolov8"),
        help="YOLO folder for non-bruise images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/label_audit"),
        help="Output folder for CSV reports and preview images.",
    )
    parser.add_argument("--max-previews", type=int, default=200, help="Maximum flagged previews to save.")
    return parser.parse_args()


def image_files(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing image folder: {images_dir}")
    return sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def parse_label(label_path: Path) -> tuple[list[dict], list[str]]:
    errors: list[str] = []
    boxes: list[dict] = []
    if not label_path.exists():
        return boxes, ["missing_label"]

    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for idx, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"line_{idx}_not_yolo_box")
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = [float(v) for v in parts[1:]]
        except ValueError:
            errors.append(f"line_{idx}_non_numeric")
            continue
        row_errors = []
        if cls != 0:
            row_errors.append(f"line_{idx}_unexpected_class_{cls}")
        if not 0 <= x <= 1:
            row_errors.append(f"line_{idx}_x_out_of_range")
        if not 0 <= y <= 1:
            row_errors.append(f"line_{idx}_y_out_of_range")
        if not 0 < w <= 1:
            row_errors.append(f"line_{idx}_w_out_of_range")
        if not 0 < h <= 1:
            row_errors.append(f"line_{idx}_h_out_of_range")
        errors.extend(row_errors)
        boxes.append({"class": cls, "x": x, "y": y, "w": w, "h": h, "area": w * h})
    return boxes, errors


def audit_split(root: Path, positive: bool) -> list[dict]:
    images_dir = root / "train" / "images"
    labels_dir = root / "train" / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError(f"Missing label folder: {labels_dir}")

    rows: list[dict] = []
    for img_path in image_files(images_dir):
        label_path = labels_dir / f"{img_path.stem}.txt"
        boxes, errors = parse_label(label_path)
        img = cv2.imread(str(img_path))
        if img is None:
            errors.append("unreadable_image")
            img_h = img_w = 0
        else:
            img_h, img_w = img.shape[:2]

        if positive and not boxes:
            errors.append("bruise_image_has_no_box")
        if not positive and boxes:
            errors.append("non_bruise_image_has_box")

        total_area = sum(float(box["area"]) for box in boxes)
        max_area = max([float(box["area"]) for box in boxes], default=0.0)
        min_area = min([float(box["area"]) for box in boxes], default=0.0)

        if positive:
            if max_area > 0.65:
                errors.append("very_large_box")
            if boxes and max_area < 0.0025:
                errors.append("very_small_box")
            if len(boxes) > 3:
                errors.append("many_boxes")

        rows.append(
            {
                "dataset": "bruise" if positive else "non_bruise",
                "image": str(img_path),
                "label": str(label_path),
                "img_w": img_w,
                "img_h": img_h,
                "box_count": len(boxes),
                "max_box_area_ratio": max_area,
                "min_box_area_ratio": min_area,
                "total_box_area_ratio": total_area,
                "flags": ";".join(sorted(set(errors))),
            }
        )
    return rows


def draw_preview(row: dict, output_dir: Path) -> Path | None:
    img_path = Path(row["image"])
    label_path = Path(row["label"])
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    boxes, _ = parse_label(label_path)
    img_h, img_w = img.shape[:2]
    for box in boxes:
        x = float(box["x"]) * img_w
        y = float(box["y"]) * img_h
        w = float(box["w"]) * img_w
        h = float(box["h"]) * img_h
        x1 = int(round(x - w / 2))
        y1 = int(round(y - h / 2))
        x2 = int(round(x + w / 2))
        y2 = int(round(y + h / 2))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

    label = f"{row['dataset']} | {row['flags'] or 'ok'}"
    cv2.putText(img, label[:120], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{row['dataset']}__{img_path.stem}.jpg"
    out_path = output_dir / safe_name
    cv2.imwrite(str(out_path), img)
    return out_path


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    rows = audit_split(args.bruise_root, positive=True)
    rows.extend(audit_split(args.non_bruise_root, positive=False))

    flagged = [row for row in rows if row["flags"]]
    write_csv(output_dir / "label_audit_all.csv", rows)
    write_csv(output_dir / "label_audit_flagged.csv", flagged)

    preview_dir = output_dir / "flagged_previews"
    saved = 0
    for row in flagged[: args.max_previews]:
        if draw_preview(row, preview_dir) is not None:
            saved += 1

    print(f"Images audited: {len(rows)}")
    print(f"Flagged images: {len(flagged)}")
    print(f"Flagged preview images saved: {saved}")
    print(f"All report: {output_dir / 'label_audit_all.csv'}")
    print(f"Flagged report: {output_dir / 'label_audit_flagged.csv'}")
    print(f"Preview folder: {preview_dir}")


if __name__ == "__main__":
    main()
