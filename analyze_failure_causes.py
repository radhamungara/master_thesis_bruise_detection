from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from ultralytics import RTDETR


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_FOLDS_ROOTS = [Path("artifacts/fold_datasets"), Path("runs/detect/artifacts/fold_datasets")]
DEFAULT_RUN_ROOTS = [Path("artifacts/runs/cross_validation"), Path("runs/detect/artifacts/runs/cross_validation")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze why bruise detection predictions fail.")
    parser.add_argument("--folds-root", type=Path, default=None, help="Path to fold_datasets folder")
    parser.add_argument("--run-root", type=Path, default=None, help="Path to cross_validation run folder")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reports/failure_analysis"))
    parser.add_argument("--conf", type=float, default=0.25, help="Prediction confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for a correct detection")
    return parser.parse_args()


def first_existing(candidates: list[Path], required_child: str) -> Path:
    for path in candidates:
        if (path / required_child).exists():
            return path
    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find required folder. Searched: {searched}")


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    folds_root = args.folds_root or first_existing(DEFAULT_FOLDS_ROOTS, "fold_1")
    run_root = args.run_root or first_existing(DEFAULT_RUN_ROOTS, "rtdetr_fold_1")
    return folds_root, run_root


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    rows = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        cls = int(float(parts[0]))
        values = [float(v) for v in parts[1:]]
        if len(values) == 4:
            x, y, w, h = values
            rows.append((cls, x, y, w, h))
        elif len(values) >= 6 and len(values) % 2 == 0:
            xs = values[0::2]
            ys = values[1::2]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            rows.append((cls, (x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min))
    return rows


def yolo_to_xyxy(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    cx, cy, bw, bh = x * img_w, y * img_h, w * img_w, h * img_h
    return cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def image_paths(folder: Path) -> list[Path]:
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])


def analyze_split(fold_id: int, split: str, fold_dir: Path) -> list[dict]:
    rows = []
    img_dir = fold_dir / "images" / split
    label_dir = fold_dir / "labels" / split

    for img_path in image_paths(img_dir):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        labels = read_yolo_labels(label_dir / f"{img_path.stem}.txt")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if not labels:
            rows.append(
                {
                    "fold": fold_id,
                    "split": split,
                    "image": str(img_path),
                    "img_w": img_w,
                    "img_h": img_h,
                    "brightness": float(np.mean(gray)),
                    "contrast": float(np.std(gray)),
                    "gt_count": 0,
                    "has_bruise": 0,
                    "avg_box_area_ratio": 0.0,
                    "max_box_area_ratio": 0.0,
                    "min_box_area_ratio": 0.0,
                    "possible_label_issue": "negative_or_missing_label",
                }
            )
            continue

        area_ratios = [w * h for _, _, _, w, h in labels]
        issue_flags = []
        if max(area_ratios) > 0.45:
            issue_flags.append("very_large_box")
        if min(area_ratios) < 0.002:
            issue_flags.append("very_small_box")
        if len(labels) > 5:
            issue_flags.append("many_boxes")

        rows.append(
            {
                "fold": fold_id,
                "split": split,
                "image": str(img_path),
                "img_w": img_w,
                "img_h": img_h,
                "brightness": float(np.mean(gray)),
                "contrast": float(np.std(gray)),
                "gt_count": len(labels),
                "has_bruise": 1,
                "avg_box_area_ratio": float(np.mean(area_ratios)),
                "max_box_area_ratio": float(np.max(area_ratios)),
                "min_box_area_ratio": float(np.min(area_ratios)),
                "possible_label_issue": ";".join(issue_flags),
            }
        )
    return rows


def evaluate_predictions(fold_id: int, fold_dir: Path, model_path: Path, conf: float, iou_threshold: float) -> list[dict]:
    rows = []
    val_img_dir = fold_dir / "images" / "val"
    val_label_dir = fold_dir / "labels" / "val"
    model = RTDETR(str(model_path))

    for img_path in image_paths(val_img_dir):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        labels = read_yolo_labels(val_label_dir / f"{img_path.stem}.txt")
        gt_boxes = [yolo_to_xyxy(x, y, w, h, img_w, img_h) for _, x, y, w, h in labels]

        pred = model.predict(source=str(img_path), imgsz=640, conf=0.001, verbose=False)[0]
        if pred.boxes is None or len(pred.boxes) == 0:
            pred_boxes = []
            pred_scores = []
        else:
            pred_boxes = [tuple(map(float, box)) for box in pred.boxes.xyxy.cpu().numpy().tolist()]
            pred_scores = [float(score) for score in pred.boxes.conf.cpu().numpy().tolist()]

        confident_preds = [(box, score) for box, score in zip(pred_boxes, pred_scores) if score >= conf]
        matched_gt = set()
        matched_pred = set()
        best_iou = 0.0

        for gt_idx, gt_box in enumerate(gt_boxes):
            for pred_idx, (pred_box, _) in enumerate(confident_preds):
                overlap = iou(gt_box, pred_box)
                best_iou = max(best_iou, overlap)
                if overlap >= iou_threshold:
                    matched_gt.add(gt_idx)
                    matched_pred.add(pred_idx)

        false_negative_count = max(0, len(gt_boxes) - len(matched_gt))
        false_positive_count = max(0, len(confident_preds) - len(matched_pred))
        true_positive_count = len(matched_gt)

        if gt_boxes and not confident_preds:
            failure_type = "missed_bruise_no_prediction"
        elif gt_boxes and false_negative_count and false_positive_count:
            failure_type = "wrong_location_and_missed_bruise"
        elif gt_boxes and false_negative_count:
            failure_type = "missed_bruise"
        elif not gt_boxes and false_positive_count:
            failure_type = "false_positive_on_no_bruise"
        elif gt_boxes and true_positive_count:
            failure_type = "correct_or_partial_detection"
        else:
            failure_type = "true_negative"

        rows.append(
            {
                "fold": fold_id,
                "image": str(img_path),
                "gt_count": len(gt_boxes),
                "pred_count_at_conf": len(confident_preds),
                "max_conf": max(pred_scores) if pred_scores else 0.0,
                "best_iou": best_iou,
                "tp": true_positive_count,
                "fp": false_positive_count,
                "fn": false_negative_count,
                "failure_type": failure_type,
            }
        )
    return rows


def build_recommendations(dataset_df: pd.DataFrame, pred_df: pd.DataFrame) -> list[str]:
    recommendations = []
    total_images = len(dataset_df)
    val_df = dataset_df[dataset_df["split"] == "val"]
    negative_ratio = float((dataset_df["has_bruise"] == 0).mean()) if total_images else 0.0
    small_box_ratio = float((dataset_df["max_box_area_ratio"] < 0.01).mean()) if total_images else 0.0
    large_box_ratio = float((dataset_df["max_box_area_ratio"] > 0.30).mean()) if total_images else 0.0

    if total_images < 500:
        recommendations.append("Dataset size appears limited. Add more original bruise cases before relying on augmentation.")
    if negative_ratio < 0.20:
        recommendations.append("No-bruise/negative images are underrepresented. Add normal skin and no-bruise images with empty label files.")
    if small_box_ratio > 0.25:
        recommendations.append("Many bruises are very small. Add more small-bruise examples and consider higher image size during training.")
    if large_box_ratio > 0.10:
        recommendations.append("Some labels are very large. Review annotations and draw tighter boxes around visible bruise regions.")

    if not pred_df.empty:
        failure_rates = pred_df["failure_type"].value_counts(normalize=True)
        if failure_rates.get("false_positive_on_no_bruise", 0.0) > 0.10:
            recommendations.append("False positives are common. Add more no-bruise images and hard negatives such as shadows, skin folds, and normal discoloration.")
        if (pred_df["fn"] > 0).mean() > 0.25:
            recommendations.append("Missed bruises are common. Add more diverse bruise examples and check that all bruises are labeled.")
        if (pred_df["best_iou"] < 0.5).mean() > 0.50 and len(val_df) > 0:
            recommendations.append("Localization is weak. Review label consistency and avoid boxes that include large normal-skin areas.")

    if not recommendations:
        recommendations.append("No single dominant data issue was detected. Review worst false positives/false negatives manually.")
    return recommendations


def main() -> None:
    args = parse_args()
    folds_root, run_root = resolve_paths(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows = []
    prediction_rows = []

    for fold_id in range(1, 6):
        fold_dir = folds_root / f"fold_{fold_id}"
        model_path = run_root / f"rtdetr_fold_{fold_id}" / "weights" / "best.pt"
        if not fold_dir.exists():
            continue

        for split in ["train", "val"]:
            dataset_rows.extend(analyze_split(fold_id, split, fold_dir))

        if model_path.exists():
            prediction_rows.extend(evaluate_predictions(fold_id, fold_dir, model_path, args.conf, args.iou))

    dataset_df = pd.DataFrame(dataset_rows)
    pred_df = pd.DataFrame(prediction_rows)

    dataset_df.to_csv(args.output_dir / "dataset_label_analysis.csv", index=False)
    if not pred_df.empty:
        pred_df.to_csv(args.output_dir / "prediction_failure_analysis.csv", index=False)

    split_summary = (
        dataset_df.groupby(["fold", "split"])
        .agg(
            images=("image", "count"),
            bruise_images=("has_bruise", "sum"),
            no_bruise_images=("has_bruise", lambda s: int((s == 0).sum())),
            total_boxes=("gt_count", "sum"),
            avg_boxes_per_image=("gt_count", "mean"),
            avg_box_area_ratio=("avg_box_area_ratio", "mean"),
            avg_brightness=("brightness", "mean"),
            avg_contrast=("contrast", "mean"),
        )
        .reset_index()
    )
    split_summary.to_csv(args.output_dir / "fold_dataset_summary.csv", index=False)

    if not pred_df.empty:
        failure_summary = (
            pred_df.groupby(["fold", "failure_type"])
            .agg(images=("image", "count"), avg_max_conf=("max_conf", "mean"), avg_best_iou=("best_iou", "mean"))
            .reset_index()
        )
        failure_summary.to_csv(args.output_dir / "failure_type_summary.csv", index=False)
    else:
        failure_summary = pd.DataFrame()

    recommendations = build_recommendations(dataset_df, pred_df)
    report = {
        "folds_root": str(folds_root),
        "run_root": str(run_root),
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "recommendations": recommendations,
    }
    (args.output_dir / "failure_analysis_recommendations.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "failure_analysis_recommendations.txt").write_text("\n".join(f"- {r}" for r in recommendations), encoding="utf-8")

    print(f"Saved analysis to: {args.output_dir}")
    print("\nMain recommendations:")
    for recommendation in recommendations:
        print(f"- {recommendation}")


if __name__ == "__main__":
    main()
