from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import RTDETR


DEFAULT_FAILURE_CSV = Path("final_result_download/failure_analysis/prediction_failure_analysis.csv")
DEFAULT_RUN_ROOT = Path("fold_training_backup/runs/detect/artifacts/runs/cross_validation")
DEFAULT_OUTPUT_DIR = Path("final_result_download/bad_prediction_examples")
BAD_FAILURE_TYPES = {
    "false_positive_on_no_bruise",
    "wrong_location_and_missed_bruise",
    "missed_bruise",
    "missed_bruise_no_prediction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save representative bad prediction examples with annotations.")
    parser.add_argument("--failure-csv", type=Path, default=DEFAULT_FAILURE_CSV)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-per-type", type=int, default=20)
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for idx, part in enumerate(parts):
        if part == "images" and idx + 1 < len(parts):
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def read_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    labels = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, x, y, w, h = parts
        labels.append((int(float(cls)), float(x), float(y), float(w), float(h)))
    return labels


def yolo_to_xyxy(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    cx, cy, bw, bh = x * img_w, y * img_h, w * img_w, h * img_h
    return (
        int(max(0, cx - bw / 2)),
        int(max(0, cy - bh / 2)),
        int(min(img_w - 1, cx + bw / 2)),
        int(min(img_h - 1, cy + bh / 2)),
    )


def draw_text(image, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def draw_ground_truth(image, image_path: Path) -> None:
    img_h, img_w = image.shape[:2]
    for _, x, y, w, h in read_yolo_labels(label_path_for_image(image_path)):
        x1, y1, x2, y2 = yolo_to_xyxy(x, y, w, h, img_w, img_h)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 180, 0), 2)
        draw_text(image, "GT", x1, max(18, y1 - 6), (0, 220, 0))


def draw_predictions(image, model: RTDETR | None, image_path: Path, conf: float) -> None:
    if model is None:
        return
    pred = model.predict(source=str(image_path), imgsz=640, conf=0.001, verbose=False)[0]
    if pred.boxes is None or len(pred.boxes) == 0:
        return
    boxes = pred.boxes.xyxy.cpu().numpy().tolist()
    scores = pred.boxes.conf.cpu().numpy().tolist()
    for box, score in zip(boxes, scores):
        if float(score) < conf:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 220), 2)
        draw_text(image, f"Pred {float(score):.2f}", x1, max(18, y1 - 6), (0, 0, 255))


def selected_examples(df: pd.DataFrame, max_per_type: int) -> pd.DataFrame:
    bad_df = df[df["failure_type"].isin(BAD_FAILURE_TYPES)].copy()
    for col in ["gt_count", "pred_count_at_conf", "max_conf", "best_iou", "fp", "fn"]:
        bad_df[col] = pd.to_numeric(bad_df[col], errors="coerce").fillna(0.0)

    selected = []
    for failure_type, group in bad_df.groupby("failure_type"):
        if failure_type == "false_positive_on_no_bruise":
            group = group.sort_values(["fp", "max_conf"], ascending=[False, False])
        elif failure_type == "wrong_location_and_missed_bruise":
            group = group.sort_values(["fn", "fp", "best_iou"], ascending=[False, False, True])
        else:
            group = group.sort_values(["fn", "max_conf"], ascending=[False, True])
        selected.append(group.head(max_per_type))

    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()


def save_summary(df: pd.DataFrame, selected_df: pd.DataFrame, output_dir: Path) -> None:
    bad_df = df[df["failure_type"].isin(BAD_FAILURE_TYPES)].copy()
    summary = (
        bad_df.groupby("failure_type")
        .agg(
            wrong_images=("image", "count"),
            avg_confidence=("max_conf", lambda s: pd.to_numeric(s, errors="coerce").mean()),
            avg_best_iou=("best_iou", lambda s: pd.to_numeric(s, errors="coerce").mean()),
        )
        .reset_index()
        .sort_values("wrong_images", ascending=False)
    )
    summary.to_csv(output_dir / "wrong_prediction_count_summary.csv", index=False)
    bad_df.to_csv(output_dir / "all_wrong_predictions.csv", index=False)
    selected_df.to_csv(output_dir / "saved_bad_prediction_examples.csv", index=False)

    total_wrong = len(bad_df)
    lines = [
        "# Bad Prediction Examples Summary",
        "",
        f"Total wrong prediction images: {total_wrong}",
        "",
        "| Failure type | Wrong images | Average confidence | Average best IoU |",
        "|---|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['failure_type']} | {int(row['wrong_images'])} | "
            f"{float(row['avg_confidence']):.3f} | {float(row['avg_best_iou']):.3f} |"
        )
    lines.extend(
        [
            "",
            "Annotated examples use green boxes for ground truth and red boxes for model predictions.",
        ]
    )
    (output_dir / "bad_prediction_examples_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.failure_csv)
    selected_df = selected_examples(df, args.max_per_type)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    models: dict[int, RTDETR | None] = {}
    saved_rows = []

    for idx, row in selected_df.iterrows():
        fold = int(row["fold"])
        failure_type = str(row["failure_type"])
        image_path = Path(row["image"])
        if not image_path.exists():
            continue

        if fold not in models:
            model_path = args.run_root / f"rtdetr_fold_{fold}" / "weights" / "best.pt"
            models[fold] = RTDETR(str(model_path)) if model_path.exists() else None

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        draw_ground_truth(image, image_path)
        draw_predictions(image, models[fold], image_path, args.conf)
        header = (
            f"Fold {fold} | {failure_type} | GT={row['gt_count']} Pred={row['pred_count_at_conf']} "
            f"Conf={float(row['max_conf']):.2f} IoU={float(row['best_iou']):.2f}"
        )
        draw_text(image, header, 10, 24, (255, 255, 255))

        out_dir = args.output_dir / failure_type
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"fold_{fold}_{idx:03d}_{image_path.name}"
        cv2.imwrite(str(out_path), image)

        saved_row = row.to_dict()
        saved_row["saved_image"] = str(out_path)
        saved_rows.append(saved_row)

    saved_df = pd.DataFrame(saved_rows)
    save_summary(df, saved_df, args.output_dir)

    print(f"Saved bad prediction examples to: {args.output_dir}")
    print(f"Saved annotated images: {len(saved_df)}")
    print(f"Summary: {args.output_dir / 'bad_prediction_examples_summary.md'}")


if __name__ == "__main__":
    main()
