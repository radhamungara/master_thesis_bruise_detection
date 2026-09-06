from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import RTDETR


FAILURE_CSV = Path("latest_training_result/reports/failure_analysis_corrected/prediction_failure_analysis.csv")
RUN_ROOT = Path("latest_training_result/latest_training_download/cross_validation")
OUT_PNG = Path("thesis_draft/figures/Figure_5_7_RT_DETR_Large_Representative_Localisation_Failure_Examples.png")
OUT_CSV = Path("thesis_draft/figures/Figure_5_7_RT_DETR_Large_Representative_Localisation_Failure_Examples_values.csv")
CONF = 0.25


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for idx, part in enumerate(parts):
        if part == "images" and idx + 1 < len(parts):
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def read_yolo_labels(label_path: Path) -> list[tuple[float, float, float, float]]:
    labels = []
    if not label_path.exists():
        return labels
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, x, y, w, h = parts
        labels.append((float(x), float(y), float(w), float(h)))
    return labels


def yolo_to_xyxy(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    cx, cy = x * img_w, y * img_h
    bw, bh = w * img_w, h * img_h
    return (
        int(max(0, cx - bw / 2)),
        int(max(0, cy - bh / 2)),
        int(min(img_w - 1, cx + bw / 2)),
        int(min(img_h - 1, cy + bh / 2)),
    )


def draw_label(img: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_ground_truth(img: np.ndarray, image_path: Path) -> None:
    h, w = img.shape[:2]
    for x, y, bw, bh in read_yolo_labels(label_path_for_image(image_path)):
        x1, y1, x2, y2 = yolo_to_xyxy(x, y, bw, bh, w, h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 180, 0), 3)
        draw_label(img, "GT", x1, max(22, y1 - 7), (0, 120, 0))


def draw_predictions(img: np.ndarray, model: RTDETR, image_path: Path) -> int:
    result = model.predict(source=str(image_path), imgsz=640, conf=CONF, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return 0
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    drawn = 0
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 220), 3)
        draw_label(img, f"Pred {score:.2f}", x1, max(22, y1 - 7), (0, 0, 180))
        drawn += 1
    return drawn


def resize_to_tile(img: np.ndarray, width: int = 620, height: int = 470) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(width / w, height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def select_examples(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["gt_count", "pred_count_at_conf", "max_conf", "best_iou", "tp", "fp", "fn"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    candidates = df[
        (df["failure_type"] == "wrong_location_and_missed_bruise")
        & (df["gt_count"] > 0)
        & (df["pred_count_at_conf"] > 0)
    ].copy()
    candidates["exists"] = candidates.apply(lambda row: resolve_image_path(Path(row["image"]), int(row["fold"])).exists(), axis=1)
    candidates = candidates[candidates["exists"]]
    candidates = candidates.sort_values(["best_iou", "fn", "max_conf"], ascending=[True, False, False])
    return candidates.head(4).reset_index(drop=True)


def resolve_image_path(image_path: Path, fold: int) -> Path:
    if image_path.exists():
        return image_path

    search_dir = Path("artifacts/fold_datasets") / f"fold_{fold}" / "images" / "val"
    if not search_dir.exists():
        return image_path

    stem = image_path.stem
    matches = sorted(search_dir.glob(f"*{stem}*{image_path.suffix}"))
    if matches:
        return matches[0]

    name = image_path.name
    compact = name.replace("Bruises__", "")
    matches = sorted(search_dir.glob(f"*{compact}"))
    if matches:
        return matches[0]

    return image_path


def main() -> None:
    df = pd.read_csv(FAILURE_CSV)
    selected = select_examples(df)
    if len(selected) < 4:
        raise RuntimeError(f"Expected at least four localisation examples, found {len(selected)}")

    models: dict[int, RTDETR] = {}
    tiles = []
    saved_rows = []
    panel_names = ["A", "B", "C", "D"]

    for panel, row in zip(panel_names, selected.itertuples(index=False)):
        fold = int(row.fold)
        if fold not in models:
            model_path = RUN_ROOT / f"rtdetr_fold_{fold}" / "weights" / "best.pt"
            if not model_path.exists():
                raise FileNotFoundError(model_path)
            models[fold] = RTDETR(str(model_path))

        image_path = resolve_image_path(Path(row.image), fold)
        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError(f"Could not read {image_path}")

        draw_ground_truth(img, image_path)
        drawn_predictions = draw_predictions(img, models[fold], image_path)
        tile = resize_to_tile(img)

        header = (
            f"{panel}. Fold {fold}: localisation failure | "
            f"GT={int(row.gt_count)}, Pred={int(row.pred_count_at_conf)}, "
            f"best IoU={float(row.best_iou):.2f}"
        )
        cv2.putText(tile, header, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(tile, header, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
        tiles.append(tile)

        saved_rows.append(
            {
                "panel": panel,
                "fold": fold,
                "image": row.image,
                "gt_count": int(row.gt_count),
                "pred_count_at_conf": int(row.pred_count_at_conf),
                "drawn_predictions": drawn_predictions,
                "max_conf": float(row.max_conf),
                "best_iou": float(row.best_iou),
                "tp": int(row.tp),
                "fp": int(row.fp),
                "fn": int(row.fn),
                "failure_type": row.failure_type,
            }
        )

    gap = 35
    title_h = 90
    foot_h = 70
    tile_h, tile_w = tiles[0].shape[:2]
    canvas = np.full((title_h + 2 * tile_h + gap + foot_h, 2 * tile_w + gap, 3), 255, dtype=np.uint8)
    title = "RT-DETR-Large Representative Localisation Failure Examples"
    cv2.putText(canvas, title, (130, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (30, 30, 30), 3, cv2.LINE_AA)

    positions = [
        (0, title_h),
        (tile_w + gap, title_h),
        (0, title_h + tile_h + gap),
        (tile_w + gap, title_h + tile_h + gap),
    ]
    for tile, (x, y) in zip(tiles, positions):
        canvas[y : y + tile_h, x : x + tile_w] = tile

    note = "Green boxes show ground-truth bruise annotations; red boxes show RT-DETR-Large predictions at confidence 0.25."
    cv2.putText(canvas, note, (20, canvas.shape[0] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 2, cv2.LINE_AA)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_PNG), canvas)
    pd.DataFrame(saved_rows).to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
    main()
