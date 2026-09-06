from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPORTS_ROOT = Path("artifacts/reports")
OUT_PNG = Path("thesis_draft/figures/Figure_5_7_RT_DETR_Large_Balanced_Representative_Localisation_Failure_Examples.png")
OUT_CSV = Path("thesis_draft/figures/Figure_5_7_RT_DETR_Large_Balanced_Representative_Localisation_Failure_Examples_values.csv")


def to_local_path(path_text: str) -> Path:
    prefix = "/home/radha_mungara/Bruise Detection/"
    if path_text.startswith(prefix):
        return Path(path_text[len(prefix) :])
    return Path(path_text)


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
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 2, cv2.LINE_AA)


def draw_ground_truth(img: np.ndarray, image_path: Path) -> None:
    h, w = img.shape[:2]
    for x, y, bw, bh in read_yolo_labels(label_path_for_image(image_path)):
        x1, y1, x2, y2 = yolo_to_xyxy(x, y, bw, bh, w, h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 180, 0), 3)
        draw_label(img, "GT", x1, max(22, y1 - 7), (0, 120, 0))


def resize_to_tile(img: np.ndarray, width: int = 700, height: int = 500) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(width / w, height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def load_records() -> pd.DataFrame:
    frames = []
    for fold_dir in sorted(REPORTS_ROOT.glob("fold_*_analysis")):
        csv_path = fold_dir / "image_level_analysis.csv"
        if not csv_path.exists():
            continue
        fold = int(fold_dir.name.split("_")[1])
        frame = pd.read_csv(csv_path)
        frame.insert(0, "fold", fold)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No fold image_level_analysis.csv files found under {REPORTS_ROOT}")
    return pd.concat(frames, ignore_index=True)


def select_examples(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["gt_present", "pred_present", "matched_iou", "max_conf", "best_iou", "gt_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["local_image"] = df["image"].map(lambda value: str(to_local_path(str(value))))
    candidates = df[
        (df["gt_present"] == 1)
        & (df["matched_iou"] == 0)
        & (df["pred_present"] == 0)
        & (df["local_image"].map(lambda value: Path(value).exists()))
    ].copy()
    selected = (
        candidates.sort_values(["fold", "max_conf"], ascending=[True, False])
        .groupby("fold", as_index=False)
        .head(1)
        .sort_values("max_conf", ascending=False)
        .head(4)
        .reset_index(drop=True)
    )
    if len(selected) < 4:
        raise RuntimeError(f"Expected four examples, found {len(selected)}")
    return selected


def main() -> None:
    selected = select_examples(load_records())
    display_images = []
    panel_titles = []
    rows = []
    panels = ["A", "B", "C", "D"]

    for panel, row in zip(panels, selected.itertuples(index=False)):
        image_path = Path(row.local_image)
        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError(f"Could not read {image_path}")
        draw_ground_truth(img, image_path)
        display_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        panel_titles.append(
            f"{panel}. Fold {int(row.fold)}: missed localisation\n"
            f"GT={int(row.gt_count)}, max conf={float(row.max_conf):.2f}, IoU={float(row.best_iou):.2f}"
        )
        rows.append(
            {
                "panel": panel,
                "fold": int(row.fold),
                "image": row.image,
                "local_image": str(image_path),
                "gt_count": int(row.gt_count),
                "pred_present": int(row.pred_present),
                "matched_iou": int(row.matched_iou),
                "max_conf": float(row.max_conf),
                "best_iou": float(row.best_iou),
            }
        )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.titlesize": 18})
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10), dpi=200)
    for ax, image, panel_title in zip(axes.ravel(), display_images, panel_titles):
        ax.imshow(image)
        ax.set_title(panel_title, fontweight="bold", pad=8)
        ax.axis("off")

    fig.suptitle("RT-DETR-Large Representative Localisation Failure Examples", fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.025,
        "Green boxes show ground-truth bruise annotations. The stored RT-DETR-Large evaluation recorded no matched prediction. "
        "Examples are from Balanced cluster dataset-1000 fold-level validation records; prediction-presence threshold was 0.25.",
        ha="center",
        va="bottom",
        fontsize=9,
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
    main()
