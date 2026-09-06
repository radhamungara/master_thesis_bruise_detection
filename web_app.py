from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import RTDETR


DEFAULT_MODEL = "runs/detect/artifacts/runs/cross_validation/rtdetr_fold_5/weights/best.pt"


@st.cache_resource
def load_model(model_path: str) -> RTDETR:
    return RTDETR(model_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to RT-DETR best.pt model")
    parser.add_argument("--conf", default=0.25, type=float, help="Prediction confidence threshold")
    return parser.parse_known_args()[0]


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def suppress_duplicate_boxes(detections: pd.DataFrame, iou_threshold: float, max_detections: int) -> pd.DataFrame:
    if detections.empty:
        return detections

    detections = detections.sort_values("confidence", ascending=False).reset_index(drop=True)
    keep: list[int] = []
    boxes = detections[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)

    for idx, box in enumerate(boxes):
        if len(keep) >= max_detections:
            break
        if all(box_iou(box, boxes[kept_idx]) < iou_threshold for kept_idx in keep):
            keep.append(idx)

    return detections.iloc[keep].reset_index(drop=True)


def detections_to_table(result, iou_threshold: float, max_detections: int) -> pd.DataFrame:
    names = result.names
    rows = []
    if result.boxes is None or len(result.boxes) == 0:
        return pd.DataFrame(columns=["class", "confidence", "x1", "y1", "x2", "y2"])

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    for box, conf, cls_id in zip(boxes, confs, classes):
        rows.append(
            {
                "class": names.get(int(cls_id), str(cls_id)) if isinstance(names, dict) else str(cls_id),
                "confidence": round(float(conf), 4),
                "x1": round(float(box[0]), 1),
                "y1": round(float(box[1]), 1),
                "x2": round(float(box[2]), 1),
                "y2": round(float(box[3]), 1),
            }
        )
    detections = pd.DataFrame(rows)
    return suppress_duplicate_boxes(detections, iou_threshold=iou_threshold, max_detections=max_detections)


def draw_detections(image_rgb: np.ndarray, detections: pd.DataFrame) -> np.ndarray:
    annotated = image_rgb.copy()
    color = (20, 95, 255)
    for _, row in detections.iterrows():
        x1, y1, x2, y2 = [int(row[c]) for c in ["x1", "y1", "x2", "y2"]]
        label = f"{row['class']} {row['confidence']:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        label_y = max(20, y1 - 8)
        cv2.putText(annotated, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    return annotated


def image_download_bytes(image_rgb: np.ndarray) -> bytes:
    pil_image = Image.fromarray(image_rgb)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title="Bruise Detection", layout="wide")
    st.title("Bruise Detection")

    with st.sidebar:
        st.header("Model")
        model_path = st.text_input("Model path", value=args.model)
        conf = st.slider("Confidence threshold", min_value=0.01, max_value=0.95, value=float(args.conf), step=0.01)
        iou = st.slider("Duplicate IoU filter", min_value=0.05, max_value=0.95, value=0.30, step=0.05)
        max_detections = st.number_input("Max detections", min_value=1, max_value=50, value=5, step=1)
        model_exists = Path(model_path).exists()
        st.caption("Model found" if model_exists else "Model not found")

    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded is None:
        st.info("Upload an image to run bruise detection.")
        return

    if not Path(model_path).exists():
        st.error(f"Model file does not exist: {model_path}")
        return

    image = Image.open(uploaded).convert("RGB")
    image_rgb = np.array(image)

    with st.spinner("Running detection..."):
        model = load_model(model_path)
        result = model.predict(source=image_rgb, conf=conf, iou=iou, max_det=int(max_detections), verbose=False)[0]
        detections = detections_to_table(result, iou_threshold=iou, max_detections=int(max_detections))
        annotated_rgb = draw_detections(image_rgb, detections)

    left, right = st.columns([2, 1])
    with left:
        st.image(annotated_rgb, caption="Detected result", use_container_width=True)
        st.download_button(
            "Download annotated image",
            data=image_download_bytes(annotated_rgb),
            file_name="bruise_detection_result.png",
            mime="image/png",
        )

    with right:
        st.metric("Detections", len(detections))
        if detections.empty:
            st.warning("No detections above the selected confidence threshold.")
        else:
            st.dataframe(detections, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
