from __future__ import annotations

import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

from .config import HyperoptSubsetConfig, PatchConfig, PathsConfig, YoloDatasetSource

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TRAIN_AUGMENTATIONS = [
    {"name": "brightness_change", "brightness": 1.10},
    {"name": "contrast_up", "contrast": 1.15},
    {"name": "hue_saturation", "hue_shift": 4.0, "saturation": 1.08},
]


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_image_path(train_txt_line: str, paths: PathsConfig) -> Path | None:
    p = Path(train_txt_line.strip())
    filename = p.name
    candidates = [paths.images_dir / filename]
    if p.is_absolute():
        candidates.append(p)
    candidates.extend(paths.dataset_root.rglob(filename))
    for c in candidates:
        if c.exists() and c.suffix.lower() in IMG_EXTS:
            return c
    return None


def find_label_for_image(img_path: Path | None, labels_dir: Path) -> Path | None:
    if img_path is None:
        return None
    matches = list(labels_dir.rglob(f"{img_path.stem}.txt"))
    return matches[0] if matches else None


def validate_yolo_label(label_path: Path, nc: int, allow_empty: bool = False) -> tuple[bool, list[int], list[str]]:
    errors: list[str] = []
    classes: list[int] = []
    if not label_path.exists():
        return False, [], ["missing_label"]
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return (True, [], []) if allow_empty else (False, [], ["empty_label"])
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 5 and (len(parts) < 7 or len(parts[1:]) % 2 != 0):
            errors.append(f"line_{i}_wrong_number_of_columns")
            continue
        try:
            cls = int(float(parts[0]))
            values = list(map(float, parts[1:]))
        except ValueError:
            errors.append(f"line_{i}_non_numeric_value")
            continue
        if cls < 0 or cls >= nc:
            errors.append(f"line_{i}_invalid_class_{cls}")
        if len(values) == 4:
            x, y, w, h = values
            if not (0 <= x <= 1):
                errors.append(f"line_{i}_x_out_of_range")
            if not (0 <= y <= 1):
                errors.append(f"line_{i}_y_out_of_range")
            if not (0 < w <= 1):
                errors.append(f"line_{i}_width_out_of_range")
            if not (0 < h <= 1):
                errors.append(f"line_{i}_height_out_of_range")
        else:
            xs = values[0::2]
            ys = values[1::2]
            if any(x < 0 or x > 1 for x in xs):
                errors.append(f"line_{i}_polygon_x_out_of_range")
            if any(y < 0 or y > 1 for y in ys):
                errors.append(f"line_{i}_polygon_y_out_of_range")
        classes.append(cls)
    return len(errors) == 0, classes, errors


def make_group_id(image_path: str) -> str:
    stem = Path(image_path).stem
    stem = re.sub(r"\.rf\.[a-zA-Z0-9]+$", "", stem)
    for pat in [
        r"__rep_\d+$",
        r"_aug\d+$",
        r"-aug\d+$",
        r"_rotate\d+$",
        r"_rot\d+$",
        r"_flip.*$",
        r"_bright.*$",
        r"_zoom.*$",
        r"_blur.*$",
        r"_crop.*$",
    ]:
        stem = re.sub(pat, "", stem, flags=re.IGNORECASE)
    return stem


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)


def safe_write_empty_label(dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("", encoding="utf-8")


def _parse_yolo_points(label_path: Path, img_w: int, img_h: int) -> list[tuple[int, np.ndarray]]:
    rows: list[tuple[int, np.ndarray]] = []
    if not label_path.exists():
        return rows

    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        cls = int(float(parts[0]))
        values = [float(v) for v in parts[1:]]
        if len(values) == 4:
            x, y, w, h = values
            x1 = (x - w / 2) * img_w
            y1 = (y - h / 2) * img_h
            x2 = (x + w / 2) * img_w
            y2 = (y + h / 2) * img_h
            points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        else:
            xs = np.array(values[0::2], dtype=np.float32) * img_w
            ys = np.array(values[1::2], dtype=np.float32) * img_h
            points = np.stack([xs, ys], axis=1)
        rows.append((cls, points))
    return rows


def _transform_labels(label_path: Path, matrix: np.ndarray, img_w: int, img_h: int) -> list[str]:
    transformed_rows: list[str] = []
    for cls, points in _parse_yolo_points(label_path, img_w=img_w, img_h=img_h):
        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        homogeneous = np.hstack([points, ones])
        transformed = homogeneous @ matrix.T
        transformed[:, 0] = np.clip(transformed[:, 0], 0, img_w - 1)
        transformed[:, 1] = np.clip(transformed[:, 1], 0, img_h - 1)

        x1, y1 = transformed.min(axis=0)
        x2, y2 = transformed.max(axis=0)
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w < 2 or box_h < 2:
            continue

        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        norm_w = box_w / img_w
        norm_h = box_h / img_h
        transformed_rows.append(f"{cls} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
    return transformed_rows


def write_detection_box_label(src_label: Path, dst_label: Path, img_w: int, img_h: int) -> int:
    """Write a YOLO detection-box label from box or polygon source labels."""
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    label_rows = _transform_labels(src_label, matrix=identity, img_w=img_w, img_h=img_h)
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    dst_label.write_text("\n".join(label_rows) + ("\n" if label_rows else ""), encoding="utf-8")
    return len(label_rows)


def _label_boxes_xyxy(label_path: Path, img_w: int, img_h: int) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    for cls, points in _parse_yolo_points(label_path, img_w=img_w, img_h=img_h):
        x1, y1 = points.min(axis=0)
        x2, y2 = points.max(axis=0)
        boxes.append((cls, float(x1), float(y1), float(x2), float(y2)))
    return boxes


def _box_context_crop_bounds(
    boxes: list[tuple[int, float, float, float, float]],
    img_w: int,
    img_h: int,
    context_scale: float,
) -> tuple[int, int, int, int]:
    x1 = min(box[1] for box in boxes)
    y1 = min(box[2] for box in boxes)
    x2 = max(box[3] for box in boxes)
    y2 = max(box[4] for box in boxes)
    box_w = max(2.0, x2 - x1)
    box_h = max(2.0, y2 - y1)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    crop_w = min(float(img_w), max(box_w * context_scale, img_w * 0.35))
    crop_h = min(float(img_h), max(box_h * context_scale, img_h * 0.35))
    left = center_x - crop_w / 2.0
    top = center_y - crop_h / 2.0
    left = min(max(left, 0.0), max(0.0, img_w - crop_w))
    top = min(max(top, 0.0), max(0.0, img_h - crop_h))
    right = min(float(img_w), left + crop_w)
    bottom = min(float(img_h), top + crop_h)
    return int(round(left)), int(round(top)), int(round(right)), int(round(bottom))


def _deterministic_rng_for_path(path: Path) -> np.random.Generator:
    seed = sum((i + 1) * ord(ch) for i, ch in enumerate(path.stem)) % (2**32)
    return np.random.default_rng(seed)


def _random_crop_bounds(img_w: int, img_h: int, crop_fraction: float, rng: np.random.Generator) -> tuple[int, int, int, int]:
    crop_fraction = min(max(crop_fraction, 0.5), 1.0)
    crop_w = max(2, int(round(img_w * crop_fraction)))
    crop_h = max(2, int(round(img_h * crop_fraction)))
    max_left = max(0, img_w - crop_w)
    max_top = max(0, img_h - crop_h)
    left = int(rng.integers(0, max_left + 1)) if max_left else 0
    top = int(rng.integers(0, max_top + 1)) if max_top else 0
    return left, top, left + crop_w, top + crop_h


def _crop_resize_label_rows(
    label_path: Path,
    crop_bounds: tuple[int, int, int, int],
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> list[str]:
    left, top, right, bottom = crop_bounds
    crop_w = max(1, right - left)
    crop_h = max(1, bottom - top)
    rows: list[str] = []
    for cls, points in _parse_yolo_points(label_path, img_w=src_w, img_h=src_h):
        transformed = points.copy()
        transformed[:, 0] = (transformed[:, 0] - left) * (dst_w / crop_w)
        transformed[:, 1] = (transformed[:, 1] - top) * (dst_h / crop_h)
        transformed[:, 0] = np.clip(transformed[:, 0], 0, dst_w - 1)
        transformed[:, 1] = np.clip(transformed[:, 1], 0, dst_h - 1)

        x1, y1 = transformed.min(axis=0)
        x2, y2 = transformed.max(axis=0)
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w < 2 or box_h < 2:
            continue

        x_center = ((x1 + x2) / 2) / dst_w
        y_center = ((y1 + y2) / 2) / dst_h
        norm_w = box_w / dst_w
        norm_h = box_h / dst_h
        rows.append(f"{cls} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
    return rows


def _context_crop_augmentation(
    image: np.ndarray,
    label_path: Path,
    img_path: Path,
    spec: dict,
) -> tuple[np.ndarray, list[str]]:
    img_h, img_w = image.shape[:2]
    boxes = _label_boxes_xyxy(label_path, img_w=img_w, img_h=img_h)
    if boxes:
        crop_bounds = _box_context_crop_bounds(
            boxes,
            img_w=img_w,
            img_h=img_h,
            context_scale=float(spec.get("context_scale", 2.0)),
        )
        label_rows = _crop_resize_label_rows(
            label_path,
            crop_bounds=crop_bounds,
            src_w=img_w,
            src_h=img_h,
            dst_w=img_w,
            dst_h=img_h,
        )
    else:
        crop_bounds = _random_crop_bounds(
            img_w=img_w,
            img_h=img_h,
            crop_fraction=float(spec.get("negative_crop_fraction", 0.75)),
            rng=_deterministic_rng_for_path(img_path),
        )
        label_rows = []

    left, top, right, bottom = crop_bounds
    crop = image[top:bottom, left:right]
    resized = cv2.resize(crop, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    return resized, label_rows


def _empty_image_features() -> dict:
    return {
        "img_w": np.nan,
        "img_h": np.nan,
        "brightness": np.nan,
        "contrast": np.nan,
        "skin_tone_proxy": np.nan,
        "box_count": 0,
        "mean_box_area_ratio": 0.0,
        "max_box_area_ratio": 0.0,
        "mean_box_x": np.nan,
        "mean_box_y": np.nan,
        "difficulty_score": 0.0,
        "feature_status": "unreadable",
    }


def image_representative_features(row: pd.Series) -> dict:
    image = cv2.imread(str(Path(row["image"])))
    if image is None:
        return _empty_image_features()

    img_h, img_w = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    features = {
        "img_w": int(img_w),
        "img_h": int(img_h),
        "brightness": float(lab[:, :, 0].mean()),
        "contrast": float(lab[:, :, 0].std()),
        "skin_tone_proxy": float(hsv[:, :, 0].mean()),
        "box_count": 0,
        "mean_box_area_ratio": 0.0,
        "max_box_area_ratio": 0.0,
        "mean_box_x": np.nan,
        "mean_box_y": np.nan,
        "difficulty_score": 0.0,
        "feature_status": "ok",
    }
    if bool(row["positive"]):
        boxes = _label_boxes_xyxy(Path(row["label"]), img_w=img_w, img_h=img_h)
        if boxes:
            areas = []
            centers_x = []
            centers_y = []
            for _, x1, y1, x2, y2 in boxes:
                box_w = max(0.0, x2 - x1)
                box_h = max(0.0, y2 - y1)
                areas.append((box_w * box_h) / float(img_w * img_h))
                centers_x.append(((x1 + x2) / 2.0) / img_w)
                centers_y.append(((y1 + y2) / 2.0) / img_h)
            features["box_count"] = len(boxes)
            features["mean_box_area_ratio"] = float(np.mean(areas))
            features["max_box_area_ratio"] = float(np.max(areas))
            features["mean_box_x"] = float(np.mean(centers_x))
            features["mean_box_y"] = float(np.mean(centers_y))
            small_box_score = 1.0 - min(features["mean_box_area_ratio"] / 0.05, 1.0)
            low_contrast_score = 1.0 - min(features["contrast"] / 60.0, 1.0)
            features["difficulty_score"] = float((small_box_score + low_contrast_score) / 2.0)
    else:
        low_contrast_score = 1.0 - min(features["contrast"] / 60.0, 1.0)
        features["difficulty_score"] = float(low_contrast_score)

    return features


def _quantile_bin(values: pd.Series, bins: int) -> pd.Series:
    if values.nunique(dropna=True) <= 1:
        return pd.Series(["all"] * len(values), index=values.index)
    try:
        return pd.qcut(values.rank(method="first"), q=min(bins, len(values)), labels=False, duplicates="drop").astype(str)
    except ValueError:
        return pd.Series(["all"] * len(values), index=values.index)


def _sample_representative_class(class_df: pd.DataFrame, count: int, random_state: int) -> pd.DataFrame:
    if len(class_df) < count:
        class_name = "Bruises" if bool(class_df["positive"].iloc[0]) else "Non-Bruises"
        raise ValueError(f"Requested {count} {class_name} images, but only {len(class_df)} valid images are available.")

    df = class_df.copy().reset_index(drop=True)
    df["brightness_bin"] = _quantile_bin(df["brightness"], bins=4)
    df["tone_bin"] = _quantile_bin(df["skin_tone_proxy"], bins=4)
    df["difficulty_bin"] = _quantile_bin(df["difficulty_score"], bins=3)
    if bool(df["positive"].iloc[0]):
        df["size_bin"] = _quantile_bin(df["mean_box_area_ratio"], bins=4)
        df["selection_bin"] = (
            df["brightness_bin"] + "_"
            + df["tone_bin"] + "_"
            + df["difficulty_bin"] + "_"
            + df["size_bin"]
        )
    else:
        df["selection_bin"] = df["brightness_bin"] + "_" + df["tone_bin"] + "_" + df["difficulty_bin"]

    rng = np.random.default_rng(random_state)
    selected_indices: list[int] = []
    grouped = list(df.groupby("selection_bin", sort=True).groups.values())
    rng.shuffle(grouped)
    while len(selected_indices) < count:
        added = False
        for group_indices in grouped:
            available = [int(i) for i in group_indices if int(i) not in selected_indices]
            if not available:
                continue
            selected_indices.append(int(rng.choice(available)))
            added = True
            if len(selected_indices) == count:
                break
        if not added:
            break

    if len(selected_indices) < count:
        remaining = [int(i) for i in df.index if int(i) not in selected_indices]
        rng.shuffle(remaining)
        selected_indices.extend(remaining[: count - len(selected_indices)])

    return df.iloc[selected_indices].copy().reset_index(drop=True)


def _sample_representative_by_cluster(
    class_df: pd.DataFrame,
    count_per_cluster: int,
    random_state: int,
) -> pd.DataFrame:
    if count_per_cluster <= 0:
        raise ValueError(f"Cluster sample count must be positive, got {count_per_cluster}.")

    sampled_groups = []
    class_name = "Bruises" if bool(class_df["positive"].iloc[0]) else "Non-Bruises"
    for cluster, cluster_df in class_df.groupby("cluster", sort=True):
        if len(cluster_df) < count_per_cluster:
            raise ValueError(
                f"Requested {count_per_cluster} {class_name} images from cluster {cluster}, "
                f"but only {len(cluster_df)} valid images are available."
            )
        sampled_groups.append(
            _sample_representative_class(
                cluster_df,
                count=count_per_cluster,
                random_state=random_state + len(sampled_groups),
            )
        )

    return pd.concat(sampled_groups, ignore_index=True)


def _axis_starts(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    final_start = length - patch_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _patch_label_rows(
    boxes: list[tuple[int, float, float, float, float]],
    left: int,
    top: int,
    patch_size: int,
    min_box_size: int,
) -> list[str]:
    rows: list[str] = []
    patch_right = left + patch_size
    patch_bottom = top + patch_size
    for cls, x1, y1, x2, y2 in boxes:
        ix1 = max(x1, left)
        iy1 = max(y1, top)
        ix2 = min(x2, patch_right)
        iy2 = min(y2, patch_bottom)
        box_w = ix2 - ix1
        box_h = iy2 - iy1
        if box_w < min_box_size or box_h < min_box_size:
            continue

        x_center = ((ix1 + ix2) / 2 - left) / patch_size
        y_center = ((iy1 + iy2) / 2 - top) / patch_size
        norm_w = box_w / patch_size
        norm_h = box_h / patch_size
        rows.append(f"{cls} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
    return rows


def _crop_patch(image: np.ndarray, left: int, top: int, patch_size: int) -> np.ndarray:
    patch = image[top : top + patch_size, left : left + patch_size]
    pad_h = patch_size - patch.shape[0]
    pad_w = patch_size - patch.shape[1]
    if pad_h > 0 or pad_w > 0:
        patch = cv2.copyMakeBorder(patch, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return patch


def _limit_patch_candidates(candidates: list[dict], limit: int) -> list[dict]:
    if limit < 0 or len(candidates) <= limit:
        return candidates
    if limit == 0:
        return []
    idxs = np.linspace(0, len(candidates) - 1, num=limit, dtype=int)
    return [candidates[int(i)] for i in idxs]


def write_image_patches(
    img_src: Path,
    lab_src: Path,
    img_dst_dir: Path,
    lab_dst_dir: Path,
    dst_stem: str,
    positive: bool,
    patch_config: PatchConfig,
) -> list[dict]:
    rows: list[dict] = []
    candidates: list[dict] = []
    image = cv2.imread(str(img_src))
    if image is None:
        return [
            {
                "source_image": str(img_src),
                "status": "skipped_unreadable",
            }
        ]

    img_h, img_w = image.shape[:2]
    boxes = _label_boxes_xyxy(lab_src, img_w=img_w, img_h=img_h) if positive else []
    x_starts = _axis_starts(img_w, patch_config.size, patch_config.stride)
    y_starts = _axis_starts(img_h, patch_config.size, patch_config.stride)
    kept_empty_negative_patches = 0

    img_dst_dir.mkdir(parents=True, exist_ok=True)
    lab_dst_dir.mkdir(parents=True, exist_ok=True)
    for top in y_starts:
        for left in x_starts:
            label_rows = _patch_label_rows(
                boxes,
                left=left,
                top=top,
                patch_size=patch_config.size,
                min_box_size=patch_config.min_box_size,
            )
            if positive and not label_rows and not patch_config.keep_empty_positive_patches:
                continue
            if not positive and not patch_config.keep_empty_negative_patches:
                continue
            if (
                not positive
                and not label_rows
                and patch_config.max_empty_negative_patches_per_image >= 0
                and kept_empty_negative_patches >= patch_config.max_empty_negative_patches_per_image
            ):
                continue

            if not positive and not label_rows:
                kept_empty_negative_patches += 1
            candidates.append(
                {
                    "source_image": str(img_src),
                    "x": left,
                    "y": top,
                    "boxes": len(label_rows),
                    "label_rows": label_rows,
                    "status": "created",
                }
            )

    if positive:
        candidates = _limit_patch_candidates(candidates, patch_config.max_positive_patches_per_image)

    for candidate in candidates:
        left = int(candidate["x"])
        top = int(candidate["y"])
        label_rows = candidate.pop("label_rows")
        patch_stem = f"{dst_stem}__patch_x{left}_y{top}"
        patch_img_path = img_dst_dir / f"{patch_stem}{img_src.suffix}"
        patch_label_path = lab_dst_dir / f"{patch_stem}.txt"
        cv2.imwrite(str(patch_img_path), _crop_patch(image, left=left, top=top, patch_size=patch_config.size))
        patch_label_path.write_text("\n".join(label_rows) + ("\n" if label_rows else ""), encoding="utf-8")
        candidate["patch_image"] = str(patch_img_path)
        candidate["patch_label"] = str(patch_label_path)
        rows.append(candidate)
    return rows


def write_fold_image(
    img_src: Path,
    lab_src: Path,
    img_dst_dir: Path,
    lab_dst_dir: Path,
    dst_name: str,
    positive: bool,
) -> None:
    safe_copy(img_src, img_dst_dir / dst_name)
    lab_dst = lab_dst_dir / f"{Path(dst_name).stem}.txt"
    if positive:
        image = cv2.imread(str(img_src))
        if image is None:
            safe_copy(lab_src, lab_dst)
        else:
            img_h, img_w = image.shape[:2]
            write_detection_box_label(lab_src, lab_dst, img_w=img_w, img_h=img_h)
    else:
        safe_write_empty_label(lab_dst)


def _augmentation_matrix(img_w: int, img_h: int, spec: dict) -> np.ndarray:
    if spec.get("flip_lr"):
        return np.array([[-1.0, 0.0, img_w - 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    degrees = float(spec.get("degrees", 0.0))
    scale = float(spec.get("scale", 1.0))
    matrix = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), degrees, scale).astype(np.float32)
    translate_x = float(spec.get("translate_x", 0.0)) * img_w
    translate_y = float(spec.get("translate_y", 0.0)) * img_h
    matrix[:, 2] += [translate_x, translate_y]
    return matrix


def _apply_brightness(image: np.ndarray, brightness: float) -> np.ndarray:
    if brightness == 1.0:
        return image
    adjusted = image.astype(np.float32) * brightness
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def _apply_color_adjustments(image: np.ndarray, spec: dict) -> np.ndarray:
    adjusted = _apply_brightness(image, float(spec.get("brightness", 1.0)))

    contrast = float(spec.get("contrast", 1.0))
    if contrast != 1.0:
        adjusted = np.clip((adjusted.astype(np.float32) - 127.5) * contrast + 127.5, 0, 255).astype(np.uint8)

    hue_shift = float(spec.get("hue_shift", 0.0))
    saturation = float(spec.get("saturation", 1.0))
    if hue_shift != 0.0 or saturation != 1.0:
        hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    blur = int(spec.get("blur", 0))
    if blur > 1:
        if blur % 2 == 0:
            blur += 1
        adjusted = cv2.GaussianBlur(adjusted, (blur, blur), 0)

    return adjusted


def augment_train_split(fold_dir: Path, augmentations: list[dict] | None = None) -> pd.DataFrame:
    """Physically augment only the training split of one fold."""
    augmentations = augmentations or TRAIN_AUGMENTATIONS
    img_train_dir = fold_dir / "images" / "train"
    lab_train_dir = fold_dir / "labels" / "train"
    rows: list[dict] = []

    original_images = sorted(
        [
            p
            for p in img_train_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS and "__aug_" not in p.stem
        ]
    )
    for img_path in original_images:
        label_path = lab_train_dir / f"{img_path.stem}.txt"
        image = cv2.imread(str(img_path))
        if image is None:
            rows.append({"image": str(img_path), "status": "skipped_unreadable"})
            continue

        img_h, img_w = image.shape[:2]
        for spec in augmentations:
            aug_name = str(spec["name"])
            if spec.get("context_crop"):
                aug_img, label_rows = _context_crop_augmentation(
                    image=image,
                    label_path=label_path,
                    img_path=img_path,
                    spec=spec,
                )
            else:
                matrix = _augmentation_matrix(img_w=img_w, img_h=img_h, spec=spec)
                aug_img = cv2.warpAffine(
                    image,
                    matrix,
                    (img_w, img_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                aug_img = _apply_color_adjustments(aug_img, spec)
                label_rows = _transform_labels(label_path, matrix=matrix, img_w=img_w, img_h=img_h)

            aug_stem = f"{img_path.stem}__aug_{aug_name}"
            aug_img_path = img_train_dir / f"{aug_stem}{img_path.suffix}"
            aug_label_path = lab_train_dir / f"{aug_stem}.txt"
            cv2.imwrite(str(aug_img_path), aug_img)

            aug_label_path.write_text("\n".join(label_rows) + ("\n" if label_rows else ""), encoding="utf-8")
            rows.append(
                {
                    "source_image": str(img_path),
                    "augmented_image": str(aug_img_path),
                    "augmented_label": str(aug_label_path),
                    "augmentation": aug_name,
                    "boxes": len(label_rows),
                    "status": "created",
                }
            )

    return pd.DataFrame(rows)


def find_source_label_for_image(img_path: Path, source: YoloDatasetSource) -> Path:
    try:
        relative_image_path = img_path.relative_to(source.images_dir)
        mirrored_label_path = source.labels_dir / relative_image_path.with_suffix(".txt")
        if mirrored_label_path.exists():
            return mirrored_label_path
    except ValueError:
        mirrored_label_path = None

    parts = img_path.parts
    if "images" in parts:
        image_idx = parts.index("images")
        clustered_label_path = Path(*parts[:image_idx], "labels", *parts[image_idx + 1 :]).with_suffix(".txt")
        if clustered_label_path.exists():
            return clustered_label_path

    flat_label_path = source.labels_dir / f"{img_path.stem}.txt"
    if flat_label_path.exists():
        return flat_label_path

    matches = sorted(source.labels_dir.rglob(f"{img_path.stem}.txt"))
    return matches[0] if matches else (mirrored_label_path or flat_label_path)


def cluster_name_for_image(img_path: Path, source: YoloDatasetSource) -> str:
    try:
        relative_image_path = img_path.relative_to(source.images_dir)
        if len(relative_image_path.parts) > 1:
            return relative_image_path.parts[0]
    except ValueError:
        pass
    if img_path.parent.name == "images" and img_path.parent.parent.name:
        return img_path.parent.parent.name
    return "unclustered"


def safe_dataset_stem(row: pd.Series, img_src: Path) -> str:
    cluster = str(row.get("cluster", "unclustered"))
    safe_cluster = re.sub(r"[^A-Za-z0-9_.-]+", "_", cluster)
    return f"{row['dataset']}__{safe_cluster}__{img_src.stem}"


def load_source_dataframe(source: YoloDatasetSource) -> tuple[pd.DataFrame, list[str]]:
    if source.images_dir.exists():
        image_roots = [source.images_dir]
    else:
        train_dir = source.root / "train"
        image_roots = sorted([p for p in train_dir.glob("*/images") if p.is_dir()]) if train_dir.exists() else []
    if not image_roots:
        raise FileNotFoundError(
            f"Missing image folder: expected {source.images_dir} or clustered folders like {source.root / 'train' / 'cluster_0' / 'images'}"
        )

    if not source.labels_dir.exists() and not any((p.parent / "labels").exists() for p in image_roots):
        raise FileNotFoundError(
            f"Missing label folder: expected {source.labels_dir} or clustered folders like {source.root / 'train' / 'cluster_0' / 'labels'}"
        )

    image_paths = sorted([p for root in image_roots for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])
    records = []
    for img_path in image_paths:
        label_path = find_source_label_for_image(img_path, source)
        cluster = cluster_name_for_image(img_path, source)
        records.append(
            {
                "dataset": source.name,
                "cluster": cluster,
                "image": str(img_path),
                "label": str(label_path),
                "image_exists": img_path.exists(),
                "label_exists": label_path.exists(),
                "positive": source.positive,
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        return df, [source.class_name]

    matched_df = df[(df["image_exists"]) & (df["label_exists"])].copy().reset_index(drop=True)
    validation_rows = []
    for _, row in matched_df.iterrows():
        valid, classes, errors = validate_yolo_label(Path(row["label"]), nc=1, allow_empty=not source.positive)
        main_class = 0 if source.positive else 1
        validation_rows.append(
            {
                "valid_label": valid,
                "classes": classes,
                "main_class": main_class,
                "label_errors": errors,
            }
        )

    validated_df = pd.concat([matched_df, pd.DataFrame(validation_rows)], axis=1)
    return validated_df, [source.class_name]


def load_clean_dataframe(paths: PathsConfig) -> tuple[pd.DataFrame, list[str]]:
    source_frames = []
    for source in paths.dataset_sources:
        source_df, _ = load_source_dataframe(source)
        source_frames.append(source_df)

    if not source_frames:
        raise ValueError("No dataset sources were configured.")

    validated_df = pd.concat(source_frames, ignore_index=True)
    clean_df = validated_df[validated_df["valid_label"]].copy().reset_index(drop=True)
    if clean_df.empty:
        raise ValueError("No valid images and labels were found in the configured dataset sources.")

    clean_df["group_id"] = clean_df["image"].apply(make_group_id)
    clean_df["fold_label"] = clean_df["main_class"].astype(str) + "__" + clean_df["cluster"].astype(str)
    class_names = ["Bruises"]
    return clean_df, class_names


def create_fold_dataset(
    fold_id: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    class_names: list[str],
    paths: PathsConfig,
    augment_train: bool = True,
    patch_config: PatchConfig | None = None,
) -> tuple[Path, pd.DataFrame]:
    fold_dir = paths.folds_dir / f"fold_{fold_id}"
    if fold_dir.exists():
        shutil.rmtree(fold_dir)
    img_train_dir = fold_dir / "images" / "train"
    img_val_dir = fold_dir / "images" / "val"
    lab_train_dir = fold_dir / "labels" / "train"
    lab_val_dir = fold_dir / "labels" / "val"
    for d in [img_train_dir, img_val_dir, lab_train_dir, lab_val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    patch_rows: list[dict] = []

    def write_split(split_df: pd.DataFrame, split: str, img_dir: Path, lab_dir: Path) -> None:
        for _, row in split_df.iterrows():
            img_src = Path(row["image"])
            lab_src = Path(row["label"])
            dst_stem = safe_dataset_stem(row, img_src)
            dst_name = f"{dst_stem}{img_src.suffix}"
            positive = bool(row["positive"])
            if patch_config and patch_config.enabled:
                rows = write_image_patches(
                    img_src=img_src,
                    lab_src=lab_src,
                    img_dst_dir=img_dir,
                    lab_dst_dir=lab_dir,
                    dst_stem=dst_stem,
                    positive=positive,
                    patch_config=patch_config,
                )
                for patch_row in rows:
                    patch_row.update({"fold": fold_id, "split": split, "dataset": row["dataset"], "cluster": row["cluster"]})
                patch_rows.extend(rows)
            else:
                write_fold_image(
                    img_src=img_src,
                    lab_src=lab_src,
                    img_dst_dir=img_dir,
                    lab_dst_dir=lab_dir,
                    dst_name=dst_name,
                    positive=positive,
                )

    write_split(train_df, "train", img_train_dir, lab_train_dir)
    write_split(val_df, "val", img_val_dir, lab_val_dir)

    if patch_rows:
        paths.reports_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(patch_rows).to_csv(paths.reports_dir / f"fold_{fold_id}_patches.csv", index=False)

    fold_yaml = {
        "path": str(fold_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": class_names,
    }
    yaml_path = fold_dir / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(fold_yaml, f, sort_keys=False)

    augmentation_df = augment_train_split(fold_dir) if augment_train else pd.DataFrame()
    if not augmentation_df.empty:
        paths.reports_dir.mkdir(parents=True, exist_ok=True)
        augmentation_df.to_csv(paths.reports_dir / f"fold_{fold_id}_train_augmentation.csv", index=False)

    return yaml_path, augmentation_df


def build_folds(
    paths: PathsConfig,
    n_splits: int = 5,
    random_state: int = 42,
    augment_train: bool = True,
    patch_config: PatchConfig | None = None,
) -> tuple[list[Path], pd.DataFrame, list[str]]:
    clean_df, class_names = load_clean_dataframe(paths)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    X = clean_df["image"].values
    y = clean_df["fold_label"].values
    groups = clean_df["group_id"].values
    fold_yaml_paths: list[Path] = []
    fold_info: list[dict] = []
    for fold_id, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups), start=1):
        train_df = clean_df.iloc[train_idx].copy().reset_index(drop=True)
        val_df = clean_df.iloc[val_idx].copy().reset_index(drop=True)
        yaml_path, augmentation_df = create_fold_dataset(
            fold_id,
            train_df,
            val_df,
            class_names,
            paths,
            augment_train=augment_train,
            patch_config=patch_config,
        )
        total_train_count = len(list((paths.folds_dir / f"fold_{fold_id}" / "images" / "train").glob("*")))
        train_patch_count = total_train_count - int(len(augmentation_df))
        val_patch_count = len(list((paths.folds_dir / f"fold_{fold_id}" / "images" / "val").glob("*")))
        fold_yaml_paths.append(yaml_path)
        fold_info.append(
            {
                "fold": fold_id,
                "train_images": len(train_df),
                "val_images": len(val_df),
                "train_cutouts": train_patch_count,
                "val_cutouts": val_patch_count,
                "augmented_train_images": int(len(augmentation_df)),
                "total_train_cutouts_after_augmentation": int(total_train_count),
                "train_groups": train_df["group_id"].nunique(),
                "val_groups": val_df["group_id"].nunique(),
                "train_cluster_distribution": train_df["fold_label"].value_counts().sort_index().to_json(),
                "val_cluster_distribution": val_df["fold_label"].value_counts().sort_index().to_json(),
                "yaml_path": str(yaml_path),
            }
        )
    fold_info_df = pd.DataFrame(fold_info)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    fold_info_df.to_csv(paths.reports_dir / "fold_info.csv", index=False)
    return fold_yaml_paths, fold_info_df, class_names


def build_hyperopt_balanced_subset(
    paths: PathsConfig,
    subset_config: HyperoptSubsetConfig,
    patch_config: PatchConfig,
    augment_train: bool = True,
) -> tuple[Path, pd.DataFrame]:
    clean_df, class_names = load_clean_dataframe(paths)
    feature_rows = [image_representative_features(row) for _, row in clean_df.iterrows()]
    featured_df = pd.concat([clean_df.reset_index(drop=True), pd.DataFrame(feature_rows)], axis=1)
    featured_df = featured_df[featured_df["feature_status"] == "ok"].copy().reset_index(drop=True)

    positives = featured_df[featured_df["positive"]].copy()
    negatives = featured_df[~featured_df["positive"]].copy()
    if subset_config.positive_per_cluster is not None and subset_config.negative_per_cluster is not None:
        sampled_positive = _sample_representative_by_cluster(
            positives,
            count_per_cluster=subset_config.positive_per_cluster,
            random_state=subset_config.random_state,
        )
        sampled_negative = _sample_representative_by_cluster(
            negatives,
            count_per_cluster=subset_config.negative_per_cluster,
            random_state=subset_config.random_state + 1000,
        )
    else:
        sampled_positive = _sample_representative_class(
            positives,
            count=subset_config.per_class,
            random_state=subset_config.random_state,
        )
        sampled_negative = _sample_representative_class(
            negatives,
            count=subset_config.per_class,
            random_state=subset_config.random_state + 1,
        )
    subset_df = pd.concat([sampled_positive, sampled_negative], ignore_index=True)
    subset_df = subset_df.sample(frac=1.0, random_state=subset_config.random_state).reset_index(drop=True)

    subset_root = paths.hyperopt_subset_dir
    if subset_root.exists():
        shutil.rmtree(subset_root)
    img_train_dir = subset_root / "images" / "train"
    img_val_dir = subset_root / "images" / "val"
    lab_train_dir = subset_root / "labels" / "train"
    lab_val_dir = subset_root / "labels" / "val"
    for d in [img_train_dir, img_val_dir, lab_train_dir, lab_val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    train_rows: list[pd.DataFrame] = []
    val_rows: list[pd.DataFrame] = []
    val_fraction = min(max(subset_config.val_fraction, 0.05), 0.5)
    for _, class_df in subset_df.groupby(["positive", "cluster"], sort=False):
        val_count = max(1, int(round(len(class_df) * val_fraction)))
        shuffled = class_df.sample(frac=1.0, random_state=subset_config.random_state).reset_index(drop=True)
        val_rows.append(shuffled.iloc[:val_count].copy())
        train_rows.append(shuffled.iloc[val_count:].copy())
    train_df = pd.concat(train_rows, ignore_index=True).sample(frac=1.0, random_state=subset_config.random_state)
    val_df = pd.concat(val_rows, ignore_index=True).sample(frac=1.0, random_state=subset_config.random_state)

    patch_rows: list[dict] = []

    def write_split(split_df: pd.DataFrame, split: str, img_dir: Path, lab_dir: Path) -> None:
        for _, row in split_df.iterrows():
            img_src = Path(row["image"])
            lab_src = Path(row["label"])
            dst_stem = safe_dataset_stem(row, img_src)
            if patch_config.enabled:
                rows = write_image_patches(
                    img_src=img_src,
                    lab_src=lab_src,
                    img_dst_dir=img_dir,
                    lab_dst_dir=lab_dir,
                    dst_stem=dst_stem,
                    positive=bool(row["positive"]),
                    patch_config=patch_config,
                )
                for patch_row in rows:
                    patch_row.update(
                        {
                            "split": split,
                            "dataset": row["dataset"],
                            "cluster": row["cluster"],
                            "source_positive": bool(row["positive"]),
                        }
                    )
                patch_rows.extend(rows)
            else:
                write_fold_image(
                    img_src=img_src,
                    lab_src=lab_src,
                    img_dst_dir=img_dir,
                    lab_dst_dir=lab_dir,
                    dst_name=f"{dst_stem}{img_src.suffix}",
                    positive=bool(row["positive"]),
                )

    write_split(train_df, "train", img_train_dir, lab_train_dir)
    write_split(val_df, "val", img_val_dir, lab_val_dir)

    augmentation_df = augment_train_split(subset_root) if augment_train else pd.DataFrame()
    yaml_path = subset_root / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "path": str(subset_root.resolve()),
                "train": "images/train",
                "val": "images/val",
                "nc": len(class_names),
                "names": class_names,
            },
            f,
            sort_keys=False,
        )

    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    subset_df.to_csv(subset_root / "selected_source_images.csv", index=False)
    if patch_config.enabled:
        pd.DataFrame(patch_rows).to_csv(subset_root / f"generated_{patch_config.size}x{patch_config.size}_cutouts.csv", index=False)
    if not augmentation_df.empty:
        augmentation_df.to_csv(subset_root / "train_augmentation.csv", index=False)

    summary = {
        "source_bruise_images": int(subset_df["positive"].sum()),
        "source_non_bruise_images": int((~subset_df["positive"]).sum()),
        "positive_per_cluster": subset_config.positive_per_cluster,
        "negative_per_cluster": subset_config.negative_per_cluster,
        "source_cluster_distribution": subset_df["fold_label"].value_counts().sort_index().to_json(),
        "train_cluster_distribution": train_df["fold_label"].value_counts().sort_index().to_json(),
        "val_cluster_distribution": val_df["fold_label"].value_counts().sort_index().to_json(),
        "train_source_images": int(len(train_df)),
        "val_source_images": int(len(val_df)),
        "train_images_before_augmentation": int(len(list(img_train_dir.glob("*")))) - int(len(augmentation_df)),
        "val_images": int(len(list(img_val_dir.glob("*")))),
        "augmented_train_images": int(len(augmentation_df)),
        "total_train_images_after_augmentation": int(len(list(img_train_dir.glob("*")))),
        "patch_size": int(patch_config.size),
        "patch_stride": int(patch_config.stride),
        "patches_enabled": bool(patch_config.enabled),
        "train_imgsz": int(patch_config.train_imgsz),
        "selection_note": (
            "Optuna subset is cluster-balanced by default: 50 source images per bruise cluster and "
            "100 source images per non-bruise cluster. Within each cluster, selection is stratified across "
            "brightness, color-tone proxy, difficulty, and bruise box size bins to include varied lighting, "
            "skin tones, sizes, and hard cases."
        ),
        "yaml_path": str(yaml_path),
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(subset_root / "subset_summary.csv", index=False)
    summary_df.to_csv(paths.reports_dir / "hyperopt_balanced_subset_summary.csv", index=False)
    return yaml_path, summary_df
