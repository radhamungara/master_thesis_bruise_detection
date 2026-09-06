from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class YoloDatasetSource:
    name: str
    root: Path
    class_name: str
    positive: bool

    @property
    def images_dir(self) -> Path:
        return self.root / "train" / "images"

    @property
    def labels_dir(self) -> Path:
        return self.root / "train" / "labels"

    @property
    def data_yaml(self) -> Path:
        return self.root / "data.yaml"


@dataclass(frozen=True)
class PathsConfig:
    dataset_root: Path
    yolo_root_name: str = "ultralytics yolo"
    images_dir_name: str = "images"
    output_root: Path = Path("artifacts")

    @property
    def images_dir(self) -> Path:
        return self.dataset_root / self.images_dir_name

    @property
    def yolo_root(self) -> Path:
        return self.dataset_root / self.yolo_root_name

    @property
    def data_yaml(self) -> Path:
        return self.yolo_root / "data.yaml"

    @property
    def train_txt(self) -> Path:
        return self.yolo_root / "train.txt"

    @property
    def labels_dir(self) -> Path:
        return self.yolo_root / "labels"

    @property
    def folds_dir(self) -> Path:
        return self.output_root / "fold_datasets"

    @property
    def hyperopt_subset_dir(self) -> Path:
        return self.output_root / "hyperopt_balanced_subset"

    @property
    def runs_dir(self) -> Path:
        return self.output_root / "runs"

    @property
    def reports_dir(self) -> Path:
        return self.output_root / "reports"

    @property
    def final_results_dir(self) -> Path:
        return self.output_root / "Final Result"

    @property
    def dataset_sources(self) -> list[YoloDatasetSource]:
        bruises_root = self.dataset_root / "Bruises.yolov8"
        non_bruises_root = self.dataset_root / "Non-Bruises.yolov8"
        if not bruises_root.exists():
            bruises_root = self.dataset_root / "Bruises"
        if not bruises_root.exists():
            bruises_root = self.dataset_root / "bruise"
        if not non_bruises_root.exists():
            non_bruises_root = self.dataset_root / "Non-Bruises"
        if not non_bruises_root.exists():
            non_bruises_root = self.dataset_root / "non_bruise"
        return [
            YoloDatasetSource(
                name="Bruises",
                root=bruises_root,
                class_name="Bruises",
                positive=True,
            ),
            YoloDatasetSource(
                name="Non-Bruises",
                root=non_bruises_root,
                class_name="Non-Bruises",
                positive=False,
            ),
        ]


@dataclass(frozen=True)
class PatchConfig:
    enabled: bool = False
    size: int = 32
    stride: int = 32
    train_imgsz: int = 640
    min_box_size: int = 2
    keep_empty_positive_patches: bool = False
    keep_empty_negative_patches: bool = True
    max_positive_patches_per_image: int = 50
    max_empty_negative_patches_per_image: int = 10


@dataclass(frozen=True)
class HyperoptSubsetConfig:
    enabled: bool = True
    per_class: int = 200
    positive_per_cluster: Optional[int] = 50
    negative_per_cluster: Optional[int] = 100
    val_fraction: float = 0.2
    random_state: int = 42


HYPERPARAM_CONFIGS = [
    {
        "name": "A_baseline",
        "model": "rtdetr-l.pt",
        "epochs": 10,
        "imgsz": 640,
        "batch": 2,
        "optimizer": "AdamW",
        "lr0": 5e-5,
        "weight_decay": 1e-4,
        "patience": 10,
        "hsv_h": 0.015,
        "hsv_s": 0.4,
        "hsv_v": 0.3,
        "degrees": 2.0,
        "translate": 0.02,
        "scale": 0.10,
        "fliplr": 0.5,
        "mosaic": 0.0,
        "mixup": 0.0,
    },
    {
        "name": "B_lower_lr",
        "model": "rtdetr-l.pt",
        "epochs": 10,
        "imgsz": 640,
        "batch": 2,
        "optimizer": "AdamW",
        "lr0": 2e-5,
        "weight_decay": 1e-4,
        "patience": 10,
        "hsv_h": 0.01,
        "hsv_s": 0.3,
        "hsv_v": 0.2,
        "degrees": 3.0,
        "translate": 0.02,
        "scale": 0.2,
        "fliplr": 0.5,
        "mosaic": 0.0,
        "mixup": 0.0,
    },
    {
        "name": "C_more_aug",
        "model": "rtdetr-l.pt",
        "epochs": 10,
        "imgsz": 640,
        "batch": 2,
        "optimizer": "AdamW",
        "lr0": 5e-5,
        "weight_decay": 1e-4,
        "patience": 10,
        "hsv_h": 0.02,
        "hsv_s": 0.45,
        "hsv_v": 0.35,
        "degrees": 2.0,
        "translate": 0.02,
        "scale": 0.15,
        "fliplr": 0.5,
        "mosaic": 0.0,
        "mixup": 0.0,
    },
]
