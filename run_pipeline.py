from __future__ import annotations

import argparse
from pathlib import Path

from src.bruise_detection.config import HyperoptSubsetConfig, PatchConfig
from src.bruise_detection.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RT-DETR bruise detection pipeline")
    parser.add_argument("--dataset-root", type=Path, default=Path(r"R:\Dataset4"), help="Dataset root path")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts"), help="Output folder")
    parser.add_argument("--final-epochs", type=int, default=10, help="Epochs for CV final training")
    parser.add_argument("--n-splits", type=int, default=5, help="Stratified group k-fold count")
    parser.add_argument("--optuna-trials", type=int, default=5, help="Optuna trial count")
    parser.add_argument("--hyperopt-epochs", type=int, default=5, help="Epochs per Optuna trial")
    parser.add_argument(
        "--hyperopt-subset-per-class",
        type=int,
        default=200,
        help="Fallback balanced source images per class for Optuna only; final CV still uses full dataset",
    )
    parser.add_argument(
        "--hyperopt-positive-per-cluster",
        type=int,
        default=50,
        help="Bruise source images per cluster for Optuna only; set -1 to use --hyperopt-subset-per-class",
    )
    parser.add_argument(
        "--hyperopt-negative-per-cluster",
        type=int,
        default=100,
        help="Non-bruise source images per cluster for Optuna only; set -1 to use --hyperopt-subset-per-class",
    )
    parser.add_argument(
        "--hyperopt-subset-val-fraction",
        type=float,
        default=0.2,
        help="Validation fraction inside the balanced Optuna subset",
    )
    parser.add_argument(
        "--no-hyperopt-subset",
        action="store_true",
        help="Disable balanced Optuna subset and use fold 1 for Optuna",
    )
    parser.add_argument("--base-model", type=str, default="rtdetr-l.pt", help="Base RT-DETR model")
    parser.add_argument(
        "--no-physical-augmentation",
        action="store_true",
        help="Keep physical train-only augmentation disabled. This is the default.",
    )
    parser.add_argument(
        "--use-physical-augmentation",
        action="store_true",
        help="Enable physical train-only augmentation after each fold split",
    )
    parser.add_argument(
        "--no-patches",
        action="store_true",
        help="Disable cutout generation and train on full images",
    )
    parser.add_argument(
        "--use-patches",
        action="store_true",
        help="Enable cutout generation; by default the pipeline trains on full images",
    )
    parser.add_argument("--patch-size", type=int, default=32, help="Cutout size in pixels")
    parser.add_argument("--patch-stride", type=int, default=32, help="Cutout stride in pixels")
    parser.add_argument("--train-imgsz", type=int, default=640, help="RT-DETR training image size")
    parser.add_argument(
        "--max-positive-patches-per-image",
        type=int,
        default=50,
        help="Maximum bruise-containing cutouts to keep per bruise source image; use -1 to keep all",
    )
    parser.add_argument(
        "--max-empty-negative-patches-per-image",
        type=int,
        default=10,
        help="Maximum empty cutouts to keep per non-bruise source image; use -1 to keep all",
    )
    parser.add_argument(
        "--skip-optuna",
        action="store_true",
        help="Skip Optuna and reuse artifacts/reports/best_hyperparameters.json",
    )
    parser.add_argument(
        "--reset-optuna",
        action="store_true",
        help="Delete the saved Optuna study database before starting hyperparameter search",
    )
    parser.add_argument(
        "--optuna-study-name",
        type=str,
        default="bruise_hyperopt",
        help="Persistent Optuna study name used for resume",
    )
    parser.add_argument(
        "--resume-run",
        action="store_true",
        help="Keep existing fold/CV outputs and resume interrupted training where possible",
    )
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="bruise-detection-hyperopt", help="W&B project name")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        final_epochs=args.final_epochs,
        n_splits=args.n_splits,
        optuna_trials=args.optuna_trials,
        hyperopt_epochs=args.hyperopt_epochs,
        base_model=args.base_model,
        augment_train=args.use_physical_augmentation and not args.no_physical_augmentation,
        skip_optuna=args.skip_optuna,
        reset_optuna=args.reset_optuna,
        optuna_study_name=args.optuna_study_name,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        resume_run=args.resume_run,
        patch_config=PatchConfig(
            enabled=args.use_patches and not args.no_patches,
            size=args.patch_size,
            stride=args.patch_stride,
            train_imgsz=args.train_imgsz,
            max_positive_patches_per_image=args.max_positive_patches_per_image,
            max_empty_negative_patches_per_image=args.max_empty_negative_patches_per_image,
        ),
        hyperopt_subset_config=HyperoptSubsetConfig(
            enabled=not args.no_hyperopt_subset,
            per_class=args.hyperopt_subset_per_class,
            positive_per_cluster=None if args.hyperopt_positive_per_cluster < 0 else args.hyperopt_positive_per_cluster,
            negative_per_cluster=None if args.hyperopt_negative_per_cluster < 0 else args.hyperopt_negative_per_cluster,
            val_fraction=args.hyperopt_subset_val_fraction,
        ),
    )
