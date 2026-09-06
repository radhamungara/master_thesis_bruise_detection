from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import auc, roc_curve
from ultralytics import RTDETR
import yaml
from optuna.importance import get_param_importances
from optuna.trial import TrialState

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None


def train_rtdetr(data_yaml: Path, cfg: dict, project_dir: Path, run_name: str, resume: bool = False) -> None:
    run_dir = project_dir / run_name
    last_weight = run_dir / "weights" / "last.pt"
    alt_last_weight = Path("runs") / "detect" / run_dir / "weights" / "last.pt"
    if resume and last_weight.exists():
        model = RTDETR(str(last_weight))
        model.train(resume=True, verbose=True)
        return
    if resume and alt_last_weight.exists():
        model = RTDETR(str(alt_last_weight))
        model.train(resume=True, verbose=True)
        return

    model = RTDETR(cfg["model"])
    model.train(
        data=str(data_yaml),
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        optimizer=cfg["optimizer"],
        lr0=cfg["lr0"],
        weight_decay=cfg["weight_decay"],
        patience=cfg["patience"],
        hsv_h=cfg["hsv_h"],
        hsv_s=cfg["hsv_s"],
        hsv_v=cfg["hsv_v"],
        degrees=cfg["degrees"],
        translate=cfg["translate"],
        scale=cfg["scale"],
        fliplr=cfg["fliplr"],
        mosaic=cfg["mosaic"],
        mixup=cfg["mixup"],
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        pretrained=True,
        amp=cfg.get("amp", False),
        deterministic=cfg.get("deterministic", False),
        plots=True,
        save=True,
        verbose=True,
    )


def read_results_csv(run_dir: Path) -> pd.DataFrame | None:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def resolve_run_dir(run_dir: Path) -> Path:
    if run_dir.exists():
        return run_dir
    alt = Path("runs") / "detect" / run_dir
    if alt.exists():
        return alt
    return run_dir


def get_best_metrics(run_dir: Path) -> dict:
    run_dir = resolve_run_dir(run_dir)
    df = read_results_csv(run_dir)
    if df is None or len(df) == 0:
        return {"best_epoch": np.nan, "best_map50": np.nan, "best_map50_95": np.nan}
    possible_map95_cols = ["metrics/mAP50-95(B)", "metrics/mAP50-95", "metrics/mAP50-95_box"]
    possible_map50_cols = ["metrics/mAP50(B)", "metrics/mAP50", "metrics/mAP50_box"]
    map95_col = next((c for c in possible_map95_cols if c in df.columns), None)
    map50_col = next((c for c in possible_map50_cols if c in df.columns), None)
    if map95_col is None:
        raise ValueError(f"Could not find mAP50-95 column in {run_dir / 'results.csv'}")
    best_idx = df[map95_col].idxmax()
    return {
        "best_epoch": int(df.loc[best_idx, "epoch"]) if "epoch" in df.columns else int(best_idx),
        "best_map50": float(df.loc[best_idx, map50_col]) if map50_col else np.nan,
        "best_map50_95": float(df.loc[best_idx, map95_col]),
    }


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((col for col in candidates if col in df.columns), None)


def _total_loss(df: pd.DataFrame, prefix: str) -> pd.Series | None:
    cols = [f"{prefix}/giou_loss", f"{prefix}/cls_loss", f"{prefix}/l1_loss"]
    if not all(col in df.columns for col in cols):
        return None
    values = df[cols].apply(pd.to_numeric, errors="coerce")
    return values.sum(axis=1, min_count=len(cols))


def _epoch_values(df: pd.DataFrame) -> pd.Series:
    if "epoch" in df.columns:
        return pd.to_numeric(df["epoch"], errors="coerce")
    return pd.Series(df.index + 1)


def _plot_training_validation_loss(df: pd.DataFrame, fold_id: int, output_path: Path) -> Path:
    epoch = _epoch_values(df)
    train_total_loss = _total_loss(df, "train")
    val_total_loss = _total_loss(df, "val")

    fig, ax = plt.subplots(figsize=(10, 6))
    if train_total_loss is not None:
        ax.plot(epoch, train_total_loss, color="#1f77b4", linewidth=2.2, label="Training loss")
    if val_total_loss is not None and val_total_loss.notna().sum() >= 2:
        ax.plot(epoch, val_total_loss, color="#d62728", linewidth=2.2, label="Validation loss")

    ax.set_title(f"Fold {fold_id}: Training and Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_validation_performance(df: pd.DataFrame, fold_id: int, output_path: Path) -> Path:
    epoch = _epoch_values(df)
    map_col = _first_existing_column(
        df,
        ["metrics/mAP50-95(B)", "metrics/mAP50-95", "metrics/mAP50-95_box"],
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    if map_col is None:
        ax.text(0.5, 0.5, "mAP50-95 column not found", ha="center", va="center")
    else:
        map_values = pd.to_numeric(df[map_col], errors="coerce")
        ax.plot(epoch, map_values, color="#2ca02c", linewidth=2.2, label="Validation mAP50-95")
        if map_values.notna().any():
            best_idx = map_values.idxmax()
            best_epoch = epoch.iloc[best_idx] if hasattr(epoch, "iloc") else epoch[best_idx]
            best_score = map_values.iloc[best_idx]
            ax.scatter(best_epoch, best_score, color="#111827", zorder=3)
            ax.annotate(
                f"Best: {best_score:.3f}",
                xy=(best_epoch, best_score),
                xytext=(8, 8),
                textcoords="offset points",
            )

    ax.set_title(f"Fold {fold_id}: Validation Performance")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP50-95")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_final_result_curves(cv_df: pd.DataFrame, final_results_dir: Path) -> list[Path]:
    """Save separate per-fold curves and the best fold curves for final reporting."""
    final_results_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    manifest_rows: list[dict] = []

    if cv_df.empty or "fold" not in cv_df.columns or "run_dir" not in cv_df.columns:
        return saved_paths

    for _, row in cv_df.sort_values("fold").iterrows():
        fold_id = int(row["fold"])
        run_dir = resolve_run_dir(Path(row["run_dir"]))
        results_df = read_results_csv(run_dir)
        if results_df is None or results_df.empty:
            continue

        loss_path = _plot_training_validation_loss(
            results_df,
            fold_id,
            final_results_dir / f"fold_{fold_id}_training_validation_loss_curve.png",
        )
        performance_path = _plot_validation_performance(
            results_df,
            fold_id,
            final_results_dir / f"fold_{fold_id}_validation_performance_curve.png",
        )
        saved_paths.extend([loss_path, performance_path])
        manifest_rows.extend(
            [
                {"fold": fold_id, "curve_type": "training_validation_loss", "path": str(loss_path)},
                {"fold": fold_id, "curve_type": "validation_performance", "path": str(performance_path)},
            ]
        )

    if "best_map50_95" in cv_df.columns and cv_df["best_map50_95"].notna().any():
        best_row = cv_df.loc[pd.to_numeric(cv_df["best_map50_95"], errors="coerce").idxmax()]
        best_fold = int(best_row["fold"])
        best_run_dir = resolve_run_dir(Path(best_row["run_dir"]))
        best_results_df = read_results_csv(best_run_dir)
        if best_results_df is not None and not best_results_df.empty:
            best_loss_path = _plot_training_validation_loss(
                best_results_df,
                best_fold,
                final_results_dir / f"best_fold_{best_fold}_training_validation_loss_curve.png",
            )
            best_performance_path = _plot_validation_performance(
                best_results_df,
                best_fold,
                final_results_dir / f"best_fold_{best_fold}_validation_performance_curve.png",
            )
            saved_paths.extend([best_loss_path, best_performance_path])
            manifest_rows.extend(
                [
                    {"fold": best_fold, "curve_type": "best_training_validation_loss", "path": str(best_loss_path)},
                    {"fold": best_fold, "curve_type": "best_validation_performance", "path": str(best_performance_path)},
                ]
            )

    if manifest_rows:
        pd.DataFrame(manifest_rows).to_csv(final_results_dir / "final_result_curve_manifest.csv", index=False)

    return saved_paths


def save_fold_validation_graphs(run_dir: Path, analysis_dir: Path, fold_id: int) -> list[Path]:
    """Save validation graphs for one cross-validation fold."""
    run_dir = resolve_run_dir(run_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    results_df = read_results_csv(run_dir)
    if results_df is not None and not results_df.empty:
        x = results_df["epoch"] if "epoch" in results_df.columns else results_df.index

        metric_cols = [
            c
            for c in [
                "metrics/precision(B)",
                "metrics/recall(B)",
                "metrics/mAP50(B)",
                "metrics/mAP50-95(B)",
                "metrics/precision",
                "metrics/recall",
                "metrics/mAP50",
                "metrics/mAP50-95",
            ]
            if c in results_df.columns
        ]
        if metric_cols:
            fig, ax = plt.subplots(figsize=(10, 6))
            for col in metric_cols:
                ax.plot(x, results_df[col], marker="o", linewidth=1.8, label=col.replace("metrics/", ""))
            ax.set_title(f"Fold {fold_id} Validation Metrics")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Score")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            out_path = analysis_dir / f"fold_{fold_id}_validation_metrics.png"
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
            saved_paths.append(out_path)

        loss_cols = [
            c
            for c in results_df.columns
            if c.startswith("train/") or c.startswith("val/")
        ]
        if loss_cols:
            fig, ax = plt.subplots(figsize=(10, 6))
            for col in loss_cols:
                ax.plot(x, results_df[col], linewidth=1.6, label=col)
            ax.set_title(f"Fold {fold_id} Training and Validation Losses")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            out_path = analysis_dir / f"fold_{fold_id}_loss_curves.png"
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
            saved_paths.append(out_path)

    roc_csv = analysis_dir / "roc_curve.csv"
    if roc_csv.exists():
        roc_df = pd.read_csv(roc_csv)
        if {"fpr", "tpr"}.issubset(roc_df.columns) and not roc_df.empty:
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.plot(roc_df["fpr"], roc_df["tpr"], linewidth=2, label="ROC")
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
            ax.set_title(f"Fold {fold_id} ROC Curve")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            out_path = analysis_dir / f"fold_{fold_id}_roc_curve.png"
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
            saved_paths.append(out_path)

    summary_json = analysis_dir / "detection_summary.json"
    if summary_json.exists():
        summary = json.loads(summary_json.read_text(encoding="utf-8"))
        matrix = np.array([[summary.get("tn", 0), summary.get("fp", 0)], [summary.get("fn", 0), summary.get("tp", 0)]])
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_title(f"Fold {fold_id} Confusion Matrix")
        ax.set_xticks([0, 1], labels=["Pred No Bruise", "Pred Bruise"])
        ax.set_yticks([0, 1], labels=["True No Bruise", "True Bruise"])
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                ax.text(col, row, int(matrix[row, col]), ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        out_path = analysis_dir / f"fold_{fold_id}_confusion_matrix.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        saved_paths.append(out_path)

    image_csv = analysis_dir / "image_level_analysis.csv"
    if image_csv.exists():
        image_df = pd.read_csv(image_csv)
        if "max_conf" in image_df.columns and not image_df.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(image_df["max_conf"].fillna(0.0), bins=20, color="#2563eb", edgecolor="white")
            ax.set_title(f"Fold {fold_id} Prediction Confidence Distribution")
            ax.set_xlabel("Max Prediction Confidence")
            ax.set_ylabel("Image Count")
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            out_path = analysis_dir / f"fold_{fold_id}_confidence_histogram.png"
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
            saved_paths.append(out_path)

    return saved_paths


def save_cross_validation_summary_graphs(cv_df: pd.DataFrame, reports_dir: Path) -> list[Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    if cv_df.empty or "fold" not in cv_df.columns:
        return saved_paths

    fold_labels = [f"Fold {int(f)}" for f in cv_df["fold"]]

    metric_cols = [c for c in ["best_map50", "best_map50_95", "precision", "recall", "roc_auc"] if c in cv_df.columns]
    metric_cols = [c for c in metric_cols if pd.to_numeric(cv_df[c], errors="coerce").notna().any()]
    if metric_cols:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(cv_df))
        width = 0.8 / len(metric_cols)
        for idx, col in enumerate(metric_cols):
            values = pd.to_numeric(cv_df[col], errors="coerce").fillna(0.0)
            ax.bar(x + idx * width, values, width=width, label=col)
        ax.set_title("Cross-Validation Metrics by Fold")
        ax.set_xlabel("Fold")
        ax.set_ylabel("Score")
        ax.set_xticks(x + width * (len(metric_cols) - 1) / 2, fold_labels)
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out_path = reports_dir / "cross_validation_metric_summary.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        saved_paths.append(out_path)

    if "best_epoch" in cv_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        values = pd.to_numeric(cv_df["best_epoch"], errors="coerce").fillna(0.0)
        ax.bar(fold_labels, values, color="#0f766e")
        ax.set_title("Best Epoch by Fold")
        ax.set_xlabel("Fold")
        ax.set_ylabel("Best Epoch")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        out_path = reports_dir / "cross_validation_best_epoch.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths


def sample_hyperparams(trial: optuna.Trial, base_model: str, epochs: int, imgsz: int = 640) -> dict:
    return {
        "name": f"optuna_trial_{trial.number}",
        "model": base_model,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": trial.suggest_categorical("batch", [1, 2]),
        "optimizer": "AdamW",
        "lr0": trial.suggest_float("lr0", 1e-5, 1e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 5e-4, log=True),
        "patience": trial.suggest_int("patience", 8, 15),
        "hsv_h": trial.suggest_float("hsv_h", 0.0, 0.02),
        "hsv_s": trial.suggest_float("hsv_s", 0.2, 0.45),
        "hsv_v": trial.suggest_float("hsv_v", 0.1, 0.35),
        "degrees": trial.suggest_float("degrees", 0.0, 2.0),
        "translate": trial.suggest_float("translate", 0.0, 0.02),
        "scale": trial.suggest_float("scale", 0.05, 0.15),
        "fliplr": trial.suggest_float("fliplr", 0.0, 0.5),
        # Keep these settings conservative for stable RT-DETR training on small cutouts.
        "amp": False,
        "deterministic": False,
        "mosaic": 0.0,
        "mixup": 0.0,
    }


def save_best_config(best_cfg: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(best_cfg, indent=2), encoding="utf-8")


def load_best_config(input_path: Path, final_epochs: int | None = None) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing saved best config: {input_path}")
    cfg = json.loads(input_path.read_text(encoding="utf-8"))
    if final_epochs is not None:
        cfg["epochs"] = final_epochs
    return cfg


def _storage_url(storage_path: Path) -> str:
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{storage_path.resolve().as_posix()}"


def _completed_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return [trial for trial in study.trials if trial.state == TrialState.COMPLETE]


def _trial_row(trial: optuna.trial.FrozenTrial, base_model: str, epochs: int) -> dict:
    params = trial.params
    attrs = trial.user_attrs
    row = {
        "trial": trial.number,
        "model": attrs.get("model", base_model),
        "epochs": attrs.get("epochs", epochs),
        "imgsz": attrs.get("imgsz", 640),
        "batch": params.get("batch", attrs.get("batch", np.nan)),
        "optimizer": params.get("optimizer", attrs.get("optimizer", "")),
        "lr0": params.get("lr0", attrs.get("lr0", np.nan)),
        "weight_decay": params.get("weight_decay", attrs.get("weight_decay", np.nan)),
        "patience": params.get("patience", attrs.get("patience", np.nan)),
        "hsv_h": params.get("hsv_h", attrs.get("hsv_h", np.nan)),
        "hsv_s": params.get("hsv_s", attrs.get("hsv_s", np.nan)),
        "hsv_v": params.get("hsv_v", attrs.get("hsv_v", np.nan)),
        "degrees": params.get("degrees", attrs.get("degrees", np.nan)),
        "translate": params.get("translate", attrs.get("translate", np.nan)),
        "scale": params.get("scale", attrs.get("scale", np.nan)),
        "fliplr": params.get("fliplr", attrs.get("fliplr", np.nan)),
        "amp": attrs.get("amp", False),
        "deterministic": attrs.get("deterministic", False),
        "mosaic": attrs.get("mosaic", 0.0),
        "mixup": attrs.get("mixup", 0.0),
        "best_epoch": attrs.get("best_epoch", np.nan),
        "best_map50": attrs.get("best_map50", np.nan),
        "best_map50_95": attrs.get("best_map50_95", trial.value if trial.value is not None else np.nan),
        "status": attrs.get("status", "ok" if trial.value is not None else "failed"),
        "error": attrs.get("error", ""),
    }
    return row


def study_trials_dataframe(study: optuna.Study, base_model: str, epochs: int) -> pd.DataFrame:
    rows = [_trial_row(trial, base_model=base_model, epochs=epochs) for trial in _completed_trials(study)]
    if not rows:
        return pd.DataFrame()
    trials_df = pd.DataFrame(rows)
    trials_df["best_map50_95"] = pd.to_numeric(trials_df["best_map50_95"], errors="coerce")
    return trials_df.sort_values("best_map50_95", ascending=False, na_position="last").reset_index(drop=True)


def best_config_from_trials(trials_df: pd.DataFrame, base_model: str, epochs: int) -> dict:
    if trials_df.empty:
        raise RuntimeError("No completed Optuna trials are available for selecting best hyperparameters.")

    valid_df = trials_df[(trials_df.get("status", "failed") == "ok") & (trials_df["best_map50_95"].fillna(0) > 0)]
    if valid_df.empty:
        best_row = trials_df.iloc[0]
    else:
        best_row = valid_df.iloc[0]

    return {
        "name": f"optuna_trial_{int(best_row['trial'])}",
        "model": str(best_row.get("model", base_model)),
        "epochs": epochs,
        "imgsz": int(best_row.get("imgsz", 640)),
        "batch": int(best_row["batch"]),
        "optimizer": str(best_row["optimizer"]),
        "lr0": float(best_row["lr0"]),
        "weight_decay": float(best_row["weight_decay"]),
        "patience": int(best_row["patience"]),
        "hsv_h": float(best_row["hsv_h"]),
        "hsv_s": float(best_row["hsv_s"]),
        "hsv_v": float(best_row["hsv_v"]),
        "degrees": float(best_row["degrees"]),
        "translate": float(best_row["translate"]),
        "scale": float(best_row["scale"]),
        "fliplr": float(best_row["fliplr"]),
        "amp": bool(best_row.get("amp", False)),
        "deterministic": bool(best_row.get("deterministic", False)),
        "mosaic": float(best_row["mosaic"]),
        "mixup": float(best_row["mixup"]),
    }


def has_valid_checkpoint(run_dir: Path) -> bool:
    run_dir = resolve_run_dir(run_dir)
    best_weight = run_dir / "weights" / "best.pt"
    last_weight = run_dir / "weights" / "last.pt"
    return best_weight.exists() and best_weight.stat().st_size > 0 and last_weight.exists() and last_weight.stat().st_size > 0


def optimize_hyperparameters(
    data_yaml: Path,
    project_dir: Path,
    n_trials: int,
    epochs: int,
    base_model: str = "rtdetr-l.pt",
    use_wandb: bool = False,
    wandb_project: str = "bruise-detection-hyperopt",
    storage_path: Path | None = None,
    study_name: str = "bruise_hyperopt",
    reset_study: bool = False,
    progress_csv_path: Path | None = None,
    best_config_path: Path | None = None,
    imgsz: int = 640,
) -> tuple[dict, pd.DataFrame, optuna.Study]:
    project_dir.mkdir(parents=True, exist_ok=True)

    storage_path = storage_path or project_dir / "optuna_study.db"
    progress_csv_path = progress_csv_path or project_dir / "hyperparameter_optimization_results.csv"
    best_config_path = best_config_path or project_dir / "best_hyperparameters.json"
    if reset_study and storage_path.exists():
        storage_path.unlink()

    def objective(trial: optuna.Trial) -> float:
        cfg = sample_hyperparams(trial, base_model=base_model, epochs=epochs, imgsz=imgsz)
        run_name = f"optuna_trial_{trial.number}"
        for key, value in cfg.items():
            trial.set_user_attr(key, value)
        try:
            train_rtdetr(data_yaml=data_yaml, cfg=cfg, project_dir=project_dir, run_name=run_name)
            run_dir = resolve_run_dir(project_dir / run_name)
            metrics = get_best_metrics(run_dir)
            raw_score = metrics["best_map50_95"]
            score = float(raw_score) if raw_score is not None and np.isfinite(raw_score) else 0.0
            checkpoint_ok = has_valid_checkpoint(run_dir)
            status = "ok" if score > 0 and checkpoint_ok else "failed"
            for key, value in metrics.items():
                trial.set_user_attr(key, value)
            trial.set_user_attr("checkpoint_ok", checkpoint_ok)
            trial.set_user_attr("status", status)
            if status == "ok":
                trial.set_user_attr("error", "")
            elif not checkpoint_ok:
                trial.set_user_attr("error", "missing_or_invalid_checkpoint")
                score = 0.0
            else:
                trial.set_user_attr("error", "nan_or_zero_score")
            if use_wandb and wandb is not None:
                wb = wandb.init(project=wandb_project, name=run_name, reinit=True, config=cfg)
                wb.log({"best_map50_95": score, "best_map50": metrics["best_map50"], "best_epoch": metrics["best_epoch"]})
                wb.finish()
            return score
        except Exception as e:
            err = str(e)
            trial.set_user_attr("error", err)
            trial.set_user_attr("status", "failed")
            trial.set_user_attr("best_epoch", np.nan)
            trial.set_user_attr("best_map50", np.nan)
            trial.set_user_attr("best_map50_95", np.nan)
            return 0.0

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=_storage_url(storage_path),
        load_if_exists=True,
    )
    completed_before = len(_completed_trials(study))
    remaining_trials = max(0, n_trials - completed_before)
    print(f"Optuna study: {study_name}")
    print(f"Optuna storage: {storage_path}")
    print(f"Completed Optuna trials: {completed_before}/{n_trials}")
    print(f"New Optuna trials to run: {remaining_trials}")

    def save_progress(study: optuna.Study, _: optuna.trial.FrozenTrial) -> None:
        trials_df = study_trials_dataframe(study, base_model=base_model, epochs=epochs)
        if trials_df.empty:
            return
        progress_csv_path.parent.mkdir(parents=True, exist_ok=True)
        trials_df.to_csv(progress_csv_path, index=False)
        best_cfg = best_config_from_trials(trials_df, base_model=base_model, epochs=epochs)
        save_best_config(best_cfg, best_config_path)

    if remaining_trials:
        study.optimize(objective, n_trials=remaining_trials, catch=(Exception,), callbacks=[save_progress])

    trials_df = study_trials_dataframe(study, base_model=base_model, epochs=epochs)
    if trials_df.empty:
        raise RuntimeError("No trials were recorded. Check training logs for failures.")

    best_cfg = best_config_from_trials(trials_df, base_model=base_model, epochs=epochs)
    progress_csv_path.parent.mkdir(parents=True, exist_ok=True)
    trials_df.to_csv(progress_csv_path, index=False)
    save_best_config(best_cfg, best_config_path)
    return best_cfg, trials_df, study


def build_hyperopt_report_tables(trials_df: pd.DataFrame, study: optuna.Study | None = None) -> dict[str, pd.DataFrame]:
    if trials_df.empty:
        return {
            "best_params_table": pd.DataFrame(),
            "trial_ranking_table": pd.DataFrame(),
            "param_importance_table": pd.DataFrame(),
        }

    metric_col = "best_map50_95"
    best_row = trials_df.iloc[0].copy()
    best_score = float(best_row[metric_col])

    excluded_cols = {"trial", "best_epoch", "best_map50", "best_map50_95"}
    param_cols = [c for c in trials_df.columns if c not in excluded_cols]

    best_params_table = pd.DataFrame(
        [{"parameter": c, "best_value": best_row[c]} for c in param_cols]
    )

    trial_ranking_table = trials_df[["trial", "best_epoch", "best_map50", "best_map50_95"] + param_cols].copy()
    trial_ranking_table["delta_to_best_map50_95"] = best_score - trial_ranking_table["best_map50_95"]
    trial_ranking_table = trial_ranking_table.sort_values("best_map50_95", ascending=False).reset_index(drop=True)

    if study is not None:
        try:
            importances = get_param_importances(study)
            param_importance_table = pd.DataFrame(
                [{"parameter": k, "importance": float(v)} for k, v in importances.items()]
            ).sort_values("importance", ascending=False)
        except Exception:
            param_importance_table = pd.DataFrame(
                [{"parameter": "N/A", "importance": np.nan, "note": "importance unavailable (zero trial variance)"}]
            )
    else:
        param_importance_table = pd.DataFrame(columns=["parameter", "importance"])

    return {
        "best_params_table": best_params_table,
        "trial_ranking_table": trial_ranking_table,
        "param_importance_table": param_importance_table,
    }


def _read_fold_yaml(fold_yaml: Path) -> dict:
    with fold_yaml.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_label_file(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    if not label_path.exists():
        return rows
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) == 5:
            cls, x, y, w, h = parts
            rows.append((int(float(cls)), float(x), float(y), float(w), float(h)))
        elif len(parts) >= 7 and len(parts[1:]) % 2 == 0:
            cls = int(float(parts[0]))
            values = [float(v) for v in parts[1:]]
            xs = values[0::2]
            ys = values[1::2]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            rows.append((cls, (x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min))
    return rows


def _yolo_to_xyxy(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    cx, cy, bw, bh = x * img_w, y * img_h, w * img_w, h * img_h
    return cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def evaluate_detection_analysis(
    model_path: Path,
    fold_yaml: Path,
    output_dir: Path,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
    n_clusters: int = 3,
    imgsz: int = 640,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = _read_fold_yaml(fold_yaml)
    fold_root = Path(cfg["path"])
    val_img_dir = fold_root / "images" / "val"
    val_lbl_dir = fold_root / "labels" / "val"
    image_paths = sorted([p for p in val_img_dir.glob("*") if p.is_file()])
    model = RTDETR(str(model_path))

    y_true: list[int] = []
    y_score: list[float] = []
    tp = tn = fp = fn = 0
    image_rows = []

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        labels = _parse_label_file(val_lbl_dir / f"{img_path.stem}.txt")
        gt_boxes = [_yolo_to_xyxy(x, y, bw, bh, w, h) for _, x, y, bw, bh in labels]
        gt_present = int(len(gt_boxes) > 0)

        pred = model.predict(source=str(img_path), imgsz=imgsz, conf=0.001, verbose=False)[0]
        pred_boxes = pred.boxes.xyxy.cpu().numpy().tolist() if pred.boxes is not None and len(pred.boxes) else []
        pred_confs = pred.boxes.conf.cpu().numpy().tolist() if pred.boxes is not None and len(pred.boxes) else []
        max_conf = float(max(pred_confs)) if pred_confs else 0.0
        pred_present = int(any(c >= conf_threshold for c in pred_confs))

        matched = False
        best_iou = 0.0
        for gb in gt_boxes:
            for pb, pc in zip(pred_boxes, pred_confs):
                if pc < conf_threshold:
                    continue
                iou = _iou(gb, tuple(pb))
                best_iou = max(best_iou, iou)
                if iou >= iou_threshold:
                    matched = True
                    break
            if matched:
                break

        if gt_present and matched:
            tp += 1
        elif gt_present and not pred_present:
            fn += 1
        elif not gt_present and pred_present:
            fp += 1
        elif not gt_present and not pred_present:
            tn += 1
        elif gt_present and pred_present and not matched:
            fn += 1
            fp += 1

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        gt_area_ratio = 0.0
        if gt_boxes:
            areas = [max(0.0, (b[2] - b[0]) * (b[3] - b[1])) for b in gt_boxes]
            gt_area_ratio = float(sum(areas) / (w * h))

        y_true.append(gt_present)
        y_score.append(max_conf)
        image_rows.append(
            {
                "image": str(img_path),
                "gt_present": gt_present,
                "pred_present": pred_present,
                "matched_iou": int(matched),
                "best_iou": best_iou,
                "max_conf": max_conf,
                "brightness": brightness,
                "img_w": w,
                "img_h": h,
                "gt_count": len(gt_boxes),
                "gt_area_ratio": gt_area_ratio,
            }
        )

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    roc_auc = np.nan
    roc_path = output_dir / "roc_curve.csv"
    if len(set(y_true)) > 1:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}).to_csv(roc_path, index=False)

    image_df = pd.DataFrame(image_rows)
    image_df.to_csv(output_dir / "image_level_analysis.csv", index=False)

    if len(image_df) >= n_clusters:
        feat_cols = ["brightness", "img_w", "img_h", "gt_count", "gt_area_ratio"]
        X = image_df[feat_cols].fillna(0.0).values
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        image_df["cluster"] = kmeans.fit_predict(X)
        cluster_summary = (
            image_df.groupby("cluster")
            .agg(
                samples=("image", "count"),
                match_rate=("matched_iou", "mean"),
                avg_iou=("best_iou", "mean"),
                avg_conf=("max_conf", "mean"),
                avg_brightness=("brightness", "mean"),
                avg_gt_count=("gt_count", "mean"),
                avg_gt_area_ratio=("gt_area_ratio", "mean"),
            )
            .reset_index()
        )
        cluster_summary.to_csv(output_dir / "cluster_performance_summary.csv", index=False)
        image_df.to_csv(output_dir / "image_level_analysis_with_clusters.csv", index=False)

    summary = {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "roc_auc": None if np.isnan(roc_auc) else float(roc_auc),
        "iou_threshold": iou_threshold,
        "conf_threshold": conf_threshold,
        "images_evaluated": int(len(image_df)),
    }
    (output_dir / "detection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
