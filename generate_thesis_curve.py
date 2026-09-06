from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_RUN_ROOTS = [
    Path("artifacts/runs/cross_validation"),
    Path("runs/detect/artifacts/runs/cross_validation"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create final per-fold training and validation curves.")
    parser.add_argument("--fold", type=int, default=None, help="Optional single fold number to plot")
    parser.add_argument("--run-dir", type=Path, default=None, help="Optional path to one rtdetr_fold_N run directory")
    parser.add_argument("--output-dir", type=Path, default=Path("Final Result"), help="Output folder for final curves")
    return parser.parse_args()


def find_run_dir(fold: int, explicit_run_dir: Path | None) -> Path:
    if explicit_run_dir is not None:
        return explicit_run_dir

    run_name = f"rtdetr_fold_{fold}"
    for root in DEFAULT_RUN_ROOTS:
        candidate = root / run_name
        if (candidate / "results.csv").exists():
            return candidate

    searched = ", ".join(str(root / run_name) for root in DEFAULT_RUN_ROOTS)
    raise FileNotFoundError(f"Could not find results.csv. Searched: {searched}")


def read_fold_results(fold: int, explicit_run_dir: Path | None = None) -> tuple[Path, pd.DataFrame]:
    run_dir = find_run_dir(fold, explicit_run_dir=explicit_run_dir)
    df = pd.read_csv(run_dir / "results.csv")
    df.columns = [col.strip() for col in df.columns]
    return run_dir, df


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((col for col in candidates if col in df.columns), None)


def epoch_values(df: pd.DataFrame) -> pd.Series:
    if "epoch" in df.columns:
        return pd.to_numeric(df["epoch"], errors="coerce")
    return pd.Series(df.index + 1)


def total_loss(df: pd.DataFrame, prefix: str) -> pd.Series | None:
    cols = [f"{prefix}/giou_loss", f"{prefix}/cls_loss", f"{prefix}/l1_loss"]
    if not all(col in df.columns for col in cols):
        return None
    values = df[cols].apply(pd.to_numeric, errors="coerce")
    return values.sum(axis=1, min_count=len(cols))


def best_map50_95(df: pd.DataFrame) -> float:
    map_col = first_existing_column(
        df,
        ["metrics/mAP50-95(B)", "metrics/mAP50-95", "metrics/mAP50-95_box"],
    )
    if map_col is None:
        return float("nan")
    return float(pd.to_numeric(df[map_col], errors="coerce").max())


def save_loss_curve(df: pd.DataFrame, fold: int, output_path: Path) -> Path:
    epoch = epoch_values(df)
    train_total_loss = total_loss(df, "train")
    val_total_loss = total_loss(df, "val")

    fig, ax = plt.subplots(figsize=(10, 6))
    if train_total_loss is not None:
        ax.plot(epoch, train_total_loss, color="#1f77b4", linewidth=2.2, label="Training loss")
    if val_total_loss is not None and val_total_loss.notna().sum() >= 2:
        ax.plot(epoch, val_total_loss, color="#d62728", linewidth=2.2, label="Validation loss")

    ax.set_title(f"Fold {fold}: Training and Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_validation_performance_curve(df: pd.DataFrame, fold: int, output_path: Path) -> Path:
    epoch = epoch_values(df)
    map_col = first_existing_column(
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
            best_epoch = epoch.iloc[best_idx]
            best_score = map_values.iloc[best_idx]
            ax.scatter(best_epoch, best_score, color="#111827", zorder=3)
            ax.annotate(
                f"Best: {best_score:.3f}",
                xy=(best_epoch, best_score),
                xytext=(8, 8),
                textcoords="offset points",
            )

    ax.set_title(f"Fold {fold}: Validation Performance")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP50-95")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_fold_curves(df: pd.DataFrame, fold: int, output_dir: Path, prefix: str | None = None) -> list[Path]:
    name_prefix = prefix or f"fold_{fold}"
    return [
        save_loss_curve(df, fold, output_dir / f"{name_prefix}_training_validation_loss_curve.png"),
        save_validation_performance_curve(df, fold, output_dir / f"{name_prefix}_validation_performance_curve.png"),
    ]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    if args.fold is not None:
        _, df = read_fold_results(args.fold, explicit_run_dir=args.run_dir)
        saved_paths.extend(save_fold_curves(df, args.fold, args.output_dir))
    else:
        fold_rows = []
        for fold in range(1, 6):
            _, df = read_fold_results(fold)
            saved_paths.extend(save_fold_curves(df, fold, args.output_dir))
            fold_rows.append({"fold": fold, "best_map50_95": best_map50_95(df), "df": df})

        best_row = max(fold_rows, key=lambda row: row["best_map50_95"])
        best_fold = int(best_row["fold"])
        saved_paths.extend(save_fold_curves(best_row["df"], best_fold, args.output_dir, prefix=f"best_fold_{best_fold}"))

    pd.DataFrame({"path": [str(path) for path in saved_paths]}).to_csv(
        args.output_dir / "final_result_curve_manifest.csv",
        index=False,
    )

    print("Saved final result curves:")
    for path in saved_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
