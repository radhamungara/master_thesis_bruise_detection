from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_RESULTS_ROOT = Path("fold_training_backup/runs/detect/artifacts/runs/cross_validation")
DEFAULT_OUTPUT_DIR = Path("final_result_download")


def total_loss(df: pd.DataFrame, prefix: str) -> pd.Series:
    cols = [f"{prefix}/giou_loss", f"{prefix}/cls_loss", f"{prefix}/l1_loss"]
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing loss columns: {missing}")
    return df[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)


def read_fold_curves(results_root: Path) -> pd.DataFrame:
    rows = []
    for results_csv in sorted(results_root.glob("rtdetr_fold_*/results.csv")):
        fold_name = results_csv.parent.name
        fold_id = int(fold_name.rsplit("_", 1)[-1])
        df = pd.read_csv(results_csv)
        df.columns = [col.strip() for col in df.columns]

        map_col = next(
            (
                col
                for col in ["metrics/mAP50-95(B)", "metrics/mAP50-95", "metrics/mAP50-95_box"]
                if col in df.columns
            ),
            None,
        )
        if map_col is None:
            raise ValueError(f"Could not find mAP50-95 column in {results_csv}")

        epoch = pd.to_numeric(df["epoch"], errors="coerce") if "epoch" in df.columns else pd.Series(df.index + 1)
        fold_df = pd.DataFrame(
            {
                "fold": fold_id,
                "epoch": epoch,
                "training_loss": total_loss(df, "train"),
                "validation_loss": total_loss(df, "val"),
                "validation_map50_95": pd.to_numeric(df[map_col], errors="coerce"),
            }
        )
        rows.append(fold_df)

    if not rows:
        raise FileNotFoundError(f"No fold results.csv files found under {results_root}")
    return pd.concat(rows, ignore_index=True)


def summarize_by_epoch(curves_df: pd.DataFrame) -> pd.DataFrame:
    return (
        curves_df.groupby("epoch", as_index=False)
        .agg(
            training_loss_mean=("training_loss", "mean"),
            training_loss_std=("training_loss", "std"),
            validation_loss_mean=("validation_loss", "mean"),
            validation_loss_std=("validation_loss", "std"),
            validation_map50_95_mean=("validation_map50_95", "mean"),
            validation_map50_95_std=("validation_map50_95", "std"),
        )
        .fillna(0.0)
    )


def save_thesis_curve(summary_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epoch = summary_df["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax_loss = axes[0]
    ax_loss.plot(epoch, summary_df["training_loss_mean"], color="#1f77b4", linewidth=2.6, label="Training loss")
    ax_loss.fill_between(
        epoch,
        summary_df["training_loss_mean"] - summary_df["training_loss_std"],
        summary_df["training_loss_mean"] + summary_df["training_loss_std"],
        color="#1f77b4",
        alpha=0.12,
    )
    ax_loss.plot(epoch, summary_df["validation_loss_mean"], color="#d62728", linewidth=2.6, label="Validation loss")
    ax_loss.fill_between(
        epoch,
        summary_df["validation_loss_mean"] - summary_df["validation_loss_std"],
        summary_df["validation_loss_mean"] + summary_df["validation_loss_std"],
        color="#d62728",
        alpha=0.12,
    )
    ax_loss.set_title("Training vs. Validation Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Total loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    ax_map = axes[1]
    ax_map.plot(
        epoch,
        summary_df["validation_map50_95_mean"],
        color="#2ca02c",
        linewidth=2.8,
        label="Validation mAP50-95",
    )
    ax_map.fill_between(
        epoch,
        summary_df["validation_map50_95_mean"] - summary_df["validation_map50_95_std"],
        summary_df["validation_map50_95_mean"] + summary_df["validation_map50_95_std"],
        color="#2ca02c",
        alpha=0.12,
    )
    best_idx = summary_df["validation_map50_95_mean"].idxmax()
    best_epoch = summary_df.loc[best_idx, "epoch"]
    best_map = summary_df.loc[best_idx, "validation_map50_95_mean"]
    ax_map.scatter(best_epoch, best_map, color="#111827", zorder=4)
    ax_map.annotate(
        f"Best mean mAP50-95: {best_map:.3f}\nEpoch {int(best_epoch)}",
        xy=(best_epoch, best_map),
        xytext=(10, -35),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#111827"},
    )
    ax_map.set_title("Validation Performance Over Epochs")
    ax_map.set_xlabel("Epoch")
    ax_map.set_ylabel("mAP50-95")
    ax_map.set_ylim(bottom=0)
    ax_map.grid(True, alpha=0.3)
    ax_map.legend()

    final_train = summary_df["training_loss_mean"].iloc[-1]
    final_val = summary_df["validation_loss_mean"].iloc[-1]
    note = (
        "Interpretation: validation mAP increases and then plateaus, showing learning. "
        "The gap between training and validation loss indicates mild overfitting after the main improvement phase."
        if final_val > final_train * 1.15
        else "Interpretation: training and validation curves improve together, indicating stable learning."
    )
    fig.suptitle("RT-DETR Bruise Detection: Training and Validation Curve", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.01, note, ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    curves_df = read_fold_curves(DEFAULT_RESULTS_ROOT)
    summary_df = summarize_by_epoch(curves_df)

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_csv = DEFAULT_OUTPUT_DIR / "thesis_training_validation_summary.csv"
    output_png = DEFAULT_OUTPUT_DIR / "thesis_training_validation_curve.png"

    summary_df.to_csv(summary_csv, index=False)
    save_thesis_curve(summary_df, output_png)

    print(f"Saved thesis curve: {output_png}")
    print(f"Saved curve data: {summary_csv}")


if __name__ == "__main__":
    main()
