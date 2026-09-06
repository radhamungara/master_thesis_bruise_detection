from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


RESULTS_ROOT = Path("runs/detect/artifacts/runs/cross_validation")
OUTPUT_PATH = Path("/home/radha_mungara/best_fold_all_4_metrics_curve.png")

METRIC_COLUMNS = {
    "Precision": ["metrics/precision(B)", "metrics/precision"],
    "Recall": ["metrics/recall(B)", "metrics/recall"],
    "mAP50": ["metrics/mAP50(B)", "metrics/mAP50"],
    "mAP50-95": ["metrics/mAP50-95(B)", "metrics/mAP50-95"],
}


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"Missing metric column. Tried: {candidates}")


def main() -> None:
    best_fold = None
    best_score = -1.0
    best_df = None

    for results_csv in sorted(RESULTS_ROOT.glob("rtdetr_fold_*/results.csv")):
        df = pd.read_csv(results_csv)
        df.columns = [column.strip() for column in df.columns]
        map_col = find_column(df, METRIC_COLUMNS["mAP50-95"])
        score = pd.to_numeric(df[map_col], errors="coerce").max()

        if score > best_score:
            best_score = float(score)
            best_fold = results_csv.parent.name
            best_df = df

    if best_df is None or best_fold is None:
        raise FileNotFoundError(f"No fold results.csv files found under {RESULTS_ROOT}")

    epoch = best_df["epoch"] if "epoch" in best_df.columns else range(1, len(best_df) + 1)

    plt.figure(figsize=(10, 6))
    for label, candidates in METRIC_COLUMNS.items():
        col = find_column(best_df, candidates)
        values = pd.to_numeric(best_df[col], errors="coerce")
        plt.plot(epoch, values, linewidth=2.2, label=label)

    plt.title(f"{best_fold}: Validation Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Metric value")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300)

    print(f"Best fold: {best_fold}")
    print(f"Best mAP50-95: {best_score:.4f}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
