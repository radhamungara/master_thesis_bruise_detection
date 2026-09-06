from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "gt_present",
    "pred_present",
    "matched_iou",
    "best_iou",
    "max_conf",
    "gt_count",
]


def load_fold_image_level_records(reports_root: Path) -> pd.DataFrame:
    frames = []
    for fold_dir in sorted(reports_root.glob("fold_*_analysis")):
        csv_path = fold_dir / "image_level_analysis.csv"
        if not csv_path.exists():
            continue
        fold = int(fold_dir.name.split("_")[1])
        frame = pd.read_csv(csv_path)
        frame.insert(0, "fold", fold)
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No fold image_level_analysis.csv files found under {reports_root}")
    return pd.concat(frames, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate RT-DETR bruise-only post-evaluation analysis-cluster failure plot."
    )
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--out-png", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--title", default="RT-DETR-Large Post-Evaluation Analysis-Cluster Localisation Failures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_fold_image_level_records(args.reports_root)
    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    feature_df = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    scaled = StandardScaler().fit_transform(feature_df)
    df["analysis_cluster"] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(scaled)

    bruise_df = df[pd.to_numeric(df["gt_present"], errors="coerce").fillna(0).astype(int) == 1].copy()
    bruise_df["localisation_failure"] = (
        pd.to_numeric(bruise_df["matched_iou"], errors="coerce").fillna(0.0) <= 0.0
    )

    summary = (
        bruise_df.groupby("analysis_cluster")
        .agg(
            bruise_images=("image", "count"),
            localisation_failures=("localisation_failure", "sum"),
            avg_best_iou=("best_iou", "mean"),
            avg_confidence=("max_conf", "mean"),
        )
        .reset_index()
        .sort_values("analysis_cluster")
    )
    summary["failure_frequency_percent"] = (
        summary["localisation_failures"] / summary["bruise_images"] * 100.0
    )

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_csv, index=False)

    labels = [
        f"Analysis cluster {int(row.analysis_cluster)}\n"
        f"failures={int(row.localisation_failures)}/{int(row.bruise_images)}\n"
        f"mean IoU={row.avg_best_iou:.2f}"
        for row in summary.itertuples(index=False)
    ]

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 18, "axes.labelsize": 13})
    fig, ax = plt.subplots(figsize=(12, 7), dpi=200)
    bars = ax.bar(
        labels,
        summary["failure_frequency_percent"],
        color="#4C7F9F",
        edgecolor="#25465A",
        linewidth=1.0,
    )

    for bar, value in zip(bars, summary["failure_frequency_percent"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
        )

    ax.set_title(args.title)
    ax.set_ylabel("Localisation failure frequency among bruise images (%)")
    ax.set_xlabel("Post-evaluation analysis cluster")
    ax.set_ylim(0, max(105, summary["failure_frequency_percent"].max() + 12))
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    fig.text(
        0.5,
        0.02,
        f"Source: author's own figure generated from verified RT-DETR-Large fold-level image analysis outputs in {args.reports_root}. "
        "Only images with annotated bruises are included; failure means no matched prediction at the fixed IoU criterion.",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(args.out_png, bbox_inches="tight")

    print(f"Saved {args.out_png}")
    print(f"Saved {args.out_csv}")


if __name__ == "__main__":
    main()
