from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURES = [
    "gt_present",
    "pred_present",
    "matched_iou",
    "best_iou",
    "max_conf",
    "gt_count",
    "pred_count",
]


def read_fold_files(reports_root: Path) -> pd.DataFrame:
    rows = []
    for fold_dir in sorted(reports_root.glob("fold_*_analysis")):
        if not fold_dir.is_dir():
            continue
        csv_path = fold_dir / "image_level_analysis.csv"
        if not csv_path.exists():
            continue
        try:
            fold = int(fold_dir.name.split("_")[1])
        except (IndexError, ValueError):
            fold = np.nan
        df = pd.read_csv(csv_path)
        df.insert(0, "fold", fold)
        rows.append(df)

    if not rows:
        raise FileNotFoundError(
            f"No fold image-level files found under {reports_root}. "
            "Expected fold_*_analysis/image_level_analysis.csv"
        )
    return pd.concat(rows, ignore_index=True)


def make_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["fold", "posteval_cluster"], dropna=False)
        .agg(
            samples=("image", "count"),
            match_rate=("matched_iou", "mean"),
            pred_present_rate=("pred_present", "mean"),
            gt_present_rate=("gt_present", "mean"),
            avg_iou=("best_iou", "mean"),
            avg_conf=("max_conf", "mean"),
            avg_gt_count=("gt_count", "mean"),
            avg_pred_count=("pred_count", "mean"),
        )
        .reset_index()
    )


def run(reports_root: Path, output_dir: Path, n_clusters: int, features: list[str]) -> None:
    df = read_fold_files(reports_root)
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    feature_df = df[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    scaled = StandardScaler().fit_transform(feature_df.values)
    df["posteval_cluster"] = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    ).fit_predict(scaled)

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "image_level_analysis_with_posteval_clusters.csv", index=False)
    make_cluster_summary(df).to_csv(output_dir / "posteval_cluster_performance_summary.csv", index=False)

    feature_report = pd.DataFrame({"feature": features})
    feature_report.to_csv(output_dir / "posteval_cluster_features.csv", index=False)

    print(f"Saved: {output_dir / 'image_level_analysis_with_posteval_clusters.csv'}")
    print(f"Saved: {output_dir / 'posteval_cluster_performance_summary.csv'}")
    print(f"Saved: {output_dir / 'posteval_cluster_features.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RF-DETR post-evaluation KMeans clustering.")
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        reports_root=args.reports_root,
        output_dir=args.output_dir,
        n_clusters=args.n_clusters,
        features=args.features,
    )
