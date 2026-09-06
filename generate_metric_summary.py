from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_CV_RESULTS = Path("artifacts/reports/cross_validation_results.csv")
DEFAULT_RESULTS_ROOT = Path("fold_training_backup/runs/detect/artifacts/runs/cross_validation")
DEFAULT_OUTPUT = Path("final_result_download/object_detection_metric_summary.csv")
DEFAULT_MARKDOWN_OUTPUT = Path("final_result_download/object_detection_metric_summary.md")


METRIC_COLUMNS = {
    "Precision": {
        "cv_column": "precision",
        "result_candidates": ["metrics/precision(B)", "metrics/precision"],
        "meaning": "How many predicted bruises were actually correct",
    },
    "Recall": {
        "cv_column": "recall",
        "result_candidates": ["metrics/recall(B)", "metrics/recall"],
        "meaning": "How many real bruises were detected",
    },
    "mAP@50": {
        "cv_column": "best_map50",
        "result_candidates": ["metrics/mAP50(B)", "metrics/mAP50", "metrics/mAP50_box"],
        "meaning": "Detection accuracy at IoU 0.50",
    },
    "mAP@50-95": {
        "cv_column": "best_map50_95",
        "result_candidates": ["metrics/mAP50-95(B)", "metrics/mAP50-95", "metrics/mAP50-95_box"],
        "meaning": "Overall detection quality across IoU thresholds",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate object detection metric summary statistics.")
    parser.add_argument("--input", type=Path, default=None, help="Optional path to cross_validation_results.csv")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Folder containing rtdetr_fold_*/results.csv files",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT, help="Output Markdown path")
    return parser.parse_args()


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((col for col in candidates if col in df.columns), None)


def best_row_metrics(results_csv: Path) -> dict:
    fold_id = int(results_csv.parent.name.rsplit("_", 1)[-1])
    df = pd.read_csv(results_csv)
    df.columns = [col.strip() for col in df.columns]

    map95_col = first_existing_column(df, METRIC_COLUMNS["mAP@50-95"]["result_candidates"])
    if map95_col is None:
        raise ValueError(f"Could not find mAP@50-95 column in {results_csv}")

    map95_values = pd.to_numeric(df[map95_col], errors="coerce")
    best_idx = map95_values.idxmax()
    row = {"fold": fold_id}

    for display_name, info in METRIC_COLUMNS.items():
        metric_col = first_existing_column(df, info["result_candidates"])
        row[info["cv_column"]] = float(pd.to_numeric(df.loc[[best_idx], metric_col], errors="coerce").iloc[0]) if metric_col else pd.NA

    row["best_epoch"] = int(df.loc[best_idx, "epoch"]) if "epoch" in df.columns else int(best_idx + 1)
    row["results_csv"] = str(results_csv)
    return row


def load_fold_results(results_root: Path) -> pd.DataFrame:
    result_files = sorted(results_root.glob("rtdetr_fold_*/results.csv"))
    if not result_files:
        raise FileNotFoundError(f"No fold results.csv files found under {results_root}")
    return pd.DataFrame([best_row_metrics(path) for path in result_files])


def resolve_metric_dataframe(input_path: Path | None, results_root: Path) -> pd.DataFrame:
    if input_path is not None:
        return pd.read_csv(input_path)
    if results_root.exists():
        return load_fold_results(results_root)
    if DEFAULT_CV_RESULTS.exists():
        return pd.read_csv(DEFAULT_CV_RESULTS)
    raise FileNotFoundError(f"Could not find fold results under {results_root} or {DEFAULT_CV_RESULTS}")


def build_summary(metric_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for display_name, info in METRIC_COLUMNS.items():
        column = info["cv_column"]
        if column not in metric_df.columns:
            raise ValueError(f"Missing required metric column: {column}")

        values = pd.to_numeric(metric_df[column], errors="coerce").dropna()
        mean = float(values.mean())
        std = float(values.std()) if len(values) > 1 else 0.0
        rows.append(
            {
                "Metric": display_name,
                "Value": f"{mean:.3f} +/- {std:.3f}",
                "Meaning": info["meaning"],
                "Mean": mean,
                "Std": std,
                "Min": float(values.min()),
                "Max": float(values.max()),
                "Folds used": int(values.count()),
            }
        )
    return pd.DataFrame(rows)


def save_markdown(summary_df: pd.DataFrame, output_path: Path) -> None:
    table_df = summary_df[["Metric", "Value", "Meaning"]]
    table_lines = [
        "| Metric | Value | Meaning |",
        "|---|---:|---|",
    ]
    for _, row in table_df.iterrows():
        table_lines.append(f"| {row['Metric']} | {row['Value']} | {row['Meaning']} |")
    lines = [
        "# Object Detection Metric Summary",
        "",
        *table_lines,
        "",
        "Values are reported as mean +/- standard deviation across the five cross-validation folds.",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    metric_df = resolve_metric_dataframe(args.input, args.results_root)
    summary_df = build_summary(metric_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output, index=False)
    save_markdown(summary_df, args.markdown_output)

    print(f"Saved metric summary CSV: {args.output}")
    print(f"Saved metric summary Markdown: {args.markdown_output}")
    print(summary_df[["Metric", "Value", "Meaning"]].to_string(index=False))


if __name__ == "__main__":
    main()
