from __future__ import annotations

from pathlib import Path
import shutil
import re

import pandas as pd

from .config import HyperoptSubsetConfig, PatchConfig, PathsConfig
from .data_prep import build_folds, build_hyperopt_balanced_subset
from .training import (
    build_hyperopt_report_tables,
    evaluate_detection_analysis,
    get_best_metrics,
    load_best_config,
    optimize_hyperparameters,
    resolve_run_dir,
    save_cross_validation_summary_graphs,
    save_final_result_curves,
    save_fold_validation_graphs,
    train_rtdetr,
)


def run_pipeline(
    dataset_root: Path,
    output_root: Path,
    final_epochs: int = 10,
    n_splits: int = 5,
    optuna_trials: int = 5,
    hyperopt_epochs: int = 5,
    base_model: str = "rtdetr-l.pt",
    augment_train: bool = True,
    skip_optuna: bool = False,
    reset_optuna: bool = False,
    optuna_study_name: str = "bruise_hyperopt",
    use_wandb: bool = False,
    wandb_project: str = "bruise-detection-hyperopt",
    resume_run: bool = False,
    patch_config: PatchConfig | None = None,
    hyperopt_subset_config: HyperoptSubsetConfig | None = None,
) -> None:
    paths = PathsConfig(dataset_root=dataset_root, output_root=output_root)
    patch_config = patch_config or PatchConfig()
    hyperopt_subset_config = hyperopt_subset_config or HyperoptSubsetConfig()
    train_imgsz = patch_config.train_imgsz if patch_config.enabled else 640
    if optuna_study_name == "bruise_hyperopt":
        model_tag = re.sub(r"[^A-Za-z0-9]+", "_", Path(base_model).stem).strip("_")
        if patch_config.enabled:
            optuna_study_name = f"bruise_hyperopt_{model_tag}_patch{patch_config.size}_subset{hyperopt_subset_config.per_class}"
        else:
            optuna_study_name = f"bruise_hyperopt_{model_tag}_fullimage_subset{hyperopt_subset_config.per_class}"
    if not resume_run:
        for p in [paths.folds_dir, paths.hyperopt_subset_dir, paths.runs_dir / "cross_validation", paths.final_results_dir]:
            if p.exists():
                shutil.rmtree(p)
        if reset_optuna:
            hyperopt_dir = paths.runs_dir / "hyperparameter_optimization"
            if hyperopt_dir.exists():
                shutil.rmtree(hyperopt_dir)
        if paths.reports_dir.exists():
            for pattern in [
                "fold_*_analysis",
                "fold_*_patches.csv",
                "fold_*_train_augmentation.csv",
                "fold_info.csv",
                "cross_validation_results.csv",
                "cross_validation_summary.csv",
                "cross_validation_metric_summary.png",
                "cross_validation_best_epoch.png",
            ]:
                for p in paths.reports_dir.glob(pattern):
                    if p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink()
            if reset_optuna:
                for filename in [
                    "optuna_study.db",
                    "best_hyperparameters.json",
                    "hyperparameter_optimization_results.csv",
                    "best_parameters_table.csv",
                    "hyperopt_trial_ranking_table.csv",
                    "hyperopt_param_importance_table.csv",
                ]:
                    p = paths.reports_dir / filename
                    if p.exists():
                        p.unlink()

    for p in [paths.folds_dir, paths.runs_dir, paths.reports_dir, paths.final_results_dir]:
        p.mkdir(parents=True, exist_ok=True)

    print("Using dataset sources:")
    for source in paths.dataset_sources:
        print(f"  {source.name}: images={source.images_dir}, labels={source.labels_dir}, data={source.data_yaml}")
    if patch_config.enabled:
        print(f"Using {patch_config.size}x{patch_config.size} cutouts with stride {patch_config.stride}.")
        print(f"Training RT-DETR with imgsz={train_imgsz}.")
    else:
        print("Using full images without cutout generation.")
        print(f"Training RT-DETR with imgsz={train_imgsz}.")

    if hyperopt_subset_config.enabled and not skip_optuna:
        hyperopt_yaml, hyperopt_subset_summary = build_hyperopt_balanced_subset(
            paths,
            subset_config=hyperopt_subset_config,
            patch_config=patch_config,
            augment_train=augment_train,
        )
        print("Created balanced Optuna subset:")
        print(hyperopt_subset_summary)
    else:
        hyperopt_yaml = None

    fold_yaml_paths, fold_info_df, _ = build_folds(
        paths,
        n_splits=n_splits,
        augment_train=augment_train,
        patch_config=patch_config,
    )
    print(f"Created {len(fold_yaml_paths)} folds.")
    print(fold_info_df)

    fold1_yaml = fold_yaml_paths[0]
    optuna_data_yaml = hyperopt_yaml or fold1_yaml
    hyperopt_dir = paths.runs_dir / "hyperparameter_optimization"
    best_config_path = paths.reports_dir / "best_hyperparameters.json"
    if skip_optuna:
        print(f"Skipping Optuna. Loading saved best hyperparameters from: {best_config_path}")
        best_cfg = load_best_config(best_config_path, final_epochs=hyperopt_epochs)
        hyperopt_df = pd.read_csv(paths.reports_dir / "hyperparameter_optimization_results.csv")
    else:
        best_cfg, hyperopt_df, study = optimize_hyperparameters(
            data_yaml=optuna_data_yaml,
            project_dir=hyperopt_dir,
            n_trials=optuna_trials,
            epochs=hyperopt_epochs,
            base_model=base_model,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            storage_path=paths.reports_dir / "optuna_study.db",
            study_name=optuna_study_name,
            reset_study=reset_optuna,
            progress_csv_path=paths.reports_dir / "hyperparameter_optimization_results.csv",
            best_config_path=best_config_path,
            imgsz=train_imgsz,
        )
        hyperopt_df.to_csv(paths.reports_dir / "hyperparameter_optimization_results.csv", index=False)
        report_tables = build_hyperopt_report_tables(hyperopt_df, study=study)
        report_tables["best_params_table"].to_csv(paths.reports_dir / "best_parameters_table.csv", index=False)
        report_tables["trial_ranking_table"].to_csv(paths.reports_dir / "hyperopt_trial_ranking_table.csv", index=False)
        report_tables["param_importance_table"].to_csv(paths.reports_dir / "hyperopt_param_importance_table.csv", index=False)
    best_cfg["epochs"] = final_epochs
    best_cfg["imgsz"] = train_imgsz

    cv_results: list[dict] = []
    cv_dir = paths.runs_dir / "cross_validation"
    for fold_id, fold_yaml in enumerate(fold_yaml_paths, start=1):
        run_name = f"rtdetr_fold_{fold_id}"
        run_dir = resolve_run_dir(cv_dir / run_name)
        best_weight = run_dir / "weights" / "best.pt"
        results_csv = run_dir / "results.csv"
        print(f"Training fold {fold_id}/{len(fold_yaml_paths)}")
        if resume_run and best_weight.exists() and results_csv.exists():
            print(f"Skipping completed fold {fold_id}: {run_dir}")
        else:
            train_rtdetr(data_yaml=fold_yaml, cfg=best_cfg, project_dir=cv_dir, run_name=run_name, resume=resume_run)
            run_dir = resolve_run_dir(cv_dir / run_name)
        metrics = get_best_metrics(run_dir)
        best_weight = run_dir / "weights" / "best.pt"
        analysis_dir = paths.reports_dir / f"fold_{fold_id}_analysis"
        analysis = evaluate_detection_analysis(
            model_path=best_weight,
            fold_yaml=fold_yaml,
            output_dir=analysis_dir,
            iou_threshold=0.5,
            conf_threshold=0.25,
            n_clusters=3,
            imgsz=best_cfg["imgsz"],
        )
        graph_paths = save_fold_validation_graphs(run_dir=run_dir, analysis_dir=analysis_dir, fold_id=fold_id)
        print(f"Saved fold {fold_id} graphs:")
        for graph_path in graph_paths:
            print(f"  {graph_path}")
        cv_results.append({"fold": fold_id, "run_name": run_name, "run_dir": str(run_dir), **metrics})
        cv_results[-1].update(analysis)
        cv_results[-1]["validation_graphs"] = ";".join(str(p) for p in graph_paths)

    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(paths.reports_dir / "cross_validation_results.csv", index=False)
    final_curve_paths = save_final_result_curves(cv_df, paths.final_results_dir)
    summary_graph_paths = save_cross_validation_summary_graphs(cv_df, paths.reports_dir)

    summary = {
        "mean_mAP50": cv_df["best_map50"].mean(),
        "std_mAP50": cv_df["best_map50"].std(),
        "mean_mAP50_95": cv_df["best_map50_95"].mean(),
        "std_mAP50_95": cv_df["best_map50_95"].std(),
        "mean_best_epoch": cv_df["best_epoch"].mean(),
        "std_best_epoch": cv_df["best_epoch"].std(),
    }
    summary["summary_graphs"] = ";".join(str(p) for p in summary_graph_paths)
    summary["final_result_curves"] = ";".join(str(p) for p in final_curve_paths)
    print("Saved final result curves:")
    for curve_path in final_curve_paths:
        print(f"  {curve_path}")
    print("Saved cross-validation summary graphs:")
    for graph_path in summary_graph_paths:
        print(f"  {graph_path}")
    pd.DataFrame([summary]).to_csv(paths.reports_dir / "cross_validation_summary.csv", index=False)
    print("Pipeline complete.")
    print(cv_df)
    print(summary)
