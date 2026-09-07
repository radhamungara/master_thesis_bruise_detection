# Mungara MT Bruise Hematoma Detection

This repository contains the implementation and experimental pipeline for detecting bruises and hematomas in images using an RT-DETR object detection model. The project includes dataset preparation, label validation, cross-validation training, Optuna-based hyperparameter search, result analysis, and a Streamlit application for image-level inference.

The codebase is designed for YOLO-format bruise and non-bruise datasets and supports both full-image training and optional patch-based training.

---

## 1. Project Overview

Bruise and hematoma detection is treated as an object detection problem. The goal of this project is to train an RT-DETR model that can locate bruise regions in input images and evaluate its performance across multiple validation folds.

The pipeline supports:

- Preparing clean YOLO-format training and validation datasets
- Validating image-label pairs before training
- Handling bruise and non-bruise image sources
- Creating stratified group k-fold splits
- Running Optuna hyperparameter optimization
- Training RT-DETR models with the best hyperparameters
- Evaluating predictions using fold-level and image-level metrics
- Generating plots and CSV reports for thesis or experiment documentation
- Running a Streamlit web app for single-image bruise detection

---

## 2. Repository Structure

```text
Bruise Detection/
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- run_pipeline.py
|-- web_app.py
|-- analyze_failure_causes.py
|-- save_bad_prediction_examples.py
|-- generate_best_fold_metrics_curve.py
|-- generate_metric_summary.py
|-- generate_thesis_curve.py
|-- generate_thesis_training_validation_proof.py
|
|-- src/
|   |-- __init__.py
|   `-- bruise_detection/
|       |-- __init__.py
|       |-- config.py
|       |-- data_prep.py
|       |-- pipeline.py
|       `-- training.py
|
`-- scripts/
    |-- audit_yolo_labels.py
    |-- create_balanced_cluster_dataset.py
    |-- generate_rtdetr_balanced_localisation_failure_examples.py
    |-- generate_rtdetr_bruise_posteval_cluster_failure_plot.py
    |-- generate_rtdetr_localisation_failure_examples.py
    |-- replicate_non_bruise_images.py
    |-- replicate_non_bruises.py
    `-- rfdetr_posteval_kmeans.py
```

### Folder and File Responsibilities

| Folder / File | Purpose |
|---------------|---------|
| `run_pipeline.py` | Main command-line entry point for training and evaluation |
| `web_app.py` | Streamlit app for testing a trained model on uploaded images |
| `src/bruise_detection/config.py` | Dataset paths, patch settings, and hyperparameter configuration |
| `src/bruise_detection/data_prep.py` | Dataset loading, validation, fold generation, augmentation, and patch creation |
| `src/bruise_detection/pipeline.py` | End-to-end orchestration of data preparation, Optuna, training, and reporting |
| `src/bruise_detection/training.py` | RT-DETR training, metric extraction, evaluation, plotting, and Optuna logic |
| `scripts/` | Utility scripts for label audits, dataset balancing, clustering, and post-evaluation analysis |

---

## 3. Environment Setup

### 3.1 Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

Or using Conda:

```bash
conda create -n bruise-detection python=3.10
conda activate bruise-detection
```

### 3.2 Install Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies:

- `ultralytics`
- `opencv-python`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `pyyaml`
- `optuna`
- `wandb`
- `streamlit`

Python 3.10 or newer is recommended. A CUDA-capable GPU is recommended for model training.

---

## 4. Data Setup

The training pipeline expects a YOLO-style dataset root containing bruise and non-bruise datasets.

Default expected layout:

```text
Dataset4/
|-- Bruises.yolov8/
|   `-- train/
|       |-- images/
|       `-- labels/
|
`-- Non-Bruises.yolov8/
    `-- train/
        |-- images/
        `-- labels/
```

Supported source folder names:

| Dataset Type | Supported Folder Names |
|-------------|-------------------------|
| Bruise images | `Bruises.yolov8`, `Bruises`, `bruise` |
| Non-bruise images | `Non-Bruises.yolov8`, `Non-Bruises`, `non_bruise` |

Clustered dataset folders are also supported:

```text
train/cluster_0/images/
train/cluster_0/labels/
train/cluster_1/images/
train/cluster_1/labels/
```

Labels must be in YOLO format. The pipeline can process standard bounding-box labels and polygon labels, converting them into detection boxes where required.

---

## 5. Training Pipeline

The main training pipeline is executed through:

```bash
python run_pipeline.py
```

Default settings:

```text
dataset root: R:\Dataset4
output root: artifacts
base model: rtdetr-l.pt
cross-validation folds: 5
Optuna trials: 5
hyperparameter epochs: 5
final training epochs: 10
```

Run full training:

```bash
python run_pipeline.py --dataset-root "R:\Dataset4" --output-root artifacts
```

Run with custom settings:

```bash
python run_pipeline.py ^
  --dataset-root "R:\Dataset4" ^
  --output-root artifacts ^
  --base-model rtdetr-l.pt ^
  --n-splits 5 ^
  --optuna-trials 5 ^
  --hyperopt-epochs 5 ^
  --final-epochs 10
```

Skip Optuna when saved best hyperparameters already exist:

```bash
python run_pipeline.py --dataset-root "R:\Dataset4" --skip-optuna
```

Resume interrupted training:

```bash
python run_pipeline.py --dataset-root "R:\Dataset4" --resume-run
```

Reset Optuna study:

```bash
python run_pipeline.py --dataset-root "R:\Dataset4" --reset-optuna
```

---

## 6. Patch and Augmentation Options

By default, the pipeline trains on full images without patch generation.

Enable patch-based training:

```bash
python run_pipeline.py --dataset-root "R:\Dataset4" --use-patches --patch-size 32 --patch-stride 32
```

Patch options:

| Argument | Description |
|----------|-------------|
| `--use-patches` | Enables cutout generation |
| `--patch-size` | Patch size in pixels |
| `--patch-stride` | Patch stride in pixels |
| `--max-positive-patches-per-image` | Maximum bruise-containing patches per positive image |
| `--max-empty-negative-patches-per-image` | Maximum empty patches per non-bruise image |

Enable physical augmentation:

```bash
python run_pipeline.py --dataset-root "R:\Dataset4" --use-physical-augmentation
```

Train-only augmentation can create brightness, contrast, and hue/saturation variants. Validation data is not augmented.

---

## 7. Hyperparameter Optimization

Optuna is used to search RT-DETR training parameters. The objective is to maximize validation `mAP50-95`.

The search includes:

- Batch size
- Learning rate
- Weight decay
- Patience
- HSV augmentation values
- Translation
- Scale
- Horizontal flip probability

Generated Optuna outputs:

```text
artifacts/reports/optuna_study.db
artifacts/reports/best_hyperparameters.json
artifacts/reports/hyperparameter_optimization_results.csv
artifacts/reports/best_parameters_table.csv
artifacts/reports/hyperopt_trial_ranking_table.csv
artifacts/reports/hyperopt_param_importance_table.csv
```

---

## 8. Evaluation and Results

After training, the pipeline evaluates each fold and saves metrics, plots, and image-level analysis files.

Main result files:

```text
artifacts/reports/cross_validation_results.csv
artifacts/reports/cross_validation_summary.csv
artifacts/reports/cross_validation_metric_summary.png
artifacts/reports/cross_validation_best_epoch.png
```

Fold-level analysis is saved under:

```text
artifacts/reports/fold_1_analysis/
artifacts/reports/fold_2_analysis/
artifacts/reports/fold_3_analysis/
artifacts/reports/fold_4_analysis/
artifacts/reports/fold_5_analysis/
```

Each fold analysis may include:

- Detection summary JSON
- Image-level prediction analysis
- ROC curve data
- Cluster performance summary
- Validation metric plots
- Loss curves
- Confusion matrix
- Confidence histogram

Final reporting curves are saved under:

```text
artifacts/Final Result/
```

---

## 10. Utility Scripts

| Script | Purpose |
|--------|---------|
| `scripts/audit_yolo_labels.py` | Checks YOLO label files for formatting or annotation issues |
| `scripts/create_balanced_cluster_dataset.py` | Creates a balanced dataset using cluster information |
| `scripts/generate_rtdetr_localisation_failure_examples.py` | Generates localization failure examples |
| `scripts/generate_rtdetr_balanced_localisation_failure_examples.py` | Generates balanced localization failure examples |
| `scripts/generate_rtdetr_bruise_posteval_cluster_failure_plot.py` | Creates post-evaluation cluster failure plots |
| `scripts/replicate_non_bruise_images.py` | Replicates non-bruise images for dataset balancing |
| `scripts/replicate_non_bruises.py` | Utility for non-bruise replication workflows |
| `scripts/rfdetr_posteval_kmeans.py` | Runs KMeans-based post-evaluation analysis |

---

## 11. Files to Commit to GitLab

Commit:

```text
README.md
requirements.txt
.gitignore
run_pipeline.py
web_app.py
analyze_failure_causes.py
save_bad_prediction_examples.py
generate_best_fold_metrics_curve.py
generate_metric_summary.py
generate_thesis_curve.py
generate_thesis_training_validation_proof.py
src/
scripts/
```





