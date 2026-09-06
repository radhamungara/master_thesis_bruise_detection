# Mungara Radha MT Bruise Hematoma Detection

RT-DETR based bruise detection pipeline for training, cross-validation, evaluation, and image-level inference. The project prepares bruise and non-bruise YOLO datasets, runs Optuna hyperparameter search, trains RT-DETR across stratified group k-fold splits, generates validation reports, and provides a Streamlit app for testing trained models on new images.

## Features

- YOLO-format dataset loading for bruise and non-bruise image sets.
- Label validation for bounding-box and polygon YOLO annotations.
- Stratified group k-fold cross-validation to reduce leakage from related/augmented images.
- Optional balanced Optuna subset for faster hyperparameter search.
- Optional train-only augmentation and optional cutout/patch generation.
- RT-DETR training through Ultralytics.
- Fold-level metrics, ROC data, confusion matrix, confidence histograms, and summary plots.
- Streamlit web app for uploading an image and visualizing predicted bruise boxes.

## Project Structure

```text
.
|-- run_pipeline.py                         # Main training and evaluation entry point
|-- web_app.py                              # Streamlit inference app
|-- requirements.txt                        # Python dependencies
|-- src/
|   `-- bruise_detection/
|       |-- config.py                       # Dataset paths and pipeline configuration
|       |-- data_prep.py                    # Dataset validation, folds, augmentation, patches
|       |-- pipeline.py                     # End-to-end orchestration
|       `-- training.py                     # RT-DETR training, Optuna, metrics, plots
`-- scripts/                                # Utility scripts for audits, analysis, and plots
```

Generated folders such as `artifacts/`, `runs/`, downloaded result folders, model weights, datasets, and debug images are ignored by Git.

## Requirements

- Python 3.10 or newer recommended
- CUDA-capable GPU recommended for RT-DETR training
- YOLO-format bruise and non-bruise datasets
- Base model weights, for example `rtdetr-l.pt`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Layout

By default, the pipeline expects a dataset root with bruise and non-bruise sources:

```text
Dataset4/
|-- Bruises.yolov8/
|   `-- train/
|       |-- images/
|       `-- labels/
`-- Non-Bruises.yolov8/
    `-- train/
        |-- images/
        `-- labels/
```

The code also accepts these fallback folder names:

- Bruise data: `Bruises.yolov8`, `Bruises`, or `bruise`
- Non-bruise data: `Non-Bruises.yolov8`, `Non-Bruises`, or `non_bruise`

Clustered dataset folders are also supported when images and labels are stored under per-cluster folders such as:

```text
train/cluster_0/images/
train/cluster_0/labels/
```

## Important Files to Commit

Commit source code and documentation:

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

## Notes

- The default dataset path in `run_pipeline.py` is `R:\Dataset4`; change it with `--dataset-root` when running on another machine.
- The default base model is `rtdetr-l.pt`; download or place the model file locally before training.
- Large datasets, trained weights, and generated reports should be stored outside Git or handled with a large-file storage strategy if required.
