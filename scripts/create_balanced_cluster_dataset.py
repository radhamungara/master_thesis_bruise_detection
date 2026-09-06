from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a balanced cluster dataset for bruise detection.")
    parser.add_argument("--source-root", type=Path, required=True, help="Source clustered dataset root, e.g. ~/Dataset3")
    parser.add_argument("--target-root", type=Path, required=True, help="Target balanced dataset root")
    parser.add_argument("--bruise-per-cluster", type=int, default=45, help="Images to copy from each bruise cluster")
    parser.add_argument("--non-bruise-per-cluster", type=int, default=90, help="Images to copy from each non-bruise cluster")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    parser.add_argument("--overwrite", action="store_true", help="Delete target folder before creating it")
    return parser.parse_args()


def cluster_dirs(class_root: Path) -> list[Path]:
    train_root = class_root / "train"
    if not train_root.exists():
        raise FileNotFoundError(f"Missing train folder: {train_root}")
    clusters = sorted([p for p in train_root.iterdir() if p.is_dir() and (p / "images").exists()])
    if not clusters:
        raise FileNotFoundError(f"No cluster folders with images found in: {train_root}")
    return clusters


def image_paths(cluster_dir: Path) -> list[Path]:
    return sorted([p for p in (cluster_dir / "images").iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def label_for_image(cluster_dir: Path, image_path: Path) -> Path:
    return cluster_dir / "labels" / f"{image_path.stem}.txt"


def copy_cluster(cluster_dir: Path, target_cluster_dir: Path, count: int, rng: random.Random) -> dict:
    images = image_paths(cluster_dir)
    if len(images) < count:
        raise ValueError(f"{cluster_dir} has only {len(images)} images, but {count} requested.")

    selected = rng.sample(images, count)
    target_images = target_cluster_dir / "images"
    target_labels = target_cluster_dir / "labels"
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing_labels = 0
    for image_path in selected:
        label_path = label_for_image(cluster_dir, image_path)
        shutil.copy2(image_path, target_images / image_path.name)
        if label_path.exists():
            shutil.copy2(label_path, target_labels / label_path.name)
        else:
            (target_labels / f"{image_path.stem}.txt").write_text("", encoding="utf-8")
            missing_labels += 1
        copied += 1

    return {
        "cluster": cluster_dir.name,
        "source_images": len(images),
        "copied_images": copied,
        "missing_labels": missing_labels,
    }


def write_summary(target_root: Path, rows: list[dict]) -> None:
    lines = ["class_name,cluster_id,source_images,copied_images,missing_labels"]
    for row in rows:
        lines.append(
            f"{row['class_name']},{row['cluster']},{row['source_images']},{row['copied_images']},{row['missing_labels']}"
        )
    (target_root / "balanced_cluster_summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    target_root = args.target_root.expanduser().resolve()

    if target_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Target already exists: {target_root}. Use --overwrite to replace it.")
        shutil.rmtree(target_root)

    rng = random.Random(args.seed)
    specs = [
        ("bruise", args.bruise_per_cluster),
        ("non_bruise", args.non_bruise_per_cluster),
    ]

    rows: list[dict] = []
    for class_name, count in specs:
        class_root = source_root / class_name
        if not class_root.exists():
            raise FileNotFoundError(f"Missing class folder: {class_root}")
        for cluster_dir in cluster_dirs(class_root):
            row = copy_cluster(
                cluster_dir=cluster_dir,
                target_cluster_dir=target_root / class_name / "train" / cluster_dir.name,
                count=count,
                rng=rng,
            )
            row["class_name"] = class_name
            rows.append(row)

    for extra_file in ["cluster_summary.csv", "cluster_dataset_manifest.csv"]:
        src = source_root / extra_file
        if src.exists():
            shutil.copy2(src, target_root / extra_file)

    write_summary(target_root, rows)
    print(f"Created balanced dataset: {target_root}")
    for row in rows:
        print(f"{row['class_name']} {row['cluster']}: copied {row['copied_images']} of {row['source_images']}")
    print(f"Summary: {target_root / 'balanced_cluster_summary.csv'}")


if __name__ == "__main__":
    main()
