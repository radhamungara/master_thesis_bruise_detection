from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replicate non-bruise YOLO image/label pairs without resizing or re-encoding images."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"R:\Dataset2"),
        help="Dataset root that contains Non-Bruises.yolov8 and Bruises.yolov8.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="Target number of non-bruise images. Defaults to the bruise image count.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually create replicated files. Without this flag, only prints what would happen.",
    )
    return parser.parse_args()


def image_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def next_copy_stem(source_stem: str, copy_index: int) -> str:
    return f"{source_stem}__rep_{copy_index:03d}"


def main() -> None:
    args = parse_args()
    non_root = args.dataset_root / "Non-Bruises.yolov8" / "train"
    bruise_root = args.dataset_root / "Bruises.yolov8" / "train"
    non_images_dir = non_root / "images"
    non_labels_dir = non_root / "labels"
    bruise_images_dir = bruise_root / "images"

    for required_dir in [non_images_dir, non_labels_dir, bruise_images_dir]:
        if not required_dir.exists():
            raise FileNotFoundError(f"Missing required folder: {required_dir}")

    non_images = image_files(non_images_dir)
    bruise_images = image_files(bruise_images_dir)
    if not non_images:
        raise ValueError(f"No non-bruise images found in {non_images_dir}")

    target_count = args.target_count if args.target_count is not None else len(bruise_images)
    extra_needed = max(0, target_count - len(non_images))

    print(f"Non-bruise images: {len(non_images)}")
    print(f"Bruise images: {len(bruise_images)}")
    print(f"Target non-bruise images: {target_count}")
    print(f"Extra non-bruise copies needed: {extra_needed}")

    if extra_needed == 0:
        print("Nothing to copy.")
        return

    full_rounds = math.ceil(extra_needed / len(non_images))
    planned_pairs: list[tuple[Path, Path, Path, Path]] = []
    for round_id in range(1, full_rounds + 1):
        for image_path in non_images:
            if len(planned_pairs) >= extra_needed:
                break
            label_path = non_labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for {image_path.name}: {label_path}")

            copy_stem = next_copy_stem(image_path.stem, round_id)
            copy_image = non_images_dir / f"{copy_stem}{image_path.suffix}"
            copy_label = non_labels_dir / f"{copy_stem}.txt"
            planned_pairs.append((image_path, label_path, copy_image, copy_label))

    if not args.execute:
        print("Dry run only. Add --execute to create these replicated files.")
        print(f"First planned copy: {planned_pairs[0][0].name} -> {planned_pairs[0][2].name}")
        print(f"Last planned copy: {planned_pairs[-1][0].name} -> {planned_pairs[-1][2].name}")
        return

    created = 0
    for source_image, source_label, copy_image, copy_label in planned_pairs:
        if copy_image.exists() or copy_label.exists():
            raise FileExistsError(f"Refusing to overwrite existing replicated file: {copy_image} or {copy_label}")
        shutil.copy2(source_image, copy_image)
        shutil.copy2(source_label, copy_label)
        created += 1

    print(f"Created {created} replicated image/label pairs.")
    print(f"New non-bruise image count: {len(image_files(non_images_dir))}")


if __name__ == "__main__":
    main()
