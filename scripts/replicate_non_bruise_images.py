from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one same-size augmented copy for each non-bruise image."
    )
    parser.add_argument(
        "--source-images",
        type=Path,
        default=Path(r"R:\Dataset2\non-bruises.yolov8\train\images"),
        help="Folder containing original non-bruise images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"R:\Dataset2\non-bruises_replicated_images"),
        help="Folder where replicated image files will be saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible augmentations.",
    )
    return parser.parse_args()


def iter_images(source_images: Path) -> list[Path]:
    return sorted(
        path
        for path in source_images.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def augment_image(image: Image.Image, rng: random.Random) -> Image.Image:
    original_size = image.size
    augmented = ImageOps.mirror(image)

    augmented = ImageEnhance.Brightness(augmented).enhance(rng.uniform(0.88, 1.12))
    augmented = ImageEnhance.Contrast(augmented).enhance(rng.uniform(0.90, 1.14))
    augmented = ImageEnhance.Color(augmented).enhance(rng.uniform(0.90, 1.12))
    augmented = ImageEnhance.Sharpness(augmented).enhance(rng.uniform(0.85, 1.20))

    if rng.random() < 0.5:
        augmented = augmented.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.45)))

    if augmented.size != original_size:
        raise RuntimeError(
            f"Augmented image size changed from {original_size} to {augmented.size}"
        )

    return augmented


def output_path_for(image_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{image_path.stem}_replicated{image_path.suffix.lower()}"


def main() -> None:
    args = parse_args()
    source_images = args.source_images
    output_dir = args.output

    if not source_images.exists():
        raise FileNotFoundError(f"Source image folder does not exist: {source_images}")

    images = iter_images(source_images)
    if not images:
        raise RuntimeError(f"No image files found in: {source_images}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    created = 0
    for image_path in images:
        target_path = output_path_for(image_path, output_dir)
        with Image.open(image_path) as img:
            original = ImageOps.exif_transpose(img)
            original_size = original.size
            augmented = augment_image(original, rng)

            if augmented.size != original_size:
                raise RuntimeError(
                    f"{image_path.name}: expected {original_size}, got {augmented.size}"
                )

            save_kwargs = {}
            if target_path.suffix.lower() in {".jpg", ".jpeg"}:
                save_kwargs = {"quality": 95, "subsampling": 0}

            augmented.save(target_path, **save_kwargs)
            created += 1

    print(f"Source images: {len(images)}")
    print(f"Replicated images created: {created}")
    print(f"Output folder: {output_dir}")


if __name__ == "__main__":
    main()
