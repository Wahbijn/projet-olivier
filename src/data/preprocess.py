from pathlib import Path
import csv
import numpy as np
import cv2

from PIL import Image

INPUT_DIR = Path("data/raw/source_A/archive (2)")
OUTPUT_DIR = Path("data/processed/source_A_v1")
REPORT_PATH = Path("reports/source_A_preprocessing_report.csv")

SUPPORTED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".webp"
}

CLASS_MAPPING = {
    "saglam": "healthy",
    "hastalikli": "diseased",
}


def normalize_folder_name(name):
    """Normalize Turkish folder names."""
    return (
        name.lower()
        .replace("ı", "i")
        .replace("ş", "s")
        .replace("ğ", "g")
        .replace("ü", "u")
        .replace("ö", "o")
        .replace("ç", "c")
    )


def preprocess_image(image_path, output_path):
    """Crop the largest visible component and replace transparency."""

    with Image.open(image_path) as image:
        rgba = np.array(image.convert("RGBA"))

    alpha = rgba[:, :, 3]
    mask = (alpha > 10).astype(np.uint8)

    number, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if number <= 1:
        raise ValueError("No visible foreground component found.")

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    x = stats[largest, cv2.CC_STAT_LEFT]
    y = stats[largest, cv2.CC_STAT_TOP]
    width = stats[largest, cv2.CC_STAT_WIDTH]
    height = stats[largest, cv2.CC_STAT_HEIGHT]

    padding = max(8, int(max(width, height) * 0.03))

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(rgba.shape[1], x + width + padding)
    y2 = min(rgba.shape[0], y + height + padding)

    cropped = rgba[y1:y2, x1:x2].copy()

    selected_component = labels[y1:y2, x1:x2] == largest

    cropped[:, :, 3] = np.where(
        selected_component,
        cropped[:, :, 3],
        0
    )

    cropped_image = Image.fromarray(cropped)
    white_background = Image.new("RGBA", cropped_image.size, "white")
    white_background.alpha_composite(cropped_image)

    final_image = white_background.convert("RGB")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_image.save(output_path, format="PNG", optimize=True)

    return {
        "original_width": rgba.shape[1],
        "original_height": rgba.shape[0],
        "crop_width": x2 - x1,
        "crop_height": y2 - y1,
        "status": "OK"
    }


def main():

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report_rows = []
    class_counts = {}
    failed_count = 0

    class_folders = [
        folder for folder in INPUT_DIR.iterdir()
        if folder.is_dir()
    ]

    for class_folder in sorted(class_folders):

        original_class = normalize_folder_name(class_folder.name)

        if original_class not in CLASS_MAPPING:
            print(f"Unknown class folder: {class_folder.name}")
            continue

        class_name = CLASS_MAPPING[original_class]
        class_counts.setdefault(class_name, 0)

        image_files = sorted(
            path for path in class_folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        print(f"\nProcessing class: {class_name}")
        print(f"Images found: {len(image_files)}")

        for image_path in image_files:

            relative_path = image_path.relative_to(class_folder)

            output_path = (
                OUTPUT_DIR / class_name /
                relative_path.with_suffix(".png")
            )

            try:
                result = preprocess_image(image_path, output_path)
                class_counts[class_name] += 1

            except Exception as error:
                failed_count += 1

                result = {
                    "original_width": "",
                    "original_height": "",
                    "crop_width": "",
                    "crop_height": "",
                    "status": f"ERROR: {error}"
                }

                print(f"Error processing {image_path}: {error}")

            report_rows.append({
                "source_file": str(image_path),
                "processed_file": str(output_path),
                "original_class": class_folder.name,
                "class": class_name,
                **result
            })

    fieldnames = [
        "source_file",
        "processed_file",
        "original_class",
        "class",
        "original_width",
        "original_height",
        "crop_width",
        "crop_height",
        "status"
    ]

    with REPORT_PATH.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    print("\n===== OLIVE-CAD PREPROCESSING REPORT =====")

    for class_name, count in sorted(class_counts.items()):
        print(f"{class_name}: {count} images")

    total = sum(class_counts.values())

    print(f"\nTotal processed: {total}")
    print(f"Failed images: {failed_count}")
    print(f"Report: {REPORT_PATH}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
