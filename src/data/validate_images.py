from pathlib import Path
import csv
import hashlib
import sys

from PIL import Image, UnidentifiedImageError


DATA_ROOT = Path("data/raw/source_A/archive (2)")
REPORT_PATH = Path("reports/source_A_quality_report.csv")

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}


def calculate_hash(file_path):
    """Calculate the SHA-256 hash of a file."""
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def validate_image(file_path):
    """Check whether an image can be opened and decoded."""
    try:
        with Image.open(file_path) as image:
            image.verify()

        with Image.open(file_path) as image:
            width, height = image.size
            image_format = image.format

        return {
            "status": "OK",
            "format": image_format,
            "width": width,
            "height": height,
            "error": ""
        }

    except (UnidentifiedImageError, OSError, ValueError) as error:
        return {
            "status": "CORRUPTED",
            "format": "",
            "width": "",
            "height": "",
            "error": str(error)
        }


def main():
    if not DATA_ROOT.exists():
        print(f"Dataset folder not found: {DATA_ROOT}")
        sys.exit(1)

    class_folders = sorted(
        folder for folder in DATA_ROOT.iterdir()
        if folder.is_dir()
    )

    if not class_folders:
        print("No class folders found.")
        sys.exit(1)

    rows = []
    seen_hashes = {}
    summary = {}

    for class_folder in class_folders:
        class_name = class_folder.name

        image_files = sorted(
            file for file in class_folder.rglob("*")
            if file.is_file()
            and file.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        valid_count = 0
        corrupted_count = 0
        duplicate_count = 0

        for image_path in image_files:
            file_hash = calculate_hash(image_path)
            result = validate_image(image_path)

            duplicate_of = ""

            if result["status"] == "OK":
                valid_count += 1

                if file_hash in seen_hashes:
                    duplicate_of = seen_hashes[file_hash]
                    duplicate_count += 1
                else:
                    seen_hashes[file_hash] = str(image_path)

            else:
                corrupted_count += 1

            rows.append({
                "file_path": str(image_path),
                "class": class_name,
                "status": result["status"],
                "format": result["format"],
                "width": result["width"],
                "height": result["height"],
                "sha256": file_hash,
                "duplicate_of": duplicate_of,
                "error": result["error"]
            })

        summary[class_name] = {
            "total": len(image_files),
            "valid": valid_count,
            "corrupted": corrupted_count,
            "duplicates": duplicate_count
        }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "file_path", "class", "status", "format",
        "width", "height", "sha256", "duplicate_of", "error"
    ]

    with REPORT_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== OLIVE-CAD | SOURCE A QUALITY REPORT =====")

    total_images = 0
    total_valid = 0
    total_corrupted = 0
    total_duplicates = 0

    for class_name, values in summary.items():
        print(f"\nClass: {class_name}")
        print(f"  Total images : {values['total']}")
        print(f"  Valid images : {values['valid']}")
        print(f"  Corrupted    : {values['corrupted']}")
        print(f"  Duplicates   : {values['duplicates']}")

        total_images += values["total"]
        total_valid += values["valid"]
        total_corrupted += values["corrupted"]
        total_duplicates += values["duplicates"]

    print("\n===== GLOBAL SUMMARY =====")
    print(f"Total images      : {total_images}")
    print(f"Valid images      : {total_valid}")
    print(f"Corrupted images  : {total_corrupted}")
    print(f"Exact duplicates  : {total_duplicates}")
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
