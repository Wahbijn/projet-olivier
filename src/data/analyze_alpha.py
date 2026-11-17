from pathlib import Path
from PIL import Image
import csv

DATA_ROOT = Path("data/raw/source_A/archive (2)")
REPORT_PATH = Path("reports/source_A_alpha_report.csv")

extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

rows = []

for path in sorted(DATA_ROOT.rglob("*")):

    if not path.is_file() or path.suffix.lower() not in extensions:
        continue

    with Image.open(path) as image:
        rgba = image.convert("RGBA")

        width, height = rgba.size
        alpha = rgba.getchannel("A")

        alpha_min, alpha_max = alpha.getextrema()
        histogram = alpha.histogram()

        total_pixels = width * height

        transparent_pixels = sum(histogram[:255])
        transparent_percent = (
            transparent_pixels / total_pixels
        ) * 100

        bbox = alpha.point(
            lambda value: 255 if value > 10 else 0
        ).getbbox()

        if bbox:
            x1, y1, x2, y2 = bbox
            bbox_width = x2 - x1
            bbox_height = y2 - y1
        else:
            bbox_width = 0
            bbox_height = 0

    rows.append({
        "file": str(path),
        "class": path.parent.name,
        "width": width,
        "height": height,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "transparent_percent": round(transparent_percent, 2),
        "leaf_bbox_width": bbox_width,
        "leaf_bbox_height": bbox_height
    })

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

with REPORT_PATH.open(
    "w", newline="", encoding="utf-8-sig"
) as file:

    writer = csv.DictWriter(
        file, fieldnames=rows[0].keys()
    )

    writer.writeheader()
    writer.writerows(rows)

print("\n===== ALPHA CHANNEL ANALYSIS =====")
print(f"Images analyzed: {len(rows)}")

for class_name in sorted(set(row["class"] for row in rows)):

    class_rows = [
        row for row in rows
        if row["class"] == class_name
    ]

    average = sum(
        row["transparent_percent"] for row in class_rows
    ) / len(class_rows)

    print(f"\nClass: {class_name}")
    print(f"Images: {len(class_rows)}")
    print(f"Average transparency: {average:.2f}%")

print(f"\nReport saved: {REPORT_PATH}")