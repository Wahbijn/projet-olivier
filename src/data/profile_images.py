from pathlib import Path
from collections import Counter
import csv
from PIL import Image

DATA_ROOT = Path("data/raw/source_A/archive (2)")
REPORT_PATH = Path("reports/source_A_image_profile.csv")

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

rows = []
formats = Counter()
modes = Counter()
dimensions = Counter()

for path in sorted(DATA_ROOT.rglob("*")):
    if not path.is_file() or path.suffix.lower() not in extensions:
        continue

    with Image.open(path) as image:
        width, height = image.size
        image_format = image.format
        mode = image.mode

    formats[image_format] += 1
    modes[mode] += 1
    dimensions[f"{width}x{height}"] += 1

    rows.append({
        "file_path": str(path),
        "class": path.parent.name,
        "width": width,
        "height": height,
        "format": image_format,
        "mode": mode,
        "aspect_ratio": round(width / height, 3)
    })

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

with REPORT_PATH.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\n===== OLIVE-CAD | IMAGE PROFILE =====")
print(f"Total images analyzed: {len(rows)}")

print("\nImage formats:")
for name, count in formats.most_common():
    print(f"  {name}: {count}")

print("\nColor modes:")
for name, count in modes.most_common():
    print(f"  {name}: {count}")

print("\nMost common dimensions:")
for name, count in dimensions.most_common(10):
    print(f"  {name}: {count} images")

print("\n===== DIMENSION RANGES =====")
print(f"Minimum width: {min(r['width'] for r in rows)}")
print(f"Maximum width: {max(r['width'] for r in rows)}")
print(f"Minimum height: {min(r['height'] for r in rows)}")
print(f"Maximum height: {max(r['height'] for r in rows)}")

print(f"\nReport saved: {REPORT_PATH}")
