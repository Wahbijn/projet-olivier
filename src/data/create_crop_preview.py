
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import cv2
import random
import math

DATA_ROOT = Path("data/raw/source_A/archive (2)")
REPORT_DIR = Path("reports")

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

SAMPLE_SIZE = 8
THUMB_WIDTH = 500
THUMB_HEIGHT = 160
CELL_HEIGHT = 195
MARGIN = 15
COLUMNS = 2


def isolate_largest_component(image):
    """Keep the largest connected visible component."""

    rgba = np.array(image.convert("RGBA"))
    alpha = rgba[:, :, 3]

    mask = (alpha > 10).astype(np.uint8)

    number, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if number <= 1:
        return None

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    x = stats[largest, cv2.CC_STAT_LEFT]
    y = stats[largest, cv2.CC_STAT_TOP]
    width = stats[largest, cv2.CC_STAT_WIDTH]
    height = stats[largest, cv2.CC_STAT_HEIGHT]

    rgba[:, :, 3] = np.where(labels == largest, alpha, 0)

    isolated = Image.fromarray(rgba)

    padding = max(8, int(max(width, height) * 0.03))

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(rgba.shape[1], x + width + padding)
    y2 = min(rgba.shape[0], y + height + padding)

    cropped = isolated.crop((x1, y1, x2, y2))

    background = Image.new("RGBA", cropped.size, "white")
    background.alpha_composite(cropped)

    return background.convert("RGB"), width, height


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    for class_folder in sorted(DATA_ROOT.iterdir()):

        if not class_folder.is_dir():
            continue

        files = sorted(
            path for path in class_folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in EXTENSIONS
        )

        if not files:
            continue

        rng = random.Random(42)
        samples = rng.sample(files, min(SAMPLE_SIZE, len(files)))

        rows = math.ceil(len(samples) / COLUMNS)

        canvas = Image.new(
            "RGB",
            (
                COLUMNS * THUMB_WIDTH + (COLUMNS + 1) * MARGIN,
                rows * CELL_HEIGHT + MARGIN
            ),
            "white"
        )

        draw = ImageDraw.Draw(canvas)

        for index, path in enumerate(samples):

            with Image.open(path) as image:
                result = isolate_largest_component(image)

            if result is None:
                print(f"Warning: no visible component in {path}")
                continue

            cropped, width, height = result

            cropped.thumbnail(
                (THUMB_WIDTH, THUMB_HEIGHT),
                Image.Resampling.LANCZOS
            )

            row = index // COLUMNS
            column = index % COLUMNS

            x = MARGIN + column * THUMB_WIDTH
            y = MARGIN + row * CELL_HEIGHT

            draw.text(
                (x, y),
                f"{path.name} | Crop: {width}x{height}",
                fill="black"
            )

            canvas.paste(cropped, (x, y + 25))

            print(f"{path.name}: bbox={width}x{height}")

        output_path = (
            REPORT_DIR / f"crop_preview_clean_{class_folder.name}.jpg"
        )

        canvas.save(output_path, quality=95)
        print(f"Preview saved: {output_path}")


if __name__ == "__main__":
    main()
