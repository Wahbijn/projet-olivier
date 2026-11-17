from pathlib import Path
from PIL import Image, ImageDraw
import random
import math

DATA_ROOT = Path("data/raw/source_A/archive (2)")
OUTPUT_DIR = Path("reports")

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

THUMB_WIDTH = 700
THUMB_HEIGHT = 197
CELL_HEIGHT = 235
MARGIN = 20
COLUMNS = 2
SAMPLE_SIZE = 8

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for class_folder in sorted(DATA_ROOT.iterdir()):
    if not class_folder.is_dir():
        continue

    files = sorted(
        p for p in class_folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

    if not files:
        continue

    rng = random.Random(42)
    samples = rng.sample(files, min(SAMPLE_SIZE, len(files)))

    rows = math.ceil(len(samples) / COLUMNS)

    canvas_width = COLUMNS * THUMB_WIDTH + (COLUMNS + 1) * MARGIN
    canvas_height = rows * CELL_HEIGHT + MARGIN

    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    print(f"\nClass: {class_folder.name}")

    for index, path in enumerate(samples):
        with Image.open(path) as image:
            original_size = image.size
            rgba = image.convert("RGBA")

            alpha = rgba.getchannel("A")
            has_transparency = alpha.getextrema()[0] < 255

            background = Image.new("RGBA", rgba.size, "white")
            background.alpha_composite(rgba)

            preview = background.convert("RGB")
            preview.thumbnail(
                (THUMB_WIDTH, THUMB_HEIGHT),
                Image.Resampling.LANCZOS
            )

        row = index // COLUMNS
        column = index % COLUMNS

        x = MARGIN + column * THUMB_WIDTH
        y = MARGIN + row * CELL_HEIGHT

        draw.text(
            (x, y),
            f"{path.name} | {original_size} | Transparency: {has_transparency}",
            fill="black"
        )

        canvas.paste(preview, (x, y + 25))

    output_path = OUTPUT_DIR / f"preview_{class_folder.name}.jpg"
    canvas.save(output_path, quality=95)

    print(f"Preview saved: {output_path}")

print("\nVisual inspection files created successfully.")
