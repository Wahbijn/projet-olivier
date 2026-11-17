from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image
import hashlib
import random
import shutil
import csv
import unicodedata

SOURCE_A = Path("data/processed/source_A_v1")
SOURCE_C = Path("data/raw/source_C/roboflow_dataset")
OUTPUT = Path("data/processed/olive_cad_v1")
REPORT = Path("reports/olive_cad_v1_manifest.csv")

SEED = 42
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

MAP_A = {"healthy": "healthy", "diseased": "peacock_spot"}
MAP_C = {
    "healthy": "healthy",
    "olive_peacock_spot": "peacock_spot",
    "aculus_olearius": "aculus_olearius",
    "knot_disease": None,
}


def normalize(name):
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    return name.lower().replace(" ", "_").replace("-", "_")


def collect_images():
    records = []
    excluded = 0

    # Source A: already preprocessed
    for folder in SOURCE_A.iterdir():
        if not folder.is_dir():
            continue

        label = MAP_A.get(folder.name)
        if label is None:
            raise ValueError(f"Unknown Source A class: {folder.name}")

        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                records.append({
                    "path": path,
                    "source": "SOURCE_A",
                    "original_split": "not_applicable",
                    "label": label,
                })

    # Source C: collect all original Roboflow splits
    for split in ["train", "valid", "test"]:
        split_dir = SOURCE_C / split

        if not split_dir.exists():
            raise FileNotFoundError(split_dir)

        for folder in split_dir.iterdir():
            if not folder.is_dir():
                continue

            original_class = normalize(folder.name)

            if original_class not in MAP_C:
                raise ValueError(f"Unknown Source C class: {folder.name}")

            label = MAP_C[original_class]

            if label is None:
                excluded += sum(
                    1 for p in folder.rglob("*")
                    if p.is_file() and p.suffix.lower() in EXTENSIONS
                )
                continue

            for path in folder.rglob("*"):
                if path.is_file() and path.suffix.lower() in EXTENSIONS:
                    records.append({
                        "path": path,
                        "source": "SOURCE_C",
                        "original_split": split,
                        "label": label,
                    })

    return records, excluded


def main():
    if not SOURCE_A.exists() or not SOURCE_C.exists():
        raise FileNotFoundError("Source A or Source C not found.")

    if OUTPUT.exists():
        raise SystemExit(
            f"{OUTPUT} already exists. Rename or remove it before rerunning."
        )

    records, excluded = collect_images()
    unique = []
    seen = {}
    duplicates = 0

    # Validate images and remove exact duplicate files
    for record in records:
        path = record["path"]

        try:
            with Image.open(path) as image:
                image.verify()

            digest = hashlib.sha256(path.read_bytes()).hexdigest()

        except Exception as error:
            raise RuntimeError(f"Invalid image: {path} — {error}")

        if digest in seen:
            if seen[digest] != record["label"]:
                raise RuntimeError(
                    f"Same image has conflicting labels: {path}"
                )
            duplicates += 1
            continue

        seen[digest] = record["label"]
        record["sha256"] = digest
        unique.append(record)

    # Stratified split by source and class
    groups = defaultdict(list)

    for record in unique:
        groups[(record["source"], record["label"])].append(record)

    splits = {"train": [], "validation": [], "test": []}
    rng = random.Random(SEED)

    for key, group in sorted(groups.items()):
        rng.shuffle(group)

        n_train = round(len(group) * 0.70)
        n_validation = round(len(group) * 0.15)

        splits["train"].extend(group[:n_train])
        splits["validation"].extend(
            group[n_train:n_train + n_validation]
        )
        splits["test"].extend(group[n_train + n_validation:])

    manifest = []

    for split, items in splits.items():
        rng.shuffle(items)

        for item in items:
            source = item["source"].lower()
            digest = item["sha256"]
            extension = item["path"].suffix.lower()

            filename = f"{source}_{digest[:16]}{extension}"
            destination = OUTPUT / split / item["label"] / filename

            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item["path"], destination)

            manifest.append({
                "split": split,
                "label": item["label"],
                "source": item["source"],
                "original_split": item["original_split"],
                "image_path": destination.as_posix(),
                "sha256": digest,
            })

    REPORT.parent.mkdir(parents=True, exist_ok=True)

    with REPORT.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)

    print("\n===== OLIVE-CAD v1.0 =====")
    print(f"Valid unique images: {len(manifest)}")
    print(f"Exact duplicates removed: {duplicates}")
    print(f"Knot disease images excluded: {excluded}")

    for split, items in splits.items():
        counts = Counter(item["label"] for item in items)
        print(f"\n{split.upper()}: {len(items)}")
        for label in ["healthy", "peacock_spot", "aculus_olearius"]:
            print(f"  {label}: {counts[label]}")

    print(f"\nDataset: {OUTPUT}")
    print(f"Manifest: {REPORT}")


if __name__ == "__main__":
    main()
