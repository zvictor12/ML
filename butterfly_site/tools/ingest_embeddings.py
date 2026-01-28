import argparse
import csv
import json
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGES_PER_SPECIES = 5


def sort_key(value: str):
    return (0, int(value)) if value.isdigit() else (1, value.lower())

def load_jsonl_metadata(path: Path):
    if not path.exists():
        return {}

    metadata = {}
    with path.open(encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            if not line.strip():
                continue
            item = json.loads(line)
            item_id = str(item.get("id"))
            if not item_id:
                continue
            metadata[item_id] = {
                "species": item.get("species", "Unknown"),
            }
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Upload embeddings to Qdrant")
    parser.add_argument("--embeddings", required=True, help="Path to .npy embeddings")
    parser.add_argument("--metadata-jsonl", help="Optional JSONL with id,species,image_url")
    parser.add_argument("--images-dir", help="Optional directory with images to build payloads")
    parser.add_argument("--collection", default="butterflies")
    parser.add_argument("--url", default="http://localhost:6333")
    args = parser.parse_args()

    embeddings = np.load(args.embeddings)
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array")
    # !!!
    embeddings = embeddings.astype(np.float32, copy=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)

    embeddings /= norms

    metadata = {}
    if args.metadata_jsonl:
        metadata.update(load_jsonl_metadata(Path(args.metadata_jsonl)))

    folders = []
    images_by_folder = []
    if args.images_dir:
        images_root = Path(args.images_dir)
        if images_root.exists():
            folders = sorted(
                [p for p in images_root.iterdir() if p.is_dir()],
                key=lambda p: sort_key(p.name),
            )
            for folder in folders:
                images = sorted(
                    [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
                    key=lambda p: sort_key(p.stem),
                )
                images_by_folder.append(images)

    client = QdrantClient(url=args.url)
    vector_size = embeddings.shape[1]

    if args.collection not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=args.collection,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    points = []
    for index, vector in enumerate(embeddings):
        payload = metadata.get(str(index), {"species": "Unknown"})
        if folders:
            folder_index = index // IMAGES_PER_SPECIES
            image_index = index % IMAGES_PER_SPECIES
            if folder_index < len(folders) and image_index < len(images_by_folder[folder_index]):
                image_path = images_by_folder[folder_index][image_index]
                if "species" not in payload or payload["species"] == "Unknown":
                    payload["species"] = folders[folder_index].name
                relative_path = image_path.relative_to(images_root).as_posix()
                payload["image_url"] = f"/images/{relative_path}"
        points.append(models.PointStruct(id=index, vector=vector.tolist(), payload=payload))

    client.upsert(collection_name=args.collection, points=points)
    print(f"Uploaded {len(points)} embeddings to {args.collection}")


if __name__ == "__main__":
    main()
