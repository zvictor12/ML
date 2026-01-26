import argparse
import csv
import json
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models


def load_csv_metadata(path: Path):
    if not path.exists():
        return {}

    metadata = {}
    with path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            metadata[str(row["id"])] = {
                "species": row.get("species", "Unknown"),
                "image_url": row.get("image_url"),
            }
    return metadata


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
                "image_url": item.get("image_url"),
            }
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Upload embeddings to Qdrant")
    parser.add_argument("--embeddings", required=True, help="Path to .npy embeddings")
    parser.add_argument("--metadata", help="Optional CSV with id,species,image_url")
    parser.add_argument("--metadata-jsonl", help="Optional JSONL with id,species,image_url")
    parser.add_argument("--collection", default="butterflies")
    parser.add_argument("--url", default="http://localhost:6333")
    args = parser.parse_args()

    embeddings = np.load(args.embeddings)
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array")

    metadata = {}
    if args.metadata:
        metadata.update(load_csv_metadata(Path(args.metadata)))
    if args.metadata_jsonl:
        metadata.update(load_jsonl_metadata(Path(args.metadata_jsonl)))

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
        points.append(models.PointStruct(id=index, vector=vector.tolist(), payload=payload))

    client.upsert(collection_name=args.collection, points=points)
    print(f"Uploaded {len(points)} embeddings to {args.collection}")


if __name__ == "__main__":
    main()
