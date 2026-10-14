import json
from pathlib import Path
import faiss
import numpy as np

MAIN_DIR = Path(__file__).resolve().parent.parent
CHUNKS_PATH = MAIN_DIR / "vector_store" / "chunks.json"
INDEX_PATH = MAIN_DIR / "vector_store" / "sahayaka.faiss"
METADATA_PATH = MAIN_DIR / "vector_store" / "faiss_metadata.json"

def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_PATH}"
        )

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("chunks.json must contain a non-empty list.")

    vectors = []
    metadata = []

    for position, chunk in enumerate(chunks):
        embedding = chunk.get("embedding")

        if not isinstance(embedding, list) or not embedding:
            raise ValueError(f"Chunk at position {position} has no valid embedding.")

        vector = np.asarray(embedding, dtype=np.float32)

        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError(f"Invalid embedding values at position {position}.")

        if np.linalg.norm(vector) == 0:
            raise ValueError(f"Zero-length embedding at position {position}.")

        vectors.append(vector)

        chunk_metadata = chunk.get("metadata") or {}

        metadata.append({
            "id": chunk.get("id", position),
            "text": chunk.get("text", ""),
            "source": chunk_metadata.get("source", "unknown"),
            "page": chunk_metadata.get("page"),
            "chunk": chunk_metadata.get("chunk", position),
        })

    dimensions = {vector.shape[0] for vector in vectors}

    if len(dimensions) != 1:
        raise ValueError(f"Embeddings have inconsistent dimensions: {dimensions}")

    # FAISS requires a 2D, float32 matrix.
    matrix = np.vstack(vectors).astype(np.float32)

    # Normalize vectors so inner product equals cosine similarity.
    faiss.normalize_L2(matrix)

    dimension = matrix.shape[1]

    # Exact nearest-neighbor search using cosine similarity.
    index = faiss.IndexFlatIP(dimension)
    index.add(matrix)

    # Save index and aligned metadata.
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print("FAISS index created successfully.")
    print("Chunks indexed:", index.ntotal)
    print("Embedding dimensions:", dimension)
    print("Index saved to:", INDEX_PATH)
    print("Metadata saved to:", METADATA_PATH)


if __name__ == "__main__":
    main()
