import json
import os

import faiss
import numpy as np
from pathlib import Path

curr_dir = Path(__file__).resolve().parent.parent

INPUT_PATH = curr_dir/"vector_store" / "chunks.json"
INDEX_PATH = curr_dir/"vector_store" / "index.faiss"


def main():

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    if not chunks:
        raise ValueError("No chunks found")

    vectors = np.array(
        [
            chunk["embedding"]
            for chunk in chunks
        ],
        dtype="float32"
    )

    dimension = vectors.shape[1]

    print("Number of vectors:", vectors.shape[0])
    print("Vector dimension:", dimension)

    index = faiss.IndexFlatL2(dimension)

    index.add(vectors)

    os.makedirs(
        curr_dir / "vector_store",
        exist_ok=True
    )

    faiss.write_index(
        index,
        INDEX_PATH
    )

    print("\nFAISS index created successfully!")
    print("Index:", INDEX_PATH)
    print("Vectors stored:", index.ntotal)


if __name__ == "__main__":
    main()
