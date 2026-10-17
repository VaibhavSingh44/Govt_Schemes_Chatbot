import json
import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
import os
from pathlib import Path

CURR_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = CURR_DIR / ".env"
INDEX_PATH = CURR_DIR / "vector_store/sahayaka.faiss"
CHUNKS_PATH = CURR_DIR / "vector_store/chunks.json"

load_dotenv(dotenv_path=ENV_PATH)


api_key = os.getenv("My_API_Key")
embedding_model = os.getenv("My_Embedding_Model")

if not api_key:
    raise ValueError("My_API_Key was not found in .env")

if not embedding_model:
    raise ValueError("My_Embedding_Model was not found in .env")

client = genai.Client(api_key = api_key)


def embed_query(query):
    """Convert a user's question into an embedding vector."""

    response = client.models.embed_content(
        model=embedding_model,
        contents=query
    )

    if not response.embeddings:
        raise ValueError("Gemini returned no query embedding.")

    embedding = response.embeddings[0].values

    if not embedding:
        raise ValueError("The query embedding is empty.")

    return embedding


def search(query, top_k=3):
    """Retrieve the top-k most similar document chunks."""

    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_PATH}"
        )

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_PATH}"
        )

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    index = faiss.read_index(str(INDEX_PATH))

    with open(CHUNKS_PATH,"r",encoding="utf-8") as file:

        chunks = json.load(file)

    if index.ntotal != len(chunks):
        raise ValueError(
            f"Index contains {index.ntotal} vectors, "
            f"but chunks.json contains {len(chunks)} chunks."
        ) 

    # Generate query embedding.
    query_embedding = embed_query(query)   

    query_vector = np.asarray(
        [query_embedding],
        dtype=np.float32
    )

    if query_vector.shape[1] != index.d:
        raise ValueError(
            f"Query dimension {query_vector.shape[1]} "
            f"does not match index dimension {index.d}."
        )

    if not np.isfinite(query_vector).all():
        raise ValueError("Query embedding contains invalid values.")

    # Match the normalization used when building the index.
    faiss.normalize_L2(query_vector)

    # Search only for as many results as the index contains.
    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_vector, k)

    results = []

    for score, index_id in zip(scores[0], indices[0]):
        if index_id == -1:
            continue

        result = chunks[int(index_id)].copy()
        result["score"] = float(score)
        results.append(result)

    return results


if __name__ == "__main__":
    query = input("Search knowledge base: ")

    try:
        results = search(query, top_k=3)

        print("\n========== SEARCH RESULTS ==========")

        if not results:
            print("No matching chunks found.")

        for i, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})

            print(f"\n--- RESULT {i} ---")
            print("Source:", metadata.get("source", "unknown"))
            print("Page:", metadata.get("page", "unknown"))
            print("Chunk:", metadata.get("chunk", "unknown"))
            print(f"Cosine similarity score: {result['score']:.4f}")
            print("\nText:")
            print(result.get("text", ""))

    except Exception as error:
        print(f"\nSearch failed: {error}")
        raise