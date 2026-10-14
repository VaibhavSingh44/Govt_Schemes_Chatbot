#  Sementic search

import json

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import os


load_dotenv()


api_key = os.getenv("My_API_Key")
embedding_model = os.getenv("My_Embedding_Model")


client = OpenAI(
    api_key=api_key,
    base_url="https://integrate.api.nvidia.com/v1"
)


INDEX_PATH = "../vector_store/index.faiss"
CHUNKS_PATH = "../vector_store/chunks.json"


def embed_query(query):

    response = client.embeddings.create(
        model=embedding_model,
        input=query
    )

    return response.data[0].embedding


def search(query, top_k=3):

    index = faiss.read_index(INDEX_PATH)

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    query_vector = np.array(
        [embed_query(query)],
        dtype="float32"
    )

    distances, indices = index.search(
        query_vector,
        top_k
    )

    results = []

    for distance, index_id in zip(
        distances[0],
        indices[0]
    ):

        if index_id == -1:
            continue

        result = chunks[index_id].copy()

        result["distance"] = float(distance)

        results.append(result)

    return results


if __name__ == "__main__":

    query = input("Search knowledge base: ")

    results = search(query)

    print("\n========== RESULTS ==========")

    for i, result in enumerate(
        results,
        start=1
    ):

        print(f"\n--- RESULT {i} ---")

        print(
            "Source:",
            result["metadata"]["source"]
        )

        print(
            "Page:",
            result["metadata"]["page"]
        )

        print(
            "Distance:",
            result["distance"]
        )

        print("\nText:")
        print(result["text"])


"""
========== RESULTS ==========

--- RESULT 1 ---
Source: test.pdf
Page: 1
Distance: ...

Text:
...

--- RESULT 2 ---
Source: test.pdf
Page: 2
Distance: ...

Text:
...
"""