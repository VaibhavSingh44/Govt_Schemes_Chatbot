import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("My_API_Key")
embedding_model = os.getenv("My_Embedding_Model")

if not api_key:
    raise ValueError("My_API_Key was not found")

if not embedding_model:
    raise ValueError("My_Embedding_Model was not found")


client = OpenAI(
    api_key=api_key,
    base_url="https://integrate.api.nvidia.com/v1"
)


text = "PMFBY provides crop insurance support to farmers."


print("Embedding model:", embedding_model)
print("Creating embedding...")


response = client.embeddings.create(
    model=embedding_model,
    input=text
)


embedding = response.data[0].embedding


print("Embedding created successfully!")
print("Vector length:", len(embedding))
print("First 5 values:", embedding[:5])