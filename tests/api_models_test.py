import os
from google import genai
from pathlib import Path
from dotenv import load_dotenv

curr_dir = Path(__file__).resolve().parent.parent
env_path = curr_dir / ".env"

load_dotenv(dotenv_path=env_path)

my_api_key = os.getenv("My_API_Key")

client = genai.Client(api_key=my_api_key)

print("--- Chat / Content Generation Models ---")
for model in client.models.list():
    if "generateContent" in model.supported_actions:
        print(f"Model Name: {model.name}")

print("\n--- Embedding Models ---")
for model in client.models.list():
    if "embedContent" in model.supported_actions:
        print(f"Model Name: {model.name}")