import os

from pathlib import Path
from dotenv import load_dotenv
from google import genai

base_dir = Path(__file__).resolve().parent.parent.parent
env_path = base_dir / '.env'

load_dotenv(dotenv_path=env_path)

api_key = os.getenv("My_API_Key")
model = os.getenv("My_Model")

if not api_key:
    raise ValueError("API_KEY was not found in .env")

if not model:
    raise ValueError("MODEL was not found in .env")

print("Model:", model)
print("API key loaded:", bool(api_key))

client = genai.Client(api_key = api_key)

question = input("Ask Sahayaka AI: ")

print("MODEL:", model)
print("QUESTION:", question)

# ----->> Generic chatbot system 

# response = client.models.generate_content(
#     model = model,
#     contents = f"Provide a complete, direct, and concise answer without cutting off: {question}",
#     config = {
#         "max_output_tokens" : 800,
#         "temperature" : 0.3,
#     }
# )


# ----->> Integrating System Prompt

from prompts import SYSTEM_PROMPT

response = client.models.generate_content(
    model = model,
    contents = f"Provide a complete, direct, and concise answer without cutting off: {question}",
    config = {
        "system_instruction" : SYSTEM_PROMPT,
        "max_output_tokens" : 800,
        "temperature" : 0.3
    }
)

if response.text:
    answer = response.text
else:
    answer = "Sorry, I couldn't generate a complete answer. Please try phrasing your question differently."


print()
print("Sahayaka AI: ")
print(answer)

