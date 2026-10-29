import ollama
from config import CHAT_MODEL

def generate_answer(prompt: str) -> str:
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1},
    )
    return response["message"]["content"].strip()
