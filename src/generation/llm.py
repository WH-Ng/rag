import ollama
from generation.prompts import get_rag_messages

def generate_answer(query_text: str, hits: list[dict], model_name: str) -> str:
    messages = get_rag_messages(query_text, hits)
    response = ollama.chat(
        model=model_name,
        messages=messages,
        options={"temperature": 0, "num_ctx": 8192},
    )

    output = response["message"]["content"]
    return output