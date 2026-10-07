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

if __name__=="__main__":
    question = "An interesting fact about statistical inference"
    db_path = "/Users/wayne/Personal_Project/rag/storage/chroma_db"
    model_name = "qwen2.5:7b"

    answer = generate_answer(question, top_k=3, db_dir=db_path, model_name=model_name)

    print(answer)
