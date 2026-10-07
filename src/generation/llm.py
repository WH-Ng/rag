from retrieval.search import search
from generation.prompts import get_rag_messages
import ollama

def generate_answer(
        query_text: str,
        top_k: int,
        db_dir: str,
        model_name: str):
    
    hits = search(query_text=query_text, top_k=top_k, db_dir=db_dir)

    messages = get_rag_messages(query_text, hits)

    response=ollama.chat(
        model=model_name,
        messages=messages,
        options={
            "temperature": 0,
            "num_ctx": 8192
        }
    )

    output = response["message"]["content"]

    return output

if __name__=="__main__":
    question = "An interesting fact about statistical inference"
    db_path = "/Users/wayne/Personal_Project/rag/storage/chroma_db"
    model_name = "qwen2.5:7b"

    answer = generate_answer(question, top_k=3, db_dir=db_path, model_name=model_name)

    print(answer)
