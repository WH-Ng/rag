import ollama

LLM_MODEL = "qwen2.5:7b"     # any text model you've pulled; llama3.2 if RAM is tight

SYSTEM_PROMPT = """You are a teaching assistant answering questions about lecture slides.
Answer using ONLY the lecture excerpts provided.
After each fact, cite the slide it came from, e.g. (01_Introduction.pdf, slide 5).
If the excerpts do not contain the answer, say "I couldn't find that in the lecture slides."
Do not use outside knowledge."""


def build_context(chunks: list[dict]) -> str:
    """Label each retrieved slide with its source so the LLM can cite it."""
    parts = []
    for c in chunks:
        meta = c["metadata"]
        parts.append(
            f"<excerpt source=\"{meta['pdf_name']}, slide {meta['slide_num']}\">\n"
            f"{c['text'].strip()}\n"
            f"</excerpt>"
        )
    return "\n\n".join(parts)

def generate(question: str, chunks: list[dict]) -> str:
    """Ask the LLM to answer the question using only the retrieved chunks."""
    user_prompt = f"{build_context(chunks)}\n\nQuestion: {question}"

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={
            "temperature": 0,    # same question → same answer; good for testing
            "num_ctx": 8192,     # context window size, see note below
        },
    )
    return response["message"]["content"]


def main():
    from retrieval.retriever import retrieve

    question = "What were the scatterplots about?"
    chunks = retrieve(question)
    print(generate(question, chunks))


if __name__ == "__main__":
    main()