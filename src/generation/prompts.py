from typing import List, Dict, List, Any

def format_slide_context(hits: List[Dict]) -> str:
    
    context_blocks = []

    for hit in hits:
        print(hit.keys())
        slide_num = hit["metadata"].get("slide_num", "Unknown")
        pdf_name = hit['metadata'].get("pdf_name", "Unknown")
        text = hit["text"]

        block = f"[Source: {pdf_name} - Slide {slide_num}\n{text}]"
        context_blocks.append(block)

    cb = "\n\n---\n\n".join(context_blocks)

    return cb

def get_rag_messages(query_text: str, hits: List[Dict]) -> List[Dict[str, str]]:
    
    context_str = format_slide_context(hits)

    system_prompt = (
        "You are an expert university teaching assistant.\n"
        "Answer the student's question strictly using the provided slide context.\n"
        "Always cite the exact slide sources (e.g., [01_Introduction.pdf - Slide 3]) when stating facts.\n"
        "If the context does not contain enough information to answer, state that clearly."
    )

    user_prompt = (
        f"--- SLIDE CONTEXT ---\n{context_str}\n\n"
        f"--- QUESTION ---\n{query_text}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

if __name__=="__main__":
    import json
    path = "/Users/wayne/Personal_Project/rag/data/processed/Lectures/embedded_chunks.json"

    with open(path, 'r') as f:
        data = json.load(f)

    context_blocks = format_slide_context(data)

    print(context_blocks)