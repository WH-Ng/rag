from typing import List, Dict
import textwrap

def format_slide_context(hits: List[Dict]) -> str:
    
    context_blocks = []

    for hit in hits:
        slide_num = hit["metadata"].get("slide_num", "Unknown")
        pdf_name = hit['metadata'].get("pdf_name", "Unknown")
        text = hit["text"]

        block = f"[Source: {pdf_name} - Slide {slide_num}]\n{text}"
        context_blocks.append(block)

    cb = "\n\n---\n\n".join(context_blocks)

    return cb

def get_rag_messages(query_text: str, hits: List[Dict]) -> List[Dict[str, str]]:
    
    context_str = format_slide_context(hits)

    system_prompt = textwrap.dedent("""
        You are a teaching assistant answering questions about lecture slides.
        Answer using ONLY the lecture excerpts provided.
        After each fact, cite the slide it came from, e.g. (01_Introduction.pdf, slide 5).
        If the excerpts do not contain the answer, say "I couldn't find that in the lecture slides."
        Do not use outside knowledge.
        """).strip()

    user_prompt = (
        f"--- SLIDE CONTEXT ---\n{context_str}\n\n"
        f"--- QUESTION ---\n{query_text}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]