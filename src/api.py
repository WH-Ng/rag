import logging
import time
import ollama
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
# field adds extra validation alongside type hints

from config import DB_DIR, EMBED_MODEL, LLM_MODEL, TOP_K
from rag import answer
from retrieval.search import get_chroma_collection

logger = logging.getLogger("rag-api")

class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=TOP_K, ge=1, le=20)

class Source(BaseModel):
    pdf_name: str
    slide_num: int

class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    timings_ms: dict[str, int]

# Creates API application
app = FastAPI(title="Lecture Slides RAG API")

@app.get("/health")
def health():

    count = get_chroma_collection(DB_DIR).count()
    models = {m.model for m in ollama.list().models}

    missing_models = []

    for model in (EMBED_MODEL, LLM_MODEL):
        if model not in models and f"{model}:latest" not in models:
            missing_models.append(model)

    if missing_models:
        raise HTTPException(status_code=503, 
                            detail=f"Models not available: {missing_models}")
        # 400-semi codes = "its user's fault"
        # 500-semi codes = "servers fault"
            # 503 is service unavail
    
    return {"status": "ok", "chunks_indexed": count}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    
    result = answer(req.question, top_k=req.top_k)

    return result


