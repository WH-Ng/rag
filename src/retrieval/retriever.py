import chromadb
import ollama

EMBED_MODEL = "bge-m3"           
CHROMA_PATH = "/Users/wayne/Personal_Project/rag/storage/chroma_db" 
COLLECTION_NAME = "lectures"     

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(COLLECTION_NAME)

def retrieve(question: str, k: int = 4) -> list[dict]:
    """Return the k slides most relevant to the question."""

    question_embedding = ollama.embed(model=EMBED_MODEL, input=[question])["embeddings"][0]

    results = collection.query(query_embeddings=[question_embedding], n_results=k)

    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def main():
    question = "What are the assessments in this course?"   # change this to something you know is in your slides
    for r in retrieve(question):
        print(f"slide {r['metadata']['slide_num']:>2}  distance={r['distance']:.3f}")
        print("   ", r["text"].strip()[:150].replace("\n", " "), "\n")


if __name__ == "__main__":
    main()