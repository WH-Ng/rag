import chromadb
from ingestion.embedder import generate_embeddings
from typing import List, Dict

def get_chroma_collection(db_dir: str):
    
    """
    Connects to local persistent ChromaDB
    """
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_collection(name="lectures")
    
    return collection

def search(
        query_text: str,
        top_k: int,
        db_dir: str) -> List[Dict]:
    
    query_vector = generate_embeddings([query_text])[0]

    collection = get_chroma_collection(db_dir)

    results = collection.query(query_embeddings=[query_vector], n_results=top_k)

    hits = []

    if results and results.get('ids'):
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results.get("distances", [[]])[0]

        for i in range(len(ids)):
            hits.append(
                {
                    "id": ids[i],
                    "text": docs[i],
                    "metadata": metas[i],
                    "distance": distances[i] if distances else None,
                }
            )

    return hits

if __name__ == "__main__":

    db_dir = "/Users/wayne/Personal_Project/rag/storage/chroma_db"

    for res in search("What is statistical inference?", top_k=3, db_dir=db_dir):

        print(f"Slide {res['metadata']['slide_num']} | distance {res['distance']:.4f}")

