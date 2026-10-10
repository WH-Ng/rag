import chromadb
from ingestion.embedder import generate_embeddings
from typing import List, Dict
from functools import lru_cache # Least Recently Used (LRU) cache
from config import COLLECTION_NAME


@ lru_cache
def get_chroma_collection(db_dir: str):
    
    """
    Connects to local persistent ChromaDB. 
    If you call the function again with the same arguments, 
    it returns the pre-computed result immediately instead of running the code. Thanks for lru_cache
    """
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_collection(name=COLLECTION_NAME)
    
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

    from config import DB_DIR, TOP_K

    for res in search("What is statistical inference?", top_k=TOP_K, db_dir=DB_DIR):

        print(f"Slide {res['metadata']['slide_num']} | distance {res['distance']:.4f}")

