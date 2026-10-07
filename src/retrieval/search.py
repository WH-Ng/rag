import chromadb
from ingestion.embedder import generate_embeddings
import json
from typing import List, Dict

def get_chroma_collection(db_dir: str):
    
    """
    Connects to local persistent ChromaDB
    """
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_or_create_collection(name="lecture_slides")
    
    return collection

def search(
        query_text: str,
        top_k: int,
        db_dir=str) -> List[Dict]:
    
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

if __name__=="__main__":

    embeddings_path = "/Users/wayne/Personal_Project/rag/data/processed/Lectures/embedded_chunks.json"
    db_dir = "/Users/wayne/Personal_Project/rag/storage/chroma_db"

    with open(embeddings_path, "r") as f:
        chunks = json.load(f)

    collection = get_chroma_collection(db_dir)
    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for item in chunks:
        pdf_name = item["metadata"]["pdf_name"]
        slide_num = item["slide_num"]

        chunk_id = f"{pdf_name}_slide_{slide_num}"

        ids.append(chunk_id)
        embeddings.append(item["embedding"])
        documents.append(item["combined_text"])

        meta = item["metadata"]
        meta["image_path"] = item['image_path']
        metadatas.append(meta)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"Loaded {len(ids)} slides into ChromaDB!")

    results = search("What is statistical inference?", top_k=3, db_dir=db_dir)

    print(results)

    print("\n--- Search Results ---")
    for res in results:
        print(
            f"Slide {res['metadata']['slide_num']} | Score: {res['distance']:.4f} | Path: {res['metadata'].get('image_path')}"
        )


