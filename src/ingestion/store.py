import chromadb
import json
import os
from config import DB_DIR, PROCESSED_DIR, COLLECTION_NAME

def main():
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    data = json.load(open(os.path.join(PROCESSED_DIR, "embedded_chunks.json")))

    collection.upsert(                                        
        ids=[f"{c['metadata']['pdf_name']}_{c['slide_num']}" for c in data],
        embeddings=[c["embedding"] for c in data],
        documents=[c["combined_text"] for c in data],
        metadatas=[c["metadata"] for c in data],               
    )

    print(collection.count())

if __name__ == "__main__": 
    main()