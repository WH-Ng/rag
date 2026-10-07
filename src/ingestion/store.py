import chromadb
import json
import os
from config import DB_DIR, PROCESSED_DIR

client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection("lectures")

data = json.load(open(os.path.json(PROCESSED_DIR, "embedded_chunks.json")))

collection.upsert(                                        
    ids=[f"{c['metadata']['pdf_name']}_{c['slide_num']}" for c in data],
    embeddings=[c["embedding"] for c in data],
    documents=[c["combined_text"] for c in data],
    metadatas=[c["metadata"] for c in data],               
)

print(collection.count())

# client.delete_collection(name="lecture_slides")