import chromadb
import json

db_dir = "/Users/wayne/Personal_Project/rag/storage/chroma_db"
data_json = "/Users/wayne/Personal_Project/rag/data/processed/Lectures/embedded_chunks.json"

client = chromadb.PersistentClient(path=db_dir)
collection = client.get_or_create_collection("lectures")

data = json.load(open(data_json))

collection.upsert(                                        
    ids=[f"{c['metadata']['pdf_name']}_{c['slide_num']}" for c in data],
    embeddings=[c["embedding"] for c in data],
    documents=[c["combined_text"] for c in data],
    metadatas=[c["metadata"] for c in data],               
)

print(collection.count())

# client.delete_collection(name="lecture_slides")