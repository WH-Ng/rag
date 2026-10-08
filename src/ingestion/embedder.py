import ollama
from typing import List
import json
import os
import glob
from config import EMBED_MODEL, PROCESSED_DIR

def generate_embeddings(
      texts: List[str],
      batch_size: int=32) -> List[List[float]]:
    
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
       
       batch = texts[i:i + batch_size]

       response = ollama.embed(
            model=EMBED_MODEL,
            input=batch
        )
       
       all_embeddings.extend(response['embeddings'])

    return all_embeddings

def main():
   
   all_processsed_paths = glob.glob(os.path.join(PROCESSED_DIR, "*_chunks.json"))

   for path in sorted(all_processsed_paths):
        
        with open(path, 'r') as f:
                data = json.load(f)

        text_list = [slide['combined_text'].strip() for slide in data]
        embeddings = generate_embeddings(text_list, batch_size=32)

        for slide, embedding in zip(data, embeddings):
            slide['embedding'] = embedding

        print(f"\nSuccessfully generated embeddings for {len(data)} slides.")
        print(f"Vector dimensions: {len(embeddings[0])}")

        output_path = os.path.join(PROCESSED_DIR, "embedded_chunks.json")

        with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)   

        return data

if __name__=="__main__":
    main()
