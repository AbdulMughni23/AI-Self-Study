import os
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer 
import faiss
import numpy as np

# MongoDB connection setup
client = MongoClient("mongodb://localhost:27017/")
db = client["rag_learning_db"]
collection = db["learning_materials"]


# FAISS index path
FAISS_INDEX_PATH = 'database/vector_store/faiss_index.bin'
ID_MAP_PATH = 'database/vector_store/id_map.npy'

#load SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings():
    """Generate embeddings for text chunksand and store in FAISS index"""
    # Reterive all chunks from MongoDB
    cursor = collection.find({'file_type': 'text_chunk'})
    texts = []
    chunk_ids = []

    for doc in cursor:
        texts.append(doc['chunk_text'])
        chunk_ids.append(doc['chunk_id'])

    if not texts:
        print("No text chunks found in the database")
        return

    #Generate embeddings
    print(f"Generating embeddings for {len(texts)} chunks....")
    embeddings =  model.encode(texts, batch_size=32, show_progress_bar = True)

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))

    # Save FAISS index
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok = True )
    faiss.write_index(index, FAISS_INDEX_PATH)

    # save chunk_id mapping 
    np.save(ID_MAP_PATH, np.array(chunk_ids))
    print(f"FAISS index and ID mapping saved to {FAISS_INDEX_PATH} and {ID_MAP_PATH}")

if __name__ == '__main__':
    generate_embeddings()