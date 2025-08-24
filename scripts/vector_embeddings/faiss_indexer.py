import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# FAISS index path
FAISS_INDEX_PATH = 'E:/AI learner/database/vector_store/faiss_index.bin'
ID_MAP_PATH = 'E:/AI learner/database/vector_store/id_map.npy'

# Load SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def query_faiss_index(query, top_k = 5):
    """Query the FAISS index to reteieve top-k relevant text chunks."""
    # Load FAISS index and ID mapping
    index = faiss.read_index(FAISS_INDEX_PATH)
    chunk_ids = np.load(ID_MAP_PATH, allow_pickle=True)

    # Generate embedding for the query
    query_embedding = model.encode([query], show_progress_bar=False)[0]

    #Search FAISS index
    distances, indices = index.search(np.array([query_embedding]).astype(np.float32), top_k)

    # Reterieve corrosponding chunk IDs
    results = [(chunk_ids[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
    return results

if __name__ =='__main__':
    # Example query for texting
    test_query = "Give me an introduction to working as a physist"
    results = query_faiss_index(test_query, top_k = 5)
    print("Query results:")
    for chunk_id, distance in results:
        print(f"Chunk ID: {chunk_id}, Distance: {distance}")
        


