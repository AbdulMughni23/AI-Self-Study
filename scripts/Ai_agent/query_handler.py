import faiss
import numpy as np
from olmocr import prompts
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import requests

# from scripts.vector_embeddings.faiss_indexer import chunk_id

# FAISS index path
FAISS_INDEX_PATH = 'E:/AI learner/database/vector_store/faiss_index.bin'
ID_MAP_PATH = 'E:/AI learner/database/vector_store/id_map.npy'

# MongoDB connection setup
client = MongoClient("mongodb://localhost:27017/")
db = client["rag_learning_db"]
collection = db["learning_materials"]

# Load SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

#Ollama API endpoint
OLLAMA_API_URI = 'http://localhost:1143/api/generate'

def query_faiss_index(query, top_k =5):
    """Query the FAISS index to reterieve top_k relevant text chunks"""
    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
        chunk_ids = np.load(ID_MAP_PATH, allow_pickle=True)

        query_embedding = model.encode([query], show_progress_bar=False)[0]
        distances, indices = index.search(np.array([query_embedding]).astype(np.float32), top_k)

        #reterieve chunk texts from MongoDB
        results = []
        for idx in indices[0]:
            chunk_id = chunk_ids[idx]
            chunk_doc = collection.find_one({'chunk_id': chunk_id})
            if chunk_doc:
                results.append({
                    'chunk_id': chunk_id,
                    'text':chunk_doc['chunk_text'],
                    'distance': float(distances[0][list(indices[0]).index(idx)])
                })
        return results

    except Exception as e:
        print(f"Error querying FAISS index: {e}")
        return []

def query_ollama(prompt, model_name='mistral'):
    """Query the Ollama API with a prompt and return the response."""
    try:
        payload = {
            'model': model_name,
            'prompt': prompt,
            'stream': False
        }
        response = requests.post(OLLAMA_API_URI, json=payload)
        response.raise_for_status()
        return response.json()['response']

    except Exception as e:
        print(f"Error querying Ollama: {e}")
        return None

def generate_topic_introduction(topic):
    """Generate a topic introduction using reterived chunks as context"""
    chunks = query_faiss_index(topic, top_k =5)
    if not chunks:
        return "No relevant information found for this topic."

    context = "\n".join([chunk['text'] for chunk in chunks])
    prompt = f"""**Task:** Generate a strictly concise introduction (100-150 words) for `{topic}` using the provided context.  
                    **Key Requirements:**  
                    1. **MathML Handling:** Ignore all mathematical equations and formulae (any <math> tags). Extract only conceptual insights from surrounding text.  
                    2. **HTML Processing:** Parse and distill key ideas from HTML content, discarding markup while preserving semantic meaning.  
                    3. **Introduction Style:**  
                       - First sentence must hook with broad relevance  
                       - Use intuitive analogies/plain language (NO technical jargon)  
                       - Explain "why it matters" before "what it is"  
                       - Exclude all mathematical notations/symbols  
                    4. **Structure:**  
                       - Opening impact statement → Core concept → Real-world significance → Practical applications  
                       - Maintain neutral, authoritative tone  

                    **Context:**  
                    {context}

                    **Output Instructions:**  
                    - Word count: 110 ± 10 words  
                    - First/last sentences must be memorable and self-contained  
                    - Every claim must derive from context, with NO external knowledge  
                    - Use active voice and avoid passive constructions  
                    """

    response = query_ollama(prompt)
    if not response:
        return []

if __name__ == "__main__":
    #example usage of testing 
    topic = "working as a physicist"
    print("Generating introduction....")
    intro = generate_topic_introduction(topic)
    print(f"Introduction: /n {intro}/n")
