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
OLLAMA_API_URI = 'http://localhost:11434/api/generate'

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
    chunks = query_faiss_index(topic, top_k =1)
    if not chunks:
        return "No relevant information found for this topic."
    
    print(f"Reterived {len(chunks)} chunks for topic '{topic}'")
    # for i, chunk in enumerate(chunks):
    #     print(f"Chunk {i+1} (Distance: {chunk['distance']:.4f}): {chunk['text'][:500]}...")

    context = "\n".join([chunk['text'] for chunk in chunks])
    prompt = f"""
    You are an expert educational content creator 
    you have to respond with an API respomse to an API call to generate a concise introduction for a given topic based on the provided context.
    for the topic: "{topic}"
    use the following guidelines for your response:
    1. **HTML Processing:** Parse and distill key ideas from HTML content, discarding markup while preserving semantic meaning.  
    2. **Introduction Style:**  
       - keep the introduction academic and informative
       - Avoid conversational or casual tones
       - Avoid highly creative or narrative styles
    3. **Structure:**  
       - Brief introduction → Core concept that will be learnt → Practical applications  
       - Maintain an academic tone throughout  

    **Context:**  
    {context}

    **Output Instructions:**  
    - Use HTML format for the respose
    - use <span class="katex-mathml"> for any math expressions if required in the response
    - Every claim must derive from context, with NO external knowledge or words to be added  
    - Ensure clarity and conciseness 
    """

    response = query_ollama(prompt)
    if not response:
        return []
    else:
        return response

if __name__ == "__main__":
    #example usage of testing 
    topic = "working as a physicist"
    print("Generating introduction....")
    intro = generate_topic_introduction(topic)
    print(f"Introduction: /n {intro}/n")
