from flask import Flask, request, jsonify
from pymongo import MongoClient
from scripts.Ai_agent.query_handler import query_faiss_index, query_ollama

app = Flask(__name__)

# MongoDB connection setup
client = MongoClient("mongodb://localhost:27017/")
db = client["rag_learning_db"]
collection = db["learning_materials"]


@app.route('/get_topic_response', methods=['POST'])
def get_topic_response():
    """Generate initial AI response for a selected topic."""
    try:
        topic = request.json.get('topic')
        if not topic:
            return jsonify({'error': 'Topic is required'}), 400
        
        chunks = query_faiss_index(topic, top_k=1)
        context = "\n".join([chunk['text'] for chunk in chunks]) if chunks else ""
        prompt = f"""You are an expert educational content creator 
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
        return jsonify({'response': response or "Failed to generate response."})
    except Exception as e:
        print(f"Error in get_topic_response: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/get_ai_response', methods=['POST'])
def get_ai_response():
    """Generate AI response to a user prompt, with topic for context."""
    data = request.json
    prompt = data.get('prompt')
    topic = data.get('topic')
    
    if not prompt or not topic:
        return jsonify({"error": "Prompt is required"}), 400
    
    chunks = query_faiss_index(topic, top_k=1)
    context = "\n".join([chunk['text'] for chunk in chunks]) if chunks else ""
    ai_prompt = f"""
    You are an expert educational content creator 
    you have to respond with an API respomse to an API call to generate a concise introduction for a given topic based on the provided context.
    for the topic: "{topic}"
    use the following guidelines for your response:
    **HTML Processing:** Parse and distill key ideas from HTML content, discarding markup while preserving semantic meaning.  

    **Context:**  
    {context}

    now answer the following question based on the context:

    {prompt}

    **Output Instructions:**  
    - Use HTML format for the respose
    - use <span class="katex-mathml"> for any math expressions if required in the response
    - Every claim must derive from context, with NO external knowledge or words to be added  
    - Ensure clarity and conciseness 
    
    """
    
    response = query_ollama(ai_prompt)
    if not response:
        return jsonify({"error": "Failed to generate response"}), 500
    
    return jsonify({"response": response or "Failed to generate response"})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0',port=5000)
