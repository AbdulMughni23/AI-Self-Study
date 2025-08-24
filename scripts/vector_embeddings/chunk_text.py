import re
from pymongo import MongoClient
# from torch._dynamo.bytecode_transformation import unique_id


# MongoDB connection setup
client = MongoClient("mongodb://localhost:27017/")
db = client["rag_learning_db"]
collection = db["learning_materials"]

def chunk_text(text, min_words = 500, max_words = 1000):
    """Chunk text into segments of 500 - 1000 words"""
    # split text into sentences 
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    print(f"")
    chunks = []
    current_chunk = []
    word_count = 0

    for sentence in sentences:
        words = sentence.split()
        word_count += len(words)
        current_chunk.append(sentence)

        if word_count >= min_words:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            word_count = 0

    # Handel remaining sentences 
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    # Ensure chunk don't excede max_words
    final_chunks = []
    for chunk in chunks:
        words = chunk.split()
        if len(words) > max_words:
            # Split large chunks into smaller ones
            for i in range(0, len(words), max_words):
                final_chunks.append(' '.join(words[i:i + max_words]))
        else:
            final_chunks.append(chunk)

    return final_chunks



def process_database_chunks():
    """Reterive text from MongoDB, chunk it, and store chunks back back with unique IDs"""
    cursor = collection.find({'file_type': {'$in': ['pdf', 'text']}, 'extracted_text': {'$ne': ''} })
    for doc in cursor:
        text = doc['extracted_text']
        unique_id = doc['unique_id']
        chunks = chunk_text(text)
        # Store each chunk as a new document
        for i, chunk in enumerate(chunks):
            chunk_doc = {
                'file_type': 'text_chunk',
                'original_id': unique_id,
                'chunk_id': f"{unique_id}_{i}",
                'chunk_text': chunk,
                'word_count': len(chunk.split())
            }
            collection.insert_one(chunk_doc)
            print(f"stored chunk {chunk_doc['chunk_id']} for original ID {unique_id}")

if __name__ == '__main__':
    process_database_chunks()

