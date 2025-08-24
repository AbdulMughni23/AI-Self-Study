import os
from pymongo import MongoClient
from bson.objectid import ObjectId

# MongoBD Connection steup
client = MongoClient('mongodb://localhost:27017/')
db = client['rag_learning_db']
collection = db['learning_materials']

# Directory path for text files
text_dir = 'E:/AI learner/data/text'

def process_text(file_path):
    """Read text file and return metadata"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return{
            'file_type': 'text',
            'file_path': file_path,
            'extracted_text': text,
            'unique_id': str(ObjectId())
        }
    except Exception as e:
        print(f"Error procesing text file {file_path}: {e}")
        return None

def populate_text_database():
    """process all the text in the files in the directory and store in MongoDB (localhost)"""
    os.makedirs(text_dir, exist_ok = True)
    for filename in os.listdir(text_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(text_dir, filename)
            data = process_text(file_path)
            if data:
                collection.insert_one(data)
                print(f"stored text file: {filename}") 


if __name__ == '__main__':
    populate_text_database()

