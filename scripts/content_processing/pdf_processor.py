from logging import NullHandler
import os
from pypdf import PdfReader
import torch
import base64
from io import BytesIO
import numpy as np

from pdf2image import convert_from_path
from PIL import Image
from pymongo import MongoClient
from bson.objectid import ObjectId

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text

# MongoDB connection setup
client = MongoClient("mongodb://localhost:27017/")
db = client["rag_learning_db"]
collection = db["learning_materials"]

# Directory containing the PDF files
pdf_dir = "E:/AI learner/data/pdfs/"

model = Qwen2VLForConditionalGeneration.from_pretrained("allenai/olmOCR-7B-0225-preview", torch_dtype = torch.bfloat16).eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_pdf_page_count(pdf_path):
    try:
        pdf = PdfReader(pdf_path)
        return len(pdf.pages)
    except Exception as e:
        print(f"error counting pages: {e}")
        return None


def process_pdf(file_path):
    """Extract text form pdf using OlmOCR and return metadata"""
    try:
        # convert PDF to image (as olmOCT supports only images)
        num_pages = get_pdf_page_count(file_path)
        
        
        comp_text = ''
        for page_num in range(num_pages):
            print(f"processing page {page_num+1} of {file_path} with olmOCR")
            img = render_pdf_to_base64png(file_path, page_num+1, target_longest_image_dim=1024)

            #build teh prompt, using document metadata
            anchor_text = get_anchor_text(file_path, page_num+1, pdf_engine="pdfreport", target_length= 4000)
            prompt = build_finetuning_prompt(anchor_text)

            #build the full prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text":prompt},
                        {"type":"image_url", "image_url": {"url": f"data:image/png;base64, {img}"}}
                    ]
                }
            ]

            # applt the chat template and processor 
            text = processor.apply_chat_template(messages, tokenize =False, add_generation_prompt = True)
            main_image = Image.open(BytesIO(base64.b64decode(img)))

            print(f"main image created")

            inputs = processor(
                text = [text],
                images=[main_image],
                padding = True,
                return_tensors = "pt"
                )
            inputs = {key: value.to(device) for (key, value) in inputs.items()}
            print("inputs generated")
            #generate outputs
            output = model.generate(
                **inputs,
                temperature = 0.8,
                max_new_tokens = 150,
                num_return_sequences = 1,
                do_sample = True,
                )

            print(f"outputs generated")

            #decode the output
            prompt_length = inputs["input_ids"].shape[1]
            print(f"prompt length = {prompt_length}")

            new_token = output[:, prompt_length]
            text_output = processor.tokenizer.batch_decode(new_token, skip_special_tokens = True)

            page_text = text_output[0]
            comp_text += page_text + "\n"
        return{
            'file_type': 'pdf',
            'file_path': file_path,
            'extracted_text': comp_text,
            'unique_id': str(ObjectId)
        }

    except Exception as e:
        print(f"Error procesing PDF {file_path}: {e}")
        return None
    
def populate_pdf_database():
    """process all PDF files in the data/pdfs/ directory and store in MongoDB (localhost)"""
    os.makedirs(pdf_dir, exist_ok=True)
    for filename in os.listdir(pdf_dir):
        if filename.endswith('pdf'):
            file_path = os.path.join(pdf_dir, filename)
            data = process_pdf(file_path)
            if data:
                if not data['extracted_text']:
                    print(f'Warning: no text extracted from {filename}')
                collection.insert_one(data)
                print(f"Stored PDF : {filename}")

if __name__ == '__main__':
    populate_pdf_database()

    