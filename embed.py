import os
import PyPDF2
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
import pickle

# Directory containing PDFs
pdf_directory = "./pdfs"
output_directory = "./vectordb"
os.makedirs(output_directory, exist_ok=True)

# Chunk size and model for embeddings
chunk_size = 500  # Number of characters per chunk
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Load tokenizer and model for embeddings
tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
model = AutoModel.from_pretrained(embedding_model_name)

def pdf_to_chunks(pdf_path, chunk_size):
    """Extract chunks of text from a PDF."""
    chunks = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            for i in range(0, len(text), chunk_size):
                chunks.append(text[i:i + chunk_size])
    return chunks

def encode_texts(texts):
    """Convert a list of texts into embeddings."""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings.cpu().numpy()

# Initialize FAISS index
embedding_dim = model.config.hidden_size
index = faiss.IndexFlatL2(embedding_dim)
metadata = []  # To store metadata for embeddings

# Process each PDF
for pdf_file in os.listdir(pdf_directory):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, pdf_file)
        print(f"Processing {pdf_file}...")
        chunks = pdf_to_chunks(pdf_path, chunk_size)
        embeddings = encode_texts(chunks)

        # Add embeddings to the FAISS index
        index.add(embeddings)

        # Store metadata for each chunk
        metadata.extend([(pdf_file, i) for i in range(len(chunks))])

# Save the FAISS index and metadata
faiss.write_index(index, os.path.join(output_directory, "vector_index.faiss"))
with open(os.path.join(output_directory, "metadata.pkl"), "wb") as f:
    pickle.dump(metadata, f)

print("Vector database saved in 'vectordb' directory.")
