import os
import PyPDF2
from transformers import AutoTokenizer, AutoModel
import torch
import chromadb
from chromadb.config import Settings

# Directory containing PDFs
pdf_directory = "pdfs"
output_directory = "vectordb"
os.makedirs(output_directory, exist_ok=True)

# Chunk size and model for embeddings
chunk_size = 500  # Number of characters per chunk
overlap_size = 100  # Number of characters to overlap between chunks
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Load tokenizer and model for embeddings
tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
model = AutoModel.from_pretrained(embedding_model_name)

def pdf_to_chunks_with_pages(pdf_path, chunk_size, overlap_size):
    """Extract overlapping chunks of text and corresponding page numbers from a PDF."""
    chunks = []
    page_numbers = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            for i in range(0, len(text), chunk_size - overlap_size):
                chunks.append(text[i:i + chunk_size])
                page_numbers.append(page_num + 1)  # Page numbers start from 1
    return chunks, page_numbers

def encode_texts(texts):
    """Convert a list of texts into embeddings."""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings.cpu().numpy().tolist()

# Initialize ChromaDB client with persistent storage
client = chromadb.PersistentClient(path=output_directory)

# Create a single collection for all PDFs
collection_name = "all_pdfs"
collection = client.get_or_create_collection(name=collection_name)

# Process each PDF
for pdf_file in os.listdir(pdf_directory):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, pdf_file)
        print(f"Processing {pdf_file}...")
        chunks, page_numbers = pdf_to_chunks_with_pages(pdf_path, chunk_size, overlap_size)
        embeddings = encode_texts(chunks)

        # Add documents to the collection
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=[
                {"source": pdf_file, "chunk_id": i, "page_number": page_numbers[i]}
                for i in range(len(chunks))
            ],
            ids=[f"{pdf_file}_{i}" for i in range(len(chunks))]
        )

print("Vector database saved in 'vectordb' directory.")
