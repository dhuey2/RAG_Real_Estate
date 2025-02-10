import os
from document_processing import extract_text_with_pdfplumber, clean_extracted_text, chunk_text, save_chunks_to_file, anonymize_structured_data, anonymize_text
from vector_store import load_faiss_index, create_faiss_index, save_faiss_index, search_faiss_index, encode_query, get_chunks_by_indices
from llm_query import generate_answer_with_openai
from embeddings import save_embeddings_to_file
import json
import faiss
import numpy as np

#INPUT
QUERY = "Can you explain the main reason for this document?"

# Paths
PDF_PATH = "D:/RAG_System/data/purchase_agreement_sample_data.pdf"
CHUNKS_PATH = "D:/RAG_System/data/chunks.json"
EMBEDDINGS_PATH = "D:/RAG_System/data/embeddings.json"
INDEX_PATH = "D:/RAG_System/data/faiss_index.index"


# Step 1: Process PDF
print("Extracting text from PDF...")
parsed_text = extract_text_with_pdfplumber(PDF_PATH)
cleaned_text = clean_extracted_text(parsed_text)
anonmyized_text = anonymize_structured_data(cleaned_text)
#anonmyized_text = anonymize_text(anonmyized_text)
chunks = chunk_text(anonmyized_text)

# Save chunks for debugging
save_chunks_to_file(chunks, CHUNKS_PATH)

# Step 2: Generate Embeddings
print("Generating embeddings for chunks...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks).tolist()

# Save embeddings
save_embeddings_to_file(embeddings, "D:/RAG_System/data/embeddings.json")

# Step 3: Load/Create FAISS Index
print("Creating new FAISS index from embeddings...")

# Ensure embeddings exist
if not os.path.exists(EMBEDDINGS_PATH):
    raise FileNotFoundError(f"Embeddings file not found at {EMBEDDINGS_PATH}. Cannot create FAISS index.")

# Load fresh embeddings
with open(EMBEDDINGS_PATH, "r") as f:
    embeddings = np.array(json.load(f)["embeddings"], dtype='float32')

# Create a new FAISS index and overwrite existing one
index = create_faiss_index(embeddings)
save_faiss_index(index, INDEX_PATH)
print("FAISS index has been successfully overwritten.")

# Step 4: Query the System
print("Starting step 4")
while True:
    user_query = input("Enter your query (or type 'exit' to quit): ")
    if user_query.lower() == "exit":
        break

    query_embedding = encode_query(user_query)
    top_k = 5
    distances, indices = search_faiss_index(index, query_embedding, top_k)
    retrieved_chunks = [chunks[i] for i in indices]

    # Step 5: Generate Answer
    print("\nGenerating response from LLM...")
    answer = generate_answer_with_openai(user_query, retrieved_chunks)

    # Display Results
    print("\nUser Query:", user_query)
    print("Top Retrieved Chunks:")
    for i, (chunk, distance) in enumerate(zip(retrieved_chunks, distances)):
        print(f"{i+1}. {chunk} (Distance: {distance})")

    print("\nGenerated Answer:")
    print(answer)