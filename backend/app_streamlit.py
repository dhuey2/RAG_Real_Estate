import os
import streamlit as st
import json
import faiss
import numpy as np
from document_processing import (
    extract_text_with_pymupdf4llm, clean_extracted_text, chunk_text, save_chunks_to_file,
    anonymize_structured_data, clean_text_for_ner, anonymize_text
)
from vector_store import (
    load_faiss_index, create_faiss_index, save_faiss_index, search_faiss_index, encode_query
)
from llm_query import generate_answer_with_openai
from embeddings import save_embeddings_to_file
from sentence_transformers import SentenceTransformer

# Ensure 'data/' directory exists
os.makedirs("data", exist_ok=True)

st.set_page_config(page_title="Real Estate Document Assistant", layout="wide")

st.title("📄 Real Estate Document Assistant 🏡")
st.write("Upload a real estate document and ask questions about it.")

# File Upload
uploaded_file = st.file_uploader("Upload a PDF document:", type=["pdf"])

if uploaded_file is not None:
    # Dynamically store the uploaded file with its original name
    PDF_PATH = os.path.join("data", uploaded_file.name)

    # Save the uploaded file
    with open(PDF_PATH, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"File uploaded successfully: {uploaded_file.name}")

    # Step 1: Process PDF
    st.write("🔍 Extracting and processing text...")
    parsed_text = extract_text_with_pymupdf4llm(PDF_PATH)
    spaced_text = clean_text_for_ner(parsed_text)
    cleaned_text = clean_extracted_text(spaced_text)
    chunks = chunk_text(cleaned_text)
    anonymized_chunks = []
    for chunk in chunks:
        anonmyized_text = anonymize_text(chunk)
        anonmyized_text = anonymize_structured_data(anonmyized_text)
        anonymized_chunks.append(anonmyized_text)

    # Save chunks
    CHUNKS_PATH = "data/chunks.json"
    save_chunks_to_file(anonymized_chunks, CHUNKS_PATH)

    # Step 2: Generate Embeddings
    st.write("📌 Generating embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(anonymized_chunks).tolist()

    # Save embeddings
    EMBEDDINGS_PATH = "data/embeddings.json"
    save_embeddings_to_file(embeddings, EMBEDDINGS_PATH)

    # Step 3: Create FAISS Index
    st.write("⚡ Creating FAISS index...")
    INDEX_PATH = "data/faiss_index.index"
    index = create_faiss_index(embeddings)
    save_faiss_index(index, INDEX_PATH)

    st.success("✅ Document processing complete!")

# Query Input
user_query = st.text_input("Ask a question about the document:")

if user_query:
    st.write("🔎 Searching for relevant information...")

    # Load FAISS Index
    index = load_faiss_index("data/faiss_index.index")

    # Encode Query
    query_embedding = encode_query(user_query)
    distances, indices = search_faiss_index(index, query_embedding, top_k=5)

    # Retrieve Chunks
    with open("data/chunks.json", "r") as f:
        chunks = json.load(f)["chunks"]
    retrieved_chunks = [chunks[i] for i in indices]

    # Generate Answer
    st.write("💬 Generating AI response...")
    answer = generate_answer_with_openai(user_query, retrieved_chunks)

    # Display Response
    st.subheader("🤖 AI Response:")
    st.write(answer)
