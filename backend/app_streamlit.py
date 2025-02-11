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

# ✅ Get the base directory (fixes path issues)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ✅ Ensure 'data/' directory exists
os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(page_title="Real Estate Document Assistant", layout="wide")

st.title("📄 Real Estate Document Assistant 🏡")
st.write("Upload a real estate document and ask questions about it.")

# File Upload
uploaded_file = st.file_uploader("Upload a PDF document:", type=["pdf"])

if uploaded_file is not None:
    # ✅ Dynamically store the uploaded file with its original name
    PDF_PATH = os.path.join(DATA_DIR, uploaded_file.name)

    # ✅ Save the uploaded file
    with open(PDF_PATH, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"File uploaded successfully: {uploaded_file.name}")

    # Step 1: Process PDF
    st.write("🔍 Extracting and processing text...")
    parsed_text = extract_text_with_pymupdf4llm(PDF_PATH)
    spaced_text = clean_text_for_ner(parsed_text)
    cleaned_text = clean_extracted_text(spaced_text)
    chunks = chunk_text(cleaned_text)
    
    # ✅ Anonymize Each Chunk Individually
    anonymized_chunks = [anonymize_structured_data(anonymize_text(chunk)) for chunk in chunks]

    # ✅ Save Chunks to Data Folder
    CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.json")
    save_chunks_to_file(anonymized_chunks, CHUNKS_PATH)

    # Step 2: Generate Embeddings
    st.write("📌 Generating embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(anonymized_chunks).tolist()

    # ✅ Save Embeddings
    EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.json")
    save_embeddings_to_file(embeddings, EMBEDDINGS_PATH)

    # Step 3: Create FAISS Index
    st.write("⚡ Creating FAISS index...")
    INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.index")
    index = create_faiss_index(embeddings)
    save_faiss_index(index, INDEX_PATH)

    st.success("✅ Document processing complete! Ready for queries.")

# Query Input
user_query = st.text_input("Ask a question about the document:")

if user_query:
    st.write("🔎 Searching for relevant information...")

    # ✅ Check if FAISS Index Exists Before Querying
    INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.index")
    if not os.path.exists(INDEX_PATH):
        st.error("❌ No FAISS index found! Please upload and process a PDF first.")
    else:
        # ✅ Load FAISS Index
        index = load_faiss_index(INDEX_PATH)

        # ✅ Encode Query
        query_embedding = encode_query(user_query)
        distances, indices = search_faiss_index(index, query_embedding, top_k=5)

        # ✅ Retrieve Chunks
        CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.json")
        with open(CHUNKS_PATH, "r") as f:
            chunks_data = json.load(f)
        
        retrieved_chunks = [chunks_data["chunks"][i] for i in indices if i < len(chunks_data["chunks"])]

        # ✅ Generate Answer
        st.write("💬 Generating AI response...")
        answer = generate_answer_with_openai(user_query, retrieved_chunks)

        # ✅ Display Response
        st.subheader("🤖 AI Response:")
        st.write(answer)
