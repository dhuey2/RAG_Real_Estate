import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os

model = SentenceTransformer('all-MiniLM-L6-v2')

# Paths
index_path = "D:/RAG_System/data/faiss_index.index"
chunks_file = "D:/RAG_System/data/chunks.json"
embeddings_file = "D:/RAG_System/data/embeddings.json"

def create_faiss_index(embeddings):
    """
    Create a FAISS index for the given embeddings.
    """
    dimension = len(embeddings[0])  # Length of each embedding vector
    index = faiss.IndexFlatL2(dimension)  # L2 distance 
    index.add(np.array(embeddings, dtype='float32'))  # Add embeddings to the index
    return index

def save_faiss_index(index, file_path):
    """
    Save the FAISS index to a file.
    """
    faiss.write_index(index, file_path)
    print(f"FAISS index saved to {file_path}")

def load_faiss_index(file_path):
    """
    Load an existing FAISS index from a file.
    """
    if os.path.exists(file_path):
        return faiss.read_index(file_path)
    else:
        print("FAISS index not found. Creating a new one.")
        return None  # Return None if the index doesn't exist

def encode_query(query):
    """
    Encodes a user query into an embedding.
    """
    return model.encode([query])[0]

def search_faiss_index(index, query_embedding, top_k=5):
    """
    Searches FAISS index for the top_k most similar embeddings.
    """
    query_embedding = np.array([query_embedding], dtype="float32")
    distances, indices = index.search(query_embedding, top_k)
    return distances[0], indices[0]

def get_chunks_by_indices(chunks_file, indices):
    """
    Retrieves text chunks corresponding to the given indices.
    """
    with open(chunks_file, "r") as f:
        chunks = json.load(f)["chunks"]
    return [chunks[i] for i in indices]

# Main Execution
if __name__ == "__main__":
    # Load FAISS index (or create if missing)
    index = load_faiss_index(index_path)
    
    if index is None:  # If index doesn't exist, create it
        print("Creating new FAISS index from embeddings.")
        with open(embeddings_file, "r") as f:
            embeddings = np.array(json.load(f)["embeddings"], dtype='float32')
        
        index = create_faiss_index(embeddings)
        save_faiss_index(index, index_path)

    # Load chunks
    with open(chunks_file, "r") as f:
        chunks = json.load(f)["chunks"]

    # User query
    user_query = "What is the name of the Broker? What is the point of this document? What does point 10.4 mean?"
    query_embedding = encode_query(user_query)

    # Search FAISS index
    top_k = 5
    distances, indices = search_faiss_index(index, query_embedding, top_k=top_k)

    # Retrieve chunks
    results = get_chunks_by_indices(chunks_file, indices)

    print("User Query:", user_query)
    print("Top Results:")
    for i, (chunk, distance) in enumerate(zip(results, distances)):
        print(f"{i+1}. {chunk} (Distance: {distance})")
