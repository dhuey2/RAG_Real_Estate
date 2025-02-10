from sentence_transformers import SentenceTransformer
import json
import os

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(chunks):
    """
    Generates embeddings for a list of text chunks.
    :param chunks: List of text chunks.
    :return: List of embeddings.
    """
    embeddings = model.encode(chunks)
    return embeddings

def save_embeddings_to_file(embeddings, file_path):
    """
    Saves embeddings to a JSON file.
    :param embeddings: List of embeddings.
    :param file_path: Path to the JSON file to save.
    """
    # Ensure the directory exists
    dir_name = os.path.dirname(file_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)  # Create the directory if it doesn't exist

    # Save embeddings as JSON
    with open(file_path, "w") as f:
        json.dump({"embeddings": embeddings}, f, indent=4)
    print(f"Embeddings saved to {file_path}")

# Example Usage
if __name__ == "__main__":
    # Load chunks from the previously saved file
    chunks_file = "D:/RAG_System/data/chunks.json"
    with open(chunks_file, "r") as f:
        chunks = json.load(f)["chunks"]

    # Generate embeddings
    embeddings = generate_embeddings(chunks)
    print(f"Generated {len(embeddings)} embeddings.")

    # Save embeddings to a file
    save_embeddings_to_file(embeddings, "D:/RAG_System/data/embeddings.json")