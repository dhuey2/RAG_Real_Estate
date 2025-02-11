from llama_parse import LlamaParse
import nest_asyncio
from config import LLAMA_API_KEY

nest_asyncio.apply()

# Initialize the LlamaParse parser
parser = LlamaParse(
    api_key=LLAMA_API_KEY,        # Your API key
    result_type="text",     # Choose "text" or "markdown" based on your needs
    verbose=True            # Set to True for detailed logs
)

# Function to parse a document
def parse_document(file_path):
    # Load and parse the document
    documents = parser.load_data(file_path)
    return documents

# Example usage
file_path = "D:/RAG_System/data/purchase_agreement_sample_data.pdf"
parsed_documents = parse_document(file_path)
# Display the parsed content
for doc in parsed_documents:
    print(doc.text)