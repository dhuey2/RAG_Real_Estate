import json
import os
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pymupdf4llm
import unicodedata


def extract_text_with_pymupdf4llm(file_path):
    """
    Extracts text from a PDF using pymupdf4llm.
    """
    return pymupdf4llm.to_markdown(file_path)

def clean_extracted_text(text):
    """
    Clean text by removing extra newlines and fixing hyphenated words.
    """
    # Remove newlines where the next line continues the same sentence
    text = re.sub(r'-\n', '', text)  # Fix hyphenated line breaks
    text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with a space
    return text.strip()

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large-finetuned-conll03-english")
model = AutoModelForTokenClassification.from_pretrained("FacebookAI/xlm-roberta-large-finetuned-conll03-english")

# Create a pipeline for Named Entity Recognition
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

def anonymize_text(text):
    ner_results = ner_pipeline(text)

    ner_results = sorted(ner_results, key= lambda x: x['start'])

    #variables
    new_text = ""
    last_idx = 0

    for entity in ner_results:
        new_text += text[last_idx:entity['start']]
        new_text += entity['entity']
        last_idx = entity['end']
        #print(entity)
    
    new_text += text[last_idx:] #adding the rest of the text

    return new_text

def clean_text_for_ner(text):
    """
    Cleans extracted text to improve NER recognition.
    - Replaces curly apostrophes with straight ones
    - Adds space after periods if missing
    - Removes excessive underscores
    """
    # Normalize Unicode characters (e.g., fancy quotes → standard)
    text = unicodedata.normalize("NFKC", text)

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove excessive underscores
    text = re.sub(r'_+', ' ', text)

    text = re.sub(r'\.(?=[A-Za-z])', '. ', text)

    return text

    

def anonymize_structured_data(text):
    text = re.sub(r"\b\d{3}-\d{3}-\d{4}\b", "<PHONE>", text)  # Phone numbers
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "<EMAIL>", text)  # Emails
    return text


def chunk_text(text, max_chunk_size = 500):
    """
    Splits text into chunks of specified max size.
    """

    paragraphs = re.split(r'(?<=[.!?]) +', text)  # Split on sentence boundaries
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if len(current_chunk) + len(paragraph) <= max_chunk_size: #can add current text to chunk
            current_chunk += paragraph + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph + " "

    if current_chunk:  # Add the last chunk
        chunks.append(current_chunk.strip())

    return chunks


def save_chunks_to_file(chunks, file_path):
    """
    Saves parsed and chunked text to a JSON file.
    :param chunks: List of text chunks.
    :param file_path: Path to the JSON file to save.
    """
    # Ensure the directory exists
    dir_name = os.path.dirname(file_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)  # Create the directory if it doesn't exist

    # Write chunks to the JSON file
    with open(file_path, "w") as f:
        json.dump({"chunks": chunks}, f, indent=4)
    print(f"Chunks saved to {file_path}")




#test 1
if __name__ == "__main__":
    file_path = "D:/RAG_System/data/work_authorization.pdf"
    parsed_text = extract_text_with_pymupdf4llm(file_path)
    spaced_text = clean_text_for_ner(parsed_text)
    cleaned_text = clean_extracted_text(spaced_text)


    chunks = chunk_text(cleaned_text)
    anonymized_chunks = []
    for chunk in chunks:
        anonmyized_text = anonymize_text(chunk)
        anonmyized_text = anonymize_structured_data(anonmyized_text)
        anonymized_chunks.append(anonmyized_text)

    print(f"Generated {len(anonymized_chunks)} chunks.")
    print(anonymized_chunks[:10]) 
    print("Number of chunks generated: " , len(anonymized_chunks))


    #save chunks to json file for debugging
    save_chunks_to_file(anonymized_chunks, "D:\RAG_System\data\chunks.json")

