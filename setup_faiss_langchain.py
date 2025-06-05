# This script reads chunked text data (from 'chunked_curriculum_data.json'),
# generates embeddings for each chunk using 'intfloat/multilingual-e5-large',
# and then builds and saves a FAISS vector index using LangChain.
# The FAISS index will store both the embeddings and the original text/metadata,
# making it ready for efficient retrieval in the RAG system.

import os
import json
import gc
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings # Import the wrapper
# ------------------

# Configuration for FAISS Indexing
CHUNKED_DATA_INPUT_FILE = "chunked_curriculum_data.json"

FAISS_INDEX_OUTPUT_DIR = "curriculum_faiss_index_langchain" # This will be a folder

# Embedding Model Configuration
EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-large'

# Function to setup and populate FAISS Index using LangChain
def setup_faiss_index_with_langchain_simplified(input_json_path, faiss_output_dir, model_name):
    if not os.path.exists(input_json_path):
        print(f"❌ Error: Input JSON file not found: {input_json_path}")
        return None

    print(f"\n--- Starting FAISS Index Setup (LangChain - Simplified) from: {input_json_path} ---")

    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            chunked_data = json.load(f)
        print(f"Loaded {len(chunked_data)} chunks from input JSON.")
    except Exception as e:
        print(f"❌ Error loading chunked data from JSON: {e}")
        return None

    if not chunked_data:
        print("⚠️ No chunks found in the input JSON. Nothing to index.")
        return None

    # Initialize the embedding model (wrapped for LangChain compatibility)
    print(f"Loading embedding model: '{model_name}' and wrapping for LangChain...")
    try:
        # --- MODIFICATION HERE ---
        # Instead of `SentenceTransformer(model_name)`, use HuggingFaceEmbeddings
        # This wrapper makes the SentenceTransformer model compatible with LangChain's Embeddings interface.
        embedding_function = HuggingFaceEmbeddings(model_name=model_name)
        # -------------------------
        print("✅ Embedding model loaded and wrapped successfully.")
    except Exception as e:
        print(f"❌ Error loading embedding model '{model_name}' or wrapping: {e}")
        print("Please check your internet connection or model name.")
        return None

    # Prepare data for LangChain's FAISS.from_documents
    langchain_documents = []
    for chunk_obj in chunked_data:
        # Add the 'passage: ' prefix here, as it's crucial for E5 models' performance
        prefixed_text_content = "passage: " + chunk_obj["text_content"]
        
        doc = Document(
            page_content=prefixed_text_content,
            metadata=chunk_obj["metadata"]
        )
        langchain_documents.append(doc)
    
    # Create and Populate FAISS Index using LangChain
    print("Creating FAISS index using LangChain (this will generate embeddings internally)...")
    try:
        vectorstore = FAISS.from_documents(
            documents=langchain_documents,
            embedding=embedding_function # Pass the wrapped embedding function here
        )
        print("✅ FAISS index created and populated.")

        # Save the FAISS index to a local directory
        vectorstore.save_local(faiss_output_dir)
        print(f"✅ FAISS index saved to directory: {faiss_output_dir}")

    except Exception as e:
        print(f"❌ Error creating or saving FAISS index with LangChain: {e}")
        return None

    gc.collect()

    return vectorstore

# --- Main execution block ---
if __name__ == "__main__":
    print("Starting Step 5 (Simplified): FAISS Index Setup (LangChain Version)...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file_path = os.path.join(script_dir, CHUNKED_DATA_INPUT_FILE)
    faiss_output_full_path = os.path.join(script_dir, FAISS_INDEX_OUTPUT_DIR)

    faiss_vectorstore = setup_faiss_index_with_langchain_simplified(
        input_file_path,
        faiss_output_full_path,
        EMBEDDING_MODEL_NAME
    )

    if faiss_vectorstore:
        print("\nFAISS index (LangChain compatible) is ready for retrieval queries!")
    else:
        print("\nFailed to set up FAISS index.")