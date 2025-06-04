import os
import json
import gc
from langchain_text_splitters import RecursiveCharacterTextSplitter


TAGGED_DATA_INPUT_FILE = "tagged_curriculum_data.json" 

CHUNKED_DATA_OUTPUT_FILE = "chunked_curriculum_data.json"

# Chunking Parameters 
CHUNK_SIZE = 400       # Maximum number of characters per chunk.
CHUNK_OVERLAP = 50     # Number of characters to overlap between consecutive chunks.


# Separators to try when splitting text. The splitter tries these in order.

TEXT_SPLIT_SEPARATORS = ["\n\n", "\n", " ", ""]

#  Function to perform Chunking 
def perform_chunking(input_json_path, chunk_s, chunk_o, separators):

    all_chunks = []

    if not os.path.exists(input_json_path):
        print(f"❌ Error: Input JSON file not found: {input_json_path}")
        return []

    print(f"\n--- Starting Text Chunking from: {input_json_path} ---")

    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            tagged_documents = json.load(f)
        print(f"Loaded {len(tagged_documents)} tagged documents/pages.")
    except Exception  as e:
        print(f"❌ Error loading tagged documents from JSON: {e}")
        return []

    if not tagged_documents:
        print("⚠️ No tagged documents found in the input JSON. Nothing to chunk.")
        return []

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_s,
        chunk_overlap=chunk_o,
        length_function=len,
        separators=separators
    )

    processed_doc_count = 0
    for doc_data in tagged_documents:
        full_text_content = doc_data["text_content"]
        original_metadata = doc_data["metadata"]

        if not full_text_content.strip():
            print(f"  Skipping empty content from {original_metadata.get('source_file', 'unknown_file')}.")
            continue

        try:
            # Split the document's text content into chunks
            chunks_from_doc = text_splitter.split_text(full_text_content)
            
            print(f"  Processing '{original_metadata.get('source_file', 'unknown_file')}' (Grade: {original_metadata['grade']}, Subject: {original_metadata['subject']}): {len(chunks_from_doc)} chunks created.")

            # Attach the original document's metadata to EACH new chunk
            for i, chunk_text in enumerate(chunks_from_doc):
                chunk_metadata = original_metadata.copy() # Make a copy to avoid modifying original
                chunk_metadata["chunk_id"] = i + 1 # Add a unique ID for the chunk within its document
                
                all_chunks.append({
                    "text_content": chunk_text,
                    "metadata": chunk_metadata
                })
            processed_doc_count += 1

        except Exception as e:
            print(f"  ❌ Error chunking document from {original_metadata.get('source_file', 'unknown_file')}: {e}")
        
        gc.collect() # Clean up memory after processing each document

    print(f"\n--- Chunking Complete: {len(all_chunks)} total chunks created from {processed_doc_count} documents. ---")
    return all_chunks

# --- Main execution block ---
if __name__ == "__main__":
    print("Starting  Text Chunking...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file_path = os.path.join(script_dir, TAGGED_DATA_INPUT_FILE)
    output_file_path = os.path.join(script_dir, CHUNKED_DATA_OUTPUT_FILE)

    chunked_data = perform_chunking(
        input_file_path,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        TEXT_SPLIT_SEPARATORS
    )

    # Save the result to a JSON file. This will be the input for the next step (Embeddings).
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(chunked_data, f, ensure_ascii=False, indent=2)  # Convert Python data to JSON and write to file ensure_ascii false to keep arabic charachters readable
        print(f"\nSuccessfully saved all chunked data to: {output_file_path}")
    except Exception as e:
        print(f"❌ Error saving chunked data to JSON: {e}")

    # Optional: Print a preview of the first few chunks
    if chunked_data:
        print("\n--- Preview of First 3 Chunks ---")
        for i, chunk in enumerate(chunked_data[:3]):
            print(f"Chunk {i+1}:")
            print(f"  Metadata: {chunk['metadata']}")
            print(f"  Text (first 200 chars): {chunk['text_content'][:200]}...")
            print("-" * 30)
    else:
        print("\nNo chunks were created. Please check your input file and data.")