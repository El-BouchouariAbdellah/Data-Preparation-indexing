import os
import json
import gc
from sentence_transformers import SentenceTransformer
import time 

# Configuration for embeddings generation
CHUNKED_DATA_INPUT_FILE = "chunked_curriculum_data.json" 

# Output file for the data with embeddings added
EMBEDDED_DATA_OUTPUT_FILE = "embedded_curriculum_data.json"

#Embedding Model Configuration

EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-large'

# Function to generate Embeddings 
def generate_embeddings(input_json_path, output_json_path, model_name):

    if not os.path.exists(input_json_path):
        print(f"❌ Error: Input JSON file not found: {input_json_path}")
        return []

    print(f"\n--- Starting Embeddings Generation from: {input_json_path} ---")

    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            chunked_data = json.load(f)
        print(f"Loaded {len(chunked_data)} chunks from input JSON.")
    except Exception as e:
        print(f"❌ Error loading chunked data from JSON: {e}")
        return []

    if not chunked_data:
        print("⚠️ No chunks found in the input JSON. Nothing to embed.")
        return []

    # Initialize the embedding model once
    print(f"Loading embedding model: '{model_name}'...")
    try:
        model = SentenceTransformer(model_name)
        print("✅ Embedding model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading embedding model '{model_name}': {e}")
        print("Please check your internet connection or model name.")
        return []

    # Prepare texts for embedding
    texts_to_embed = ["passage: " + chunk['text_content'] for chunk in chunked_data] # the model work with task-specific-prefexies , 'passage' for Document content and 'query' for searsh questions 
    
    print(f"Generating embeddings for {len(texts_to_embed)} chunks...") #215
    start_time = time.time()
    
    # Generate embeddings. show_progress_bar=True is helpful for large datasets.
    # convert_to_numpy=True ensures we get a NumPy array, then we convert to list for JSON.
    embeddings = model.encode(texts_to_embed, show_progress_bar=True, convert_to_numpy=True) # time is optional here 
    
    end_time = time.time()
    print(f"✅ Embeddings generated in {end_time - start_time:.2f} seconds.")
    print(f"Shape of generated embeddings: {embeddings.shape}")  # Should be (num_chunks, embedding_dimension) e.g., (215, 1024) for E5-large

    # Add embeddings back to the chunked data structure
    embedded_data = []
    for i, chunk_obj in enumerate(chunked_data):
        chunk_obj["embedding"] = embeddings[i].tolist() # Convert numpy array to list for JSON storage
        embedded_data.append(chunk_obj)
    
    # Save the updated data to a new JSON file
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(embedded_data, f, ensure_ascii=False, indent=2)
        print(f"\nSuccessfully saved all embedded data to: {output_json_path}")
    except Exception as e:
        print(f"❌ Error saving embedded data to JSON: {e}")
        return []

    gc.collect() # Clean up memory

    return embedded_data

#  Main execution block ---
if __name__ == "__main__":
    print("Starting Embeddings Generation...")

    # Ensure this script is run from the same directory where 'chunked_curriculum_data.json' is located.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file_path = os.path.join(script_dir, CHUNKED_DATA_INPUT_FILE)
    output_file_path = os.path.join(script_dir, EMBEDDED_DATA_OUTPUT_FILE)

    # Call the function to generate embeddings
    embedded_curriculum_data = generate_embeddings(
        input_file_path,
        output_file_path,
        EMBEDDING_MODEL_NAME
    )

    # Optional: Print a preview of the first few embedded chunks
    if embedded_curriculum_data:
        print("\n--- Preview of First 2 Embedded Chunks ---")
        for i, chunk in enumerate(embedded_curriculum_data[:2]):
            print(f"Chunk {i+1}:")
            print(f"  Metadata: {chunk['metadata']}")
            print(f"  Text (first 100 chars): {chunk['text_content'][:100]}...")
            print(f"  Embedding (first 5 values): {chunk['embedding'][:5]}...") # Show just a few values
            print(f"  Embedding Dimension: {len(chunk['embedding'])}") # Should be 1024 for E5-large
            print("-" * 30)
    else:
        print("\nNo embeddings were generated. Please check your input file and model configuration.")