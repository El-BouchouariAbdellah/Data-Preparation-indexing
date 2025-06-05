import os
import json
import gc
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List # used for type hinting 
from langchain_community.embeddings import HuggingFaceEmbeddings

#  Configuration for Retrieval 
# Path to the FAISS index directory 
FAISS_INDEX_DIR = "curriculum_faiss_index_langchain" 
EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-large'

#  Retrieval Parameters
INITIAL_RETRIEVAL_K = 10 # Retrieve top 10 chunks based on semantic similarity
MAX_CHUNKS_TO_SEND_TO_LLM = 5 # Aim for 3-5 high-quality chunks

#  Function to perform Retrieval 
def perform_retrieval(query_text: str, student_grade: int, faiss_index_dir: str, model_name: str, initial_k: int, max_chunks_for_llm: int) -> List[Document]:

    print(f"\n Starting Retrieval for Query: '{query_text}' (Student Grade: {student_grade}) ---")

    # Load the embedding model 
    print(f"Loading embedding model: '{model_name}' and wrapping for LangChain...")
    try:
        embedding_function = HuggingFaceEmbeddings(model_name=model_name)
        print("✅ Embedding model loaded and wrapped successfully.")
    except Exception as e:
        print(f"❌ Error loading embedding model '{model_name}' or wrapping: {e}")
        print("Please check your internet connection or model name.")
        return []

    # Load the FAISS index from the faiss_index directory
    print(f"Loading FAISS index from: {faiss_index_dir}...")
    try:
        vectorstore = FAISS.load_local(
            faiss_index_dir,
            embedding_function, 
            allow_dangerous_deserialization=True
        )
        print("✅ FAISS index loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading FAISS index from '{faiss_index_dir}': {e}")
        print("Please ensure the directory exists and contains the FAISS index files.")
        return []

    #  Perform Similarity Search 
    # E5 models require a 'query: ' prefix for queries for optimal performance.
    prefixed_query_text = "query: " + query_text
    
    print(f"Performing similarity search for '{query_text}' (retrieving top {initial_k} raw results)...")
    retrieved_docs_with_scores = vectorstore.similarity_search_with_score(
        prefixed_query_text,
        k=initial_k
    ) 
    #inside the function the query is transformed to an embedding vector, and then searsh for the similar documents in faiss index , and calculate the similarity score , it depends on the indexing method used in the FAISS index is it IndexFlatL2 or IndexIVFFlat or IndexHNSWFlat, etc.
    #then it returns top k (10) documents with their similarity scores
    #the final output is a list of tuples, each tuple contains a Document object (tect-content and metadata) and its similarity score
    print(f"  Retrieved {len(retrieved_docs_with_scores)} raw documents from FAISS.")

    #  Apply Grade-Level Filtering
    filtered_chunks = []
    print("Applying grade-level filtering...")
    for doc, score in retrieved_docs_with_scores:
        chunk_grade = doc.metadata.get('grade')
        chunk_subject = doc.metadata.get('subject')
        source_file = doc.metadata.get('source_file')

        if chunk_grade is not None and isinstance(chunk_grade, (int, float)) and chunk_grade <= student_grade:
            filtered_chunks.append((doc, score))
            print(f"    ✅ Kept: '{doc.page_content[:50]}...' (Grade: {chunk_grade}, Subject: {chunk_subject}, Score: {score:.4f}, Source: {source_file})")
        else:
            print(f"    ❌ Filtered out: '{doc.page_content[:50]}...' (Grade: {chunk_grade}, Subject: {chunk_subject}, Score: {score:.4f}, Source: {source_file}) - too high or invalid for Grade {student_grade}")

    filtered_chunks.sort(key=lambda x: x[1]) # Sort by similarity score (ascending) x[1] is the score in the tuple (doc, score)

    final_chunks_for_llm = [doc for doc, score in filtered_chunks[:max_chunks_for_llm]]

    print(f"--- Retrieval Complete: {len(final_chunks_for_llm)} grade-appropriate chunks selected for LLM. ---")
    return final_chunks_for_llm

#  Main execution block 
if __name__ == "__main__":
    print("Starting Step 6: Retrieval Logic Development...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    faiss_index_full_path = os.path.join(script_dir, FAISS_INDEX_DIR)

    test_query_french = "Le verbe être au futur ?."
    test_grade_french = 6
    
    print("\n\n" + "="*50 + "\n")

    retrieved_chunks_french = perform_retrieval(
        test_query_french,
        test_grade_french,
        faiss_index_full_path,
        EMBEDDING_MODEL_NAME,
        INITIAL_RETRIEVAL_K,
        MAX_CHUNKS_TO_SEND_TO_LLM
    )

    print("\nRetrieval testing complete. The retrieved Document objects are ready for LLM integration.")