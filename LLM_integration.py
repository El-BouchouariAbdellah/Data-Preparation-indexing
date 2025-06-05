import os
import sys
import gc
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List

# --- Google Gemini/AI Studio Imports ---
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
# -------------------------------------

# --- Configuration (match your setup from Step 6) ---
FAISS_INDEX_DIR = "curriculum_faiss_index_langchain"
EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-large'
INITIAL_RETRIEVAL_K = 10
MAX_CHUNKS_TO_SEND_TO_LLM = 5

# --- Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
faiss_index_full_path = os.path.join(script_dir, FAISS_INDEX_DIR)

# --- GLOBAL MODELS (Load once to avoid reloading for every query) ---
embedding_model_for_query = None
faiss_vectorstore_loaded = None
gemini_model = None

# --- Helper Function to Load Components ---
def load_rag_components(faiss_path: str, embed_model_name: str):
    global embedding_model_for_query, faiss_vectorstore_loaded, gemini_model

    if embedding_model_for_query is None:
        print(f"Loading embedding model: '{embed_model_name}' for query embedding...")
        try:
            embedding_model_for_query = HuggingFaceEmbeddings(model_name=embed_model_name)
            print("✅ Embedding model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading embedding model '{embed_model_name}': {e}")
            sys.exit("Critical: Cannot load embedding model. Exiting.")

    if faiss_vectorstore_loaded is None:
        print(f"Loading FAISS index from: {faiss_path}...")
        try:
            faiss_vectorstore_loaded = FAISS.load_local(
                faiss_path,
                embedding_model_for_query,
                allow_dangerous_deserialization=True
            )
            print("✅ FAISS index loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading FAISS index from '{faiss_path}': {e}")
            sys.exit("Critical: Cannot load FAISS index. Exiting.")

    if gemini_model is None:
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        if not google_api_key:
            sys.exit("Critical: GOOGLE_API_KEY environment variable not set. Exiting.")
        
        try:
            genai.configure(api_key=google_api_key)
            gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            print("✅ Google Gemini model configured successfully.")
        except Exception as e:
            print(f"❌ Error configuring Google Gemini model: {e}")
            sys.exit("Critical: Cannot configure Gemini model. Exiting.")

# --- Retrieval Function (from Step 6) ---
def get_grade_appropriate_context(query_text: str, student_grade: int, initial_k: int, max_chunks_for_llm: int) -> List[Document]:
    if embedding_model_for_query is None or faiss_vectorstore_loaded is None:
        print("Error: RAG components not loaded. Call load_rag_components() first.")
        return []

    print(f"\n--- Retrieving context for: '{query_text}' (Grade: {student_grade}) ---")

    prefixed_query_text = "query: " + query_text
    
    retrieved_docs_with_scores = faiss_vectorstore_loaded.similarity_search_with_score(
        prefixed_query_text,
        k=initial_k
    )
    print(f"  Retrieved {len(retrieved_docs_with_scores)} raw documents from FAISS.")

    filtered_chunks = []
    for doc, score in retrieved_docs_with_scores:
        chunk_grade = doc.metadata.get('grade')
        if chunk_grade is not None and isinstance(chunk_grade, (int, float)) and chunk_grade <= student_grade:
            filtered_chunks.append((doc, score))

    filtered_chunks.sort(key=lambda x: x[1])

    final_chunks_for_llm = [doc for doc, score in filtered_chunks[:max_chunks_for_llm]]
    print(f"  Selected {len(final_chunks_for_llm)} grade-appropriate chunks.")

    return final_chunks_for_llm

# --- New: LLM Interaction Function (Prompts in French) ---
def get_llm_response(student_query: str, retrieved_context_docs: List[Document], student_grade: int) -> str:
    if gemini_model is None:
        print("Error: Le modèle Gemini n'est pas chargé. Impossible d'obtenir une réponse.") # French error message
        return "Désolé, je rencontre des difficultés pour comprendre votre question pour le moment. Veuillez réessayer." # French fallback

    context_str = "\n\n".join([doc.page_content for doc in retrieved_context_docs])
    
    if not context_str.strip():
        print("  ⚠️ Aucun contexte fourni au modèle LLM. Réponse avec un message de secours général.") # French print message
        prompt = f"""
        Vous êtes un assistant pédagogique utile et compétent pour les élèves.
        Un élève de la {student_grade}e année a posé la question suivante : "{student_query}"
        
        Aucune information pertinente du programme scolaire n'a été trouvée pour cette question spécifique à ce niveau scolaire.
        Veuillez fournir une réponse générale si possible, mais indiquez que vous n'avez pas trouvé de détails spécifiques dans le programme fourni.
        Utilisez un langage extrêmement simple et adapté à un élève de {student_grade}e année.
        """
    else:
        prompt = f"""
        Vous êtes un assistant pédagogique utile et compétent pour les élèves.
        Votre tâche est de répondre à la question d'un élève en vous basant UNIQUEMENT sur le contexte du programme scolaire fourni ci-dessous.
        
        **Instructions :**
        1.  Répondez à la question avec précision en utilisant UNIQUEMENT les informations du contexte fourni.
        2.  Ne DOIT PAS utiliser de connaissances externes. Si la réponse ne figure pas dans le contexte, indiquez "Je ne trouve pas d'informations spécifiques à ce sujet dans le programme scolaire fourni."
        3.  Adaptez votre réponse en utilisant des concepts, du vocabulaire et un style d'explication adaptés à un élève de **{student_grade}e année**.
        4.  **CRITIQUE :** Ne DOIT PAS introduire de concepts ou de terminologie qui seraient généralement enseignés dans les années supérieures à la {student_grade}e année. Respectez strictement le niveau du programme scolaire.
        5.  Expliquez clairement et de manière concise. Si la question est en arabe, répondez en arabe. Si en français, répondez en français. Si en anglais, répondez en anglais.
        
        **Question de l'élève ({student_grade}e année) :** "{student_query}"
        
        **Contexte du programme scolaire :**
        {context_str}
        
        **Veuillez fournir votre réponse maintenant :**
        """

    print("\n--- Envoi au modèle LLM ---") # French print message
    print(f"  Prompt (premiers 500 caractères) : {prompt[:500]}...") # French print message
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"❌ Erreur lors de la génération de la réponse par le modèle LLM : {e}") # French error message
        return "Désolé, je rencontre des difficultés pour générer une réponse pour le moment. Veuillez réessayer." # French fallback

# --- Main Chatbot Loop ---
if __name__ == "__main__":
    print("🚀 Démarrage du Chatbot Adapté au Niveau Scolaire 🚀") # French print message
    print("=" * 50)

    load_rag_components(faiss_index_full_path, EMBEDDING_MODEL_NAME)
    gc.collect()

    student_grade_input = input("\nEntrez la classe actuelle de l'élève (par ex. 6) : ") # French prompt
    try:
        student_grade = int(student_grade_input)
        if student_grade < 1 or student_grade > 12:
            raise ValueError
    except ValueError:
        print("Niveau invalide. Par défaut : 6e année.") # French message
        student_grade = 6

    print(f"\nSession de chatbot démarrée pour un élève de {student_grade}e année.") # French message
    print("Tapez votre question, ou 'exit' pour quitter.") # French message
    print("-" * 50)

    while True:
        user_query = input(f"Élève ({student_grade}e année) > ") # French prompt

        if user_query.lower() == 'exit':
            print("Chatbot : Au revoir !") # French message
            break

        if not user_query.strip():
            print("Chatbot : Veuillez entrer une question.") # French message
            continue

        retrieved_context_docs = get_grade_appropriate_context(
            user_query,
            student_grade,
            INITIAL_RETRIEVAL_K,
            MAX_CHUNKS_TO_SEND_TO_LLM
        )

        if retrieved_context_docs:
            print("\n--- Obtention de la réponse du modèle LLM ---") # French message
            llm_response = get_llm_response(user_query, retrieved_context_docs, student_grade)
            print("\nChatbot :", llm_response)
        else:
            print("\n--- Obtention de la réponse du modèle LLM (Aucun Contexte) ---") # French message
            llm_response_no_context = get_llm_response(user_query, [], student_grade)
            print("\nChatbot :", llm_response_no_context)
        
        print("\n" + "-" * 50)
        gc.collect()