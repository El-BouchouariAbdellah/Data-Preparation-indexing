import os
import sys
import gc
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langdetect import detect, DetectorFactory # For language detection
# Ensure reproducibility across runs for langdetect
DetectorFactory.seed = 0

#  Configuration 
FAISS_INDEX_DIR = "curriculum_faiss_index_langchain"
EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-large'
INITIAL_RETRIEVAL_K = 10
MAX_CHUNKS_TO_SEND_TO_LLM = 5

#  Paths 
script_dir = os.path.dirname(os.path.abspath(__file__))
faiss_index_full_path = os.path.join(script_dir, FAISS_INDEX_DIR)

#  GLOBAL MODELS 
embedding_model_for_query = None
faiss_vectorstore_loaded = None
gemini_model = None

#  Helper Function to Load Components 
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

#  Retrieval Function 
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

# Function to detect query language 
def detect_query_language(query: str) -> str:
    try:
        # Use langdetect to detect the language
        lang = detect(query)
        return lang
    except Exception:
        return 'en' # Default to English if detection fails or query is too short


#  LLM Interaction Function 
def get_llm_response(student_query: str, retrieved_context_docs: List[Document], student_grade: int) -> str:
    if gemini_model is None:
        return "Désolé, le modèle LLM n'est pas configuré. Impossible d'obtenir une réponse." 

    query_lang = detect_query_language(student_query)
    print(f"  Detected query language: {query_lang}")

    context_str = "\n\n".join([doc.page_content for doc in retrieved_context_docs])
    
    #  Dynamic Prompt Construction based on Detected Language 
    if query_lang == 'ar':
        # Arabic Prompt
        if not context_str.strip():
            prompt = f"""
            أنت مساعد تعليمي مفيد ومطلع للطلاب.
            سأل طالب في الصف {student_grade} السؤال التالي: "{student_query}"
            لم يتم العثور على معلومات ذات صلة في المناهج الدراسية لهذا السؤال المحدد في هذا المستوى.
            يرجى تقديم إجابة عامة إذا أمكن، ولكن اذكر أنك لم تجد تفاصيل محددة في المنهج المقدم.
            حافظ على لغتك بسيطة للغاية ومناسبة لطالب الصف {student_grade}.
            """
        else:
            prompt = f"""
            أنت مساعد تعليمي مفيد ومطلع للطلاب.
            مهمتك هي الإجابة على سؤال الطالب بناءً على سياق المناهج الدراسية المقدمة أدناه فقط.
            
            **التعليمات:**
            1.  أجب على السؤال بدقة باستخدام المعلومات الواردة في السياق المقدم فقط.
            2.  لا تستخدم أي معرفة خارجية. إذا لم تكن الإجابة موجودة في السياق، فاذكر "لا أستطيع العثور على معلومات محددة حول هذا في المنهج الدراسي المقدم."
            3.  قم بتكييف إجابتك باستخدام المفاهيم والمفردات وأسلوب الشرح المناسب لطالب في **الصف {student_grade}**.
            4.  **هام:** لا تقدم أي مفاهيم أو مصطلحات يتم تدريسها عادةً في صفوف أعلى من الصف {student_grade}. التزم تمامًا بمستوى المنهج الدراسي.
            5.  اشرح بوضوح واختصار. إذا كان السؤال باللغة العربية، أجب باللغة العربية. إذا كان بالفرنسية، أجب بالفرنسية. إذا كان بالإنجليزية، أجب بالإنجليزية.
            
            **سؤال الطالب (الصف {student_grade}):** "{student_query}"
            
            **سياق المنهج الدراسي:**
            {context_str}
            
            **يرجى تقديم إجابتك الآن:**
            """
    elif query_lang == 'fr':
        # French Prompt
        if not context_str.strip():
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
    else: # Default to English 
        # English Prompt
        if not context_str.strip():
            prompt = f"""
            You are a helpful and knowledgeable educational assistant.
            A student in Grade {student_grade} asked: "{student_query}"
            
            No relevant curriculum context was found for this specific question at this grade level.
            Please provide a general answer if you can, but state that you couldn't find specific curriculum details.
            Keep your language extremely simple and suitable for a Grade {student_grade} student.
            """
        else:
            prompt = f"""
            You are a helpful and knowledgeable educational assistant for students.
            Your task is to answer a student's question based ONLY on the provided curriculum context below.
            
            **Instructions:**
            1.  Answer the question accurately using ONLY the information from the provided context.
            2.  Do NOT use any outside knowledge. If the answer is not in the context, state "I cannot find specific information about this in the curriculum provided."
            3.  Tailor your response using concepts, vocabulary, and explanation style suitable for a **Grade {student_grade} student**.
            4.  **CRITICAL:** Do NOT introduce any concepts or terminology that would typically be taught in grades higher than Grade {student_grade}. Stick strictly to the curriculum level.
            5.  Explain clearly and concisely. If the question is in Arabic, respond in Arabic. If in French, respond in French. If in English, respond in English.
            
            **Student's Question (Grade {student_grade}):** "{student_query}"
            
            **Curriculum Context:**
            {context_str}
            
            **Please provide your answer now:**
            """

    print("\n--- Sending to LLM ---")
    print(f"  Prompt (first 500 chars): {prompt[:500]}...")
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"❌ Error generating response from LLM: {e}")
        return "Sorry, I am having trouble understanding your question right now. Please try again." # Default English fallback

# --- Main Chatbot Loop ---
if __name__ == "__main__":
    print("🚀 Démarrage du Chatbot Adapté au Niveau Scolaire 🚀")
    print("=" * 50)

    load_rag_components(faiss_index_full_path, EMBEDDING_MODEL_NAME)
    gc.collect()

    student_grade_input = input("\nEntrez la classe actuelle de l'élève (par ex. 6) : ")
    try:
        student_grade = int(student_grade_input)
        if student_grade < 1 or student_grade > 12:
            raise ValueError
    except ValueError:
        print("Niveau invalide. Par défaut : 6e année.")
        student_grade = 6

    print(f"\nSession de chatbot démarrée pour un élève de {student_grade}e année.")
    print("Tapez votre question, ou 'exit' pour quitter.")
    print("-" * 50)

    while True:
        user_query = input(f"Élève ({student_grade}e année) > ")

        if user_query.lower() == 'exit':
            print("Chatbot : Au revoir !")
            break

        if not user_query.strip():
            print("Chatbot : Veuillez entrer une question.")
            continue

        retrieved_context_docs = get_grade_appropriate_context(
            user_query,
            student_grade,
            INITIAL_RETRIEVAL_K,
            MAX_CHUNKS_TO_SEND_TO_LLM
        )

        if retrieved_context_docs:
            print("\n--- Obtention de la réponse du modèle LLM ---")
            llm_response = get_llm_response(user_query, retrieved_context_docs, student_grade)
            print("\nChatbot :", llm_response)
        else:
            print("\n--- Obtention de la réponse du modèle LLM (Aucun Contexte) ---")
            llm_response_no_context = get_llm_response(user_query, [], student_grade)
            print("\nChatbot :", llm_response_no_context)
        
        print("\n" + "-" * 50)
        gc.collect()