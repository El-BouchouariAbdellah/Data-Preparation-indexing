import os
import json
import gc 


SUBJECT_METADATA_MAP = {
    "Français": {"grade": 6, "subject": "French Language"},
    "الاجتماعيات": {"grade": 6, "subject": "Social Studies"},
    "التربية الاسلامية": {"grade": 6, "subject": "Islamic Education"},
    "الرياضيات": {"grade": 6, "subject": "Mathematics"},
    "اللغة العربية": {"grade": 6, "subject": "Arabic Language"},
    "النشاط العلمي": {"grade": 6, "subject": "Scientific Activity"},
}

# Default values if a subject folder is NOT found in the map above.
DEFAULT_GRADE = 0
DEFAULT_SUBJECT = "Unspecified"

BASE_CURRICULUM_PATH = r"C:\Users\abdel\Desktop\Data-Preparation-indexing\السادس ابتدائي" 

TAGGED_DATA_OUTPUT_FILE = "tagged_curriculum_data.json"


def tag_and_collect_documents(base_path, metadata_map, default_g, default_s):

    all_tagged_documents = []

    if not os.path.isdir(base_path):
        print(f"❌ Error: Base path does not exist: {base_path}")
        return []

    print(f"\n--- Starting Document Tagging from: {base_path} ---")

    # Loop through each folder directly inside the base path (these are your subject folders)
    for subject_folder_name in os.listdir(base_path):
        subject_full_path = os.path.join(base_path, subject_folder_name)

        # Skip if it's not a directory or if it's a hidden folder/file
        if not os.path.isdir(subject_full_path) or subject_folder_name.startswith('.'):
            continue

        # --- Determine Grade and Subject for this subject folder ---
        metadata_for_subject = metadata_map.get(subject_folder_name, { # dictionary.get(key, default_value)  
            "grade": default_g,
            "subject": default_s
        })
        current_grade = metadata_for_subject["grade"]
        current_subject = metadata_for_subject["subject"]
        print(f"\n📁 Processing subject folder: '{subject_folder_name}' (Assigned Grade: {current_grade}, Subject: {current_subject})")
        # -----------------------------------------------------------

        # Now, list .txt files DIRECTLY in the subject_full_path (e.g., in 'Français' folder)
        cleaned_text_files = [f for f in os.listdir(subject_full_path) if f.endswith('.txt')]
        if not cleaned_text_files:
            print(f"  ⚠️  No .txt files found directly in '{subject_full_path}'. Skipping this subject.")
            continue

        # Process each cleaned text file (which represents a page or a small document)
        for text_file_name in sorted(cleaned_text_files):
            file_path = os.path.join(subject_full_path, text_file_name) # Path is directly to the .txt file
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    cleaned_text_content = f.read()

                if not cleaned_text_content.strip(): # Skip empty files (after stripping whitespace)
                    print(f"    Skipping empty or whitespace-only file: {text_file_name}")
                    continue

                # Create a dictionary for this document/page, including its content and metadata
                document_data = {
                    "text_content": cleaned_text_content,
                    "metadata": {
                        "grade": current_grade,
                        "subject": current_subject,
                        "source_file": text_file_name, 
                        "original_subject_folder": subject_folder_name
                    }
                }
                all_tagged_documents.append(document_data)
                print(f"    ✅ Tagged: {text_file_name}")

            except Exception as e:
                print(f"  ❌ Error reading or processing {text_file_name}: {e}")
        
        gc.collect() # Trigger garbage collection after processing each subject's files

    print(f"\n--- Tagging Complete: {len(all_tagged_documents)} documents/pages tagged. ---")
    return all_tagged_documents

# --- Main execution block ---
if __name__ == "__main__":
    print("Starting Document Tagging...")

    # Call the tagging function
    tagged_curriculum_data = tag_and_collect_documents(
        BASE_CURRICULUM_PATH,
        SUBJECT_METADATA_MAP,
        DEFAULT_GRADE,
        DEFAULT_SUBJECT
    )

    # Save the result to a JSON file. This file will be the input for the next step (Chunking).
    try:
        # Save the output JSON in the same directory as the script.
        output_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), TAGGED_DATA_OUTPUT_FILE)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(tagged_curriculum_data, f, ensure_ascii=False, indent=2)
        print(f"\nSuccessfully saved all tagged data to: {output_file_path}")
    except Exception as e:
        print(f"❌ Error saving tagged data to JSON: {e}")

    # Optional: Print a preview of the first few tagged documents
    if tagged_curriculum_data:
        print("\n--- Preview of First 2 Tagged Documents ---")
        for i, doc in enumerate(tagged_curriculum_data[:2]):
            print(f"Document {i+1}:")
            print(f"  Metadata: {doc['metadata']}")
            print(f"  Text (first 200 chars): {doc['text_content'][:200]}...")
            print("-" * 30)
    else:
        print("\nNo documents were tagged. Please check your paths and folder structure.")