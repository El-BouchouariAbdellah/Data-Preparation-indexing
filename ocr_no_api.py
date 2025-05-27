import os
import gc
import pytesseract
from PIL import Image

def ocr_with_tesseract(image_path, lang='eng+ara+fra'):
    """Performs OCR using Tesseract (local)."""
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang=lang)
        img.close()
        del img
        return text
    except Exception as e:
        print(f"  ❌ Tesseract OCR Error for {os.path.basename(image_path)}: {e}")
        return None

def process_subject_images(subject_path, subject_name):

    subject_images_folder = os.path.join(subject_path, f"output_images_{subject_name}")
    
    if not os.path.exists(subject_images_folder):
        print(f"  ⚠️  Subject images folder does not exist: {subject_images_folder}") # Changed from error to warning
        return 0 # Return 0 for consistency

    images_files = [f for f in os.listdir(subject_images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images_files:
        print(f"  ⚠️  No images found in the subject folder: {subject_images_folder}") # Changed from error to warning
        return 0 # Return 0 for consistency
    
    subject_text_output_folder = os.path.join(subject_path, f"Extracted_text_{subject_name}") # Corrected folder name based on previous discussion
    os.makedirs(subject_text_output_folder, exist_ok=True)

    processed_count = 0

    for img_file in sorted(images_files):
        img_path = os.path.join(subject_images_folder, img_file)
        extracted_text_path = os.path.join(subject_text_output_folder, f"{os.path.splitext(img_file)[0]}.txt")

        if os.path.exists(extracted_text_path):
            print(f"    ✅ Text already extracted for {img_file}, skipping.")
            processed_count += 1
            continue

        print(f"    Processing {img_file}...")
        extracted_text = ocr_with_tesseract(img_path)
        
        if extracted_text is not None:
            print(f"    ✅ Text extracted for {img_file}.")
            with open(extracted_text_path, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            processed_count += 1
        else:
            print(f"    ❌ Failed to extract text for {img_file}.")
        
        gc.collect() # Collect after each image to manage memory

    print(f"  Processed {processed_count} images in {subject_name}.") 
    return processed_count

def main_process(base_folder_path):
    if not os.path.isdir(base_folder_path):
        print(f"  ❌ Base folder does not exist: {base_folder_path}")
        return
    
    print("🚀 Starting OCR processing...")
    print("=" * 50) # Added separator for clarity
    
    total_images_processed = 0
    subject_folders = [d for d in os.listdir(base_folder_path) if os.path.isdir(os.path.join(base_folder_path, d))]

    if not subject_folders:
        print(f"  ⚠️  No subject folders found in: {base_folder_path}. Nothing to process.") # Changed from error to warning
        return
    
    print(f"Found {len(subject_folders)} subject folders.")

    for subject_idx, subject in enumerate(subject_folders, 1):
        subject_path = os.path.join(base_folder_path, subject)
        print(f"\n📁 [{subject_idx}/{len(subject_folders)}] Processing: {subject}")

        # The function now always returns an int, so no more TypeErrors here
        processed_in_subject = process_subject_images(subject_path, subject)

        total_images_processed += processed_in_subject
        
        # Moved gc.collect() here to ensure it's called after each subject is fully processed
        gc.collect() 

    # --- Final Summary is now outside the loop ---
    print(f"\n🎉 OCR Process Complete. Total images processed by Tesseract: {total_images_processed}")


if __name__ == "__main__":
    # Ensure this path is correct for your system
    main_process(r"C:\Users\abdel\Desktop\السادس ابتدائي")