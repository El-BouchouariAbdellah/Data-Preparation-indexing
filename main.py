import os
import fitz
import shutil

subject_folder = r'C:\Users\abdel\Desktop\curriculum\الثالثة اعدادي\الرياضيات'
ocr_path = subject_folder + '_OCR'
corr_path = subject_folder + '_corrupted'

# Check if the subject folder exists
if not os.path.isdir(subject_folder):
    print(f"Error: '{subject_folder}' is not a valid directory")
    exit(1)

# Ensure the OCR & corrupted folders exists (creates it if needed)
os.makedirs(corr_path, exist_ok=True)
os.makedirs(ocr_path, exist_ok=True)

def is_image_based_pdf(pdf_path):
    """Check if a PDF is primarily image-based (requires OCR)."""
    try:
        doc = fitz.open(pdf_path)
        pages_to_check = min(2, len(doc))
        text_content = ""
        for page_num in range(pages_to_check):
            page = doc[page_num]
            text_content += page.get_text()
        doc.close()

        if len(text_content.strip()) < 100 * pages_to_check:
            try:
                shutil.copy(pdf_path, ocr_path) # Copy to OCR folder
                print(f"Copied {os.path.basename(pdf_path)} to OCR folder.")
            except Exception as e:
                print(f"Error copying file to OCR folder: {e}")
            return True
        return False

    except Exception as e:
        print(f"Error analyzing {pdf_path}: {e}")
        try:
            # Move file to error folder
            os.rename(pdf_path, os.path.join(corr_path, os.path.basename(pdf_path)))
            print(f"Moved corrupted file to: {corr_path}")
        except Exception as move_error:
            print(f"Error moving file: {move_error}")
        return False



OCR_counter = 0
Text_counter = 0
for pdf in os.listdir(subject_folder):
    pdf_path = os.path.join(subject_folder, pdf)
    if is_image_based_pdf(pdf_path):
        print(f"{pdf} is an image-based PDF.")
        OCR_counter  += 1
    else:
        print(f"{pdf} is a text-based PDF.")
        Text_counter += 1
print(f"Total image-based PDFs: {OCR_counter}")
print(f"Total text-based PDFs: {Text_counter}")
