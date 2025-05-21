import os
import fitz
import shutil

grade_folder = r'C:\Users\abdel\Desktop\curriculum\الثالثة اعدادي'


# Check if the grade folder exists
if not os.path.isdir(grade_folder):
    print(f"Error: '{grade_folder}' is not a valid directory")
    exit(1)


def is_image_based_pdf(pdf_path,ocr_path,corr_path):
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
            # Move file to corrupted folder
            os.rename(pdf_path, os.path.join(corr_path, os.path.basename(pdf_path)))
            print(f"Moved corrupted file to: {corr_path}")
        except Exception as e:
            print(f"Error moving file: {e}")
        return False


OCR_counter = 0
Text_counter = 0

for subject in os.listdir(grade_folder):
    if subject.endswith("_OCR") or subject.endswith("_corrupted"): # To avoid recursive processing
        print(f" ❌❌ Skipping '{subject}' as it is an OCR or corrupted folder")
        continue
    subject_path = os.path.join(grade_folder,subject)
    if not os.path.isdir(subject_path):
        print(f" ⚠⚠ Error: '{subject_path}' is not a valid directory")
        continue
    print(f"Processing subject: {subject}")
    ocr_path = subject_path + '_OCR'
    corr_path = subject_path + '_corrupted'
    # Ensure the OCR & corrupted folders exists (creates it if needed)
    os.makedirs(corr_path, exist_ok=True)
    os.makedirs(ocr_path, exist_ok=True)

    print("==" * 20)
    subject_OCR_Counter = 0
    subject_Text_Counter = 0

    for pdf in os.listdir(subject_path):
        pdf_path = os.path.join(subject_path, pdf)
        if is_image_based_pdf(pdf_path,ocr_path,corr_path):
            print(f"{pdf} is an image-based PDF.")
            subject_OCR_Counter += 1
            OCR_counter  += 1
        else:
            print(f"{pdf} is a text-based PDF.")
            subject_Text_Counter += 1
            Text_counter += 1

    print(f"Subject '{subject}' - Image-based PDFs: {subject_OCR_Counter}, Text-based PDFs: {subject_Text_Counter}")
    print("==" * 20)
print(f"Total image-based PDFs: {OCR_counter}")
print(f"Total text-based PDFs: {Text_counter}")
