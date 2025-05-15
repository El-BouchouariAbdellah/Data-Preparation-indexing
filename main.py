import os
import fitz

dst_path = r'C:\Users\abdel\Desktop\Data-Preparation-indexing\errors'
os.makedirs(dst_path, exist_ok=True)

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
            return True
        return False

    except Exception as e:
        print(f"Error analyzing {pdf_path}: {e}")
        try:
            # Move file to error folder
            os.rename(pdf_path, os.path.join(dst_path, os.path.basename(pdf_path)))
            print(f"Moved problematic file to: {dst_path}")
        except Exception as move_error:
            print(f"Error moving file: {move_error}")
        return False

subject_folder = r'C:\Users\abdel\Desktop\curriculum\الثالثة اعدادي\الرياضيات'

if not os.path.isdir(subject_folder):
    print(f"Error: '{subject_folder}' is not a valid directory")
    exit(1)

for pdf in os.listdir(subject_folder):
    pdf_path = os.path.join(subject_folder, pdf)
    if pdf.lower().endswith('.pdf'):
        is_image = is_image_based_pdf(pdf_path)
        if is_image:
            print(f"{pdf} is an image-based PDF.")
        else:
            print(f"{pdf} is a text-based PDF.")
