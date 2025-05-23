import os
import fitz
import shutil
import csv
import sys

if len(sys.argv) > 1:
    grade_folder = sys.argv[1]
else:
    grade_folder = input("Enter the path to the grade folder: ").strip()

grade_name = os.path.basename(grade_folder)

# Check if the grade folder exists
if not os.path.isdir(grade_folder):
    print(f"Error: '{grade_folder}' is not a valid directory")
    exit(1)

def create_report_csv(stats_data):
    total_image_based = 0
    total_text_based = 0
    total_corrupted = 0
    try:
        csv_file_path = os.path.join(os.getcwd(), grade_name +'_analysis_report.csv')
        with open(csv_file_path, "w", newline='',encoding='UTF-8') as csvfile:
            fieldnames = ['Subject', 'Image-based PDFs', 'Text-based PDFs', 'Corrupted PDFs', 'Total PDFs'] 
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for subject, data in stats_data.items():
                writer.writerow({
                    'Subject': subject,
                    'Image-based PDFs': data['image_based'],
                    'Text-based PDFs': data['text_based'],
                    'Corrupted PDFs': data['corrupted'],
                    'Total PDFs': data['total']
                })
                total_image_based += data['image_based']
                total_text_based += data['text_based']
                total_corrupted += data['corrupted']
            
            writer.writerow({
                'Subject': 'Total',
                'Image-based PDFs': total_image_based,
                'Text-based PDFs': total_text_based,
                'Corrupted PDFs': total_corrupted,
                'Total PDFs': total_image_based + total_text_based + total_corrupted
            })
        print(f"Report saved to {csv_file_path}")
        return csv_file_path
    except Exception as e:
        print(f"Error creating CSV report: {e}")
        return None


def is_image_based_pdf(pdf_path, ocr_path, corr_path):
    doc = None
    try:
        pdf_path = os.path.normpath(pdf_path)
        doc = fitz.open(pdf_path)
        pages_to_check = min(2, len(doc))
        text_content = ""
        for page_num in range(pages_to_check):
            text_content += doc[page_num].get_text()

        is_image = len(text_content.strip()) < 100 * pages_to_check

    except Exception as e:
        print(f"Error analyzing {pdf_path}: {e}")
        try:
            shutil.move(pdf_path, os.path.join(corr_path, os.path.basename(pdf_path)))
            print(f"Moved corrupted file to: {corr_path}")
        except Exception as move_err:
            print(f"Error moving corrupted file: {move_err}")
        return None

    finally:
        if doc:
            doc.close()

    if is_image:
        try:
            shutil.move(pdf_path, os.path.join(ocr_path, os.path.basename(pdf_path)))
            print(f"Moved {os.path.basename(pdf_path)} to OCR folder.")
        except Exception as e:
            print(f"Error moving file to OCR folder: {e}")
        return True
    return False


OCR_counter = 0
Text_counter = 0
Corrupted_counter = 0
stats_data = {}

for subject in os.listdir(grade_folder):
    if subject.endswith("_OCR") or subject.endswith("_corrupted"): # To avoid recursive processing
        print(f" ❌❌ Skipping '{subject}' as it is an OCR or corrupted folder")
        continue
    subject_path = os.path.join(grade_folder, subject)
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
    subject_Corrupted_Counter = 0

    for pdf in os.listdir(subject_path):
                    
        pdf_path = os.path.join(subject_path, pdf)
        image = is_image_based_pdf(pdf_path, ocr_path, corr_path)
        if image is True:
            print(f"{pdf} is an image-based PDF.")
            subject_OCR_Counter += 1
            OCR_counter += 1
        elif image is False:
            print(f"{pdf} is a text-based PDF.")
            subject_Text_Counter += 1
            Text_counter += 1
        else:
            print(f"{pdf} is a corrupted PDF.")
            subject_Corrupted_Counter += 1
            Corrupted_counter += 1
    
    stats_data[subject] = {
        'image_based': subject_OCR_Counter,
        'text_based': subject_Text_Counter,
        'corrupted': subject_Corrupted_Counter,
        'total': subject_OCR_Counter + subject_Text_Counter + subject_Corrupted_Counter
    }

    print(f"Subject '{subject}': " 
          f"Image-based PDFs: {subject_OCR_Counter}, " 
          f"Text-based PDFs: {subject_Text_Counter}, "
          f"Corrupted PDFs: {subject_Corrupted_Counter}")
    print("==" * 20)

print(f"Total image-based PDFs: {OCR_counter}")
print(f"Total text-based PDFs: {Text_counter}")
print(f"Total corrupted PDFs: {Corrupted_counter}")
csv_file_path = create_report_csv(stats_data)