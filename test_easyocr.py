import easyocr
import os
import time

def test_easyocr(image_path, languages):

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None, None

    print(f"\n--- Testing EasyOCR on: {os.path.basename(image_path)} ---")
    print(f"  Languages: {languages}")

    try:
        start_time = time.time()
        reader = easyocr.Reader(languages, gpu=False) 
        
        result = reader.readtext(image_path)
        end_time = time.time()

        print(f"  OCR completed in {end_time - start_time:.2f} seconds.")
        print("\n--- Extracted Text (EasyOCR) ---")
        extracted_text = ""
        if result:
            for (bbox, text, prob) in result:
                print(f"'{text}' (Confidence: {prob:.2f})")
                extracted_text += text + "\n"
        else:
            print("No text found.")

        output_filename = os.path.splitext(os.path.basename(image_path))[0] + "_easyocr_output.txt"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(extracted_text)
        print(f"\nExtracted text saved to {output_filename}")
        # -------------------------

        return extracted_text, output_filename

    except Exception as e:
        print(f"An error occurred during EasyOCR processing: {e}")
        return None, None

if __name__ == "__main__":
    test_image_filename = "file_136_first_cell_page_1.jpg"  # Path to your test image file

    test_languages = ['ar'] 
    extracted_text_easyocr, output_file = test_easyocr(test_image_filename, test_languages)