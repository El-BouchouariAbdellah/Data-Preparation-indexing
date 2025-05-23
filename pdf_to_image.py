import os
import sys
from pdf2image import convert_from_path
from PIL import Image

def convert_pdf_to_images(pdf_path, output_folder=None, dpi=300):
    try:
        print(f"Converting {os.path.basename(pdf_path)} to images...")
        
        pages = convert_from_path(pdf_path, dpi=dpi, fmt='JPEG')
        
        processed_images = []
        for i, page in enumerate(pages):
            processed_images.append(page)
            
            # Save image if output folder is specified
            if output_folder:
                os.makedirs(output_folder, exist_ok=True)
                image_path = os.path.join(output_folder, f"page_{i+1}.jpg")
                page.save(image_path, 'JPEG', quality=95)
                print(f"  Saved: {image_path}")
        
        print(f"✅ Successfully converted {len(processed_images)} pages from {os.path.basename(pdf_path)}")
        return processed_images
        
    except Exception as e:
        print(f"❌ Error converting {pdf_path}: {e}")
        return []

def convert_folder_to_images(folder_path, output_base_folder=None):
    results = {}
    
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder {folder_path} does not exist")
        return results
    
    # Get all subdirectories (subjects)
    subjects = [d for d in os.listdir(folder_path) 
               if os.path.isdir(os.path.join(folder_path, d))]
    
    if not subjects:
        print(f"⚠️  No subject subdirectories found in {folder_path}")
        return results
    
    print(f"Found {len(subjects)} subject folders: {subjects}")
    
    for subject in subjects:
        subject_path = os.path.join(folder_path, subject)
        
        # Get all PDF files in this subject folder
        try:
            pdf_files = [f for f in os.listdir(subject_path) 
                        if f.lower().endswith('.pdf')]
        except PermissionError:
            print(f"❌ Permission denied accessing {subject_path}")
            continue
        
        if not pdf_files:
            print(f"⚠️  No PDF files found in {subject}")
            continue
        
        print(f"Found {len(pdf_files)} PDF files: {pdf_files}")
        
        subject_results = {}
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(subject_path, pdf_file)
            pdf_name = os.path.splitext(pdf_file)[0]
            
            # Determine output folder
            if output_base_folder:
                # Create organized structure: output_base/subject/pdf_name/
                pdf_output_folder = os.path.join(output_base_folder, subject, pdf_name)
            else:
                # Create output folder next to the PDF
                pdf_output_folder = os.path.join(subject_path, f"{pdf_name}_images")
            
            print(f"  📄 Converting: {pdf_file}")
            print(f"     Output to: {pdf_output_folder}")
            
            # Convert PDF to images
            images = convert_pdf_to_images(pdf_path, pdf_output_folder)
            subject_results[pdf_file] = images
        
        results[subject] = subject_results
    
    return results



def main():
    """Main function for standalone usage"""
    if len(sys.argv) < 2:
        print("Usage: python pdf_to_image..py <folder_path> [output_folder]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.isdir(input_path):
        print(f"❌ Error: {input_path} is not a valid folder")
        sys.exit(1)
    
    # Process folder structure
    results = convert_folder_to_images(input_path, output_folder)
    
    if results:
        total_pdfs = sum(len(subject_results) for subject_results in results.values())
        total_images = sum(
            len(images) for subject_results in results.values() 
            for images in subject_results.values()
        )
        print(f"✅ Converted {total_pdfs} PDFs to {total_images} images")
    else:
        print("❌ No PDFs were converted")
        sys.exit(1)

if __name__ == "__main__":
    main()