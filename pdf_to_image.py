import os
import sys
import gc
from pdf2image import convert_from_path
from PIL import Image

def convert_pdf_to_images(pdf_path, output_folder=None, pdf_name_for_filename=None, dpi=200):
    try:
        print(f"Converting {os.path.basename(pdf_path)}...")
        
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
        
        # Process pages in batches to manage memory
        batch_size = 5  # Process 5 pages at a time
        page_count = 0
        
        # Get total page count first (lightweight operation) just to check if the pdf is valid  
        try:
            # Use a small DPI just to count pages
            temp_pages = convert_from_path(pdf_path, dpi=50, fmt='JPEG', first_page=1, last_page=1)
            del temp_pages
            gc.collect()
        except:
            pass
        
        current_page = 1
        
        while True:
            try:
                # Convert pages in small batches
                pages = convert_from_path(
                    pdf_path, 
                    dpi=dpi, 
                    fmt='JPEG',
                    first_page=current_page,
                    last_page=current_page + batch_size - 1,
                    thread_count=1  # Limit threads to reduce memory usage
                )
                
                if not pages:
                    break
                
                # Process and save each page immediately
                for i, page in enumerate(pages):
                    page_num = current_page + i
                    page_count += 1
                    
                    if output_folder:
                        # Optimize image before saving
                        if page.mode != 'RGB':
                            page = page.convert('RGB')
                        
                        image_filename = f"{pdf_name_for_filename}_page_{page_num}.jpg"
                        image_path = os.path.join(output_folder, image_filename)
                        
                        # Save with optimized settings
                        page.save(image_path, 'JPEG', quality=85, optimize=True)
                        print(f"  Saved: {image_filename}")
                    
                    # Clear page from memory immediately
                    page.close()
                    del page
                
                # Clear the batch from memory
                del pages
                gc.collect()  # Force garbage collection
                
                current_page += batch_size
                
            except Exception as batch_error:
                # If we can't get more pages, we're done
                if "Invalid page" in str(batch_error) or "out of range" in str(batch_error):
                    break
                else:
                    print(f"  ⚠️  Batch error at page {current_page}: {batch_error}")
                    current_page += 1
                    continue
        
        print(f"✅ Converted {page_count} pages from {os.path.basename(pdf_path)}")
        return page_count
        
    except Exception as e:
        print(f"❌ Error converting {pdf_path}: {e}")
        return 0
    finally:
        # Ensure cleanup
        gc.collect()

def convert_folder_to_images(folder_path):
    
    results = {}
    
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder {folder_path} does not exist")
        return results
    
    # Get all subdirectories (subjects)
    try:
        subjects = [d for d in os.listdir(folder_path) 
                   if os.path.isdir(os.path.join(folder_path, d))]
    except PermissionError:
        print(f"❌ Permission denied accessing {folder_path}")
        return results
    
    if not subjects:
        print(f"⚠️  No subdirectories found in {folder_path}")
        return results
    
    print(f"Found {len(subjects)} subject folders")
    
    total_pdfs = 0
    total_images = 0
    
    for subject_idx, subject in enumerate(subjects, 1):
        subject_path = os.path.join(folder_path, subject)
        print(f"\n📁 [{subject_idx}/{len(subjects)}] Processing: {subject}")
        
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
        
        print(f"Found {len(pdf_files)} PDF files")
        
        # Create single output folder for this subject
        subject_output_folder = os.path.join(subject_path, f"output_images_{subject}")
        
        subject_images = 0
        
        for pdf_idx, pdf_file in enumerate(pdf_files, 1):
            pdf_path = os.path.join(subject_path, pdf_file)
            pdf_name = os.path.splitext(pdf_file)[0]
            
            print(f"  📄 [{pdf_idx}/{len(pdf_files)}] {pdf_file}")
            
            # Convert PDF to images
            page_count = convert_pdf_to_images(pdf_path, subject_output_folder, pdf_name)
            subject_images += page_count
            total_pdfs += 1
            
            # Force cleanup between PDFs
            gc.collect()
        
        total_images += subject_images
        print(f"✅ {subject}: {len(pdf_files)} PDFs → {subject_images} images")
        results[subject] = len(pdf_files)
    
    print(f"\n🎉 Final Summary: {total_pdfs} PDFs → {total_images} images")
    return results

def main():
    """Main function for standalone usage"""
    if len(sys.argv) < 2:
        print("Usage: python pdf_converter.py <folder_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    if not os.path.isdir(input_path):
        print(f"❌ Error: {input_path} is not a valid folder")
        sys.exit(1)
    
    print("🚀 Starting PDF to Images conversion...")
    print("💡 Optimized for memory efficiency and stability")
    print("=" * 50)
    
    # Process folder structure
    results = convert_folder_to_images(input_path)
    
    if not results:
        print("❌ No PDFs were converted")
        sys.exit(1)

if __name__ == "__main__":
    main()