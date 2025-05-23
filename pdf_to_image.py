from pdf2image import convert_from_path
import os

pdf_path = r'C:\Users\abdel\Desktop\Mixed\file_1109_first_cell.pdf'
pdf_name = os.path.splitext(os.path.basename(pdf_path))[0] 
print(pdf_name)
output_folder = r'C:\Users\abdel\Desktop\Mixed\output_images'
os.makedirs(output_folder, exist_ok=True)

images = convert_from_path(pdf_path, dpi=300)

for i, img in enumerate(images, start=1):
    img_filename = f"{pdf_name}_{i}.jpeg"
    img.save(os.path.join(output_folder, img_filename), "JPEG")
