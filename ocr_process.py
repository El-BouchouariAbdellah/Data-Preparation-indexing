import os
import gc
from google.cloud import vision
import pytesseract
from PIL import Image

try:
    client = vision.ImageAnnotatorClient()
except Exception as e:
    print(f"Error initializing Google Vision client: {e}")
    client = None

def ocr_with_cloud_vision(image_path):
    if not client:
        print("Google Vision client is not initialized.")
        return None
    try:
        with open(image_path,'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content=content) # cloud vision api excpects an input of type vision.Image 

        response = client.document_text_detection(image=image)
        print(response) # document_text_detection is a cloud vision api method that detects text in the image
        full_text = response.full_text_annotation.text if response.full_text_annotation else "" 
        return full_text
    except Exception as e:
        print(f"  ❌ Google Cloud Vision API Error for {os.path.basename(image_path)}: {e}")
        return None
    
ocr_with_cloud_vision("Sans titre.png")