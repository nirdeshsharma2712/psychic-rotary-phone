# import os
# from PIL import Image
# import easyocr
# import cv2

# # Import your model functions
# from src.model import extract_text_from_image, predict_entity_value

# def main():
#     # Directory where your images are stored
#     images_dir = 'images'
    
#     # Directory to save output text files
#     output_dir = 'output_text'
#     os.makedirs(output_dir, exist_ok=True)  # Create output directory if not exists

#     # Initialize EasyOCR reader
#     reader = easyocr.Reader(['en'], gpu=False)

#     # Iterate over each image in the images directory
#     for image_file in os.listdir(images_dir):
#         image_path = os.path.join(images_dir, image_file)

#         # Load the image
#         image = Image.open(image_path)

#         # Extract text from the image using EasyOCR
#         text = extract_text_from_image(image, reader)

#         # Save the OCR text to a file
#         output_file = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}.txt")
#         with open(output_file, 'w') as f:
#             f.write(text)
        
#         # Print the extracted text to the console
#         print(f"Extracted text for {image_file}:\n{text}\n")

# if __name__ == '__main__':
#     main()
