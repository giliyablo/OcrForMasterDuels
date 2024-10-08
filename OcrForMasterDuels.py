import cv2
import pytesseract
from pytesseract import Output

def preprocess_for_title(frame):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Upscale the image to improve OCR accuracy
    upscale_frame = cv2.resize(gray_frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Apply thresholding to binarize the image (black and white)
    _, threshold_frame = cv2.threshold(upscale_frame, 150, 255, cv2.THRESH_BINARY)

    # Optional: Denoising to remove small specks
    denoised_frame = cv2.fastNlMeansDenoising(threshold_frame, None, h=30)

    return denoised_frame

# Load the video
video_path = 'ocr2.mp4'
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video file")

# Frame count
frame_number = 0

# Open a file to save the extracted text with utf-8 encoding
with open('extracted_text.txt', 'w', encoding='utf-8') as text_file:
    # Process the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break

        # Process every nth frame to avoid too much processing
        if frame_number % 10 == 0:
            # Crop the frame to the area of the screen where card titles are located
            # Assuming you know the coordinates of the card title region (you may need to adjust these)
            card_title_area = frame[140:440, 40:80]  # Example coordinates

            # Preprocess the frame for better OCR results
            processed_frame = preprocess_for_title(card_title_area)
            
            # Use pytesseract to do OCR on the title area
            custom_config = r'--oem 3 --psm 6'  # oem 3: Best OCR engine, psm 6: Assume a single uniform block of text
            title_text = pytesseract.image_to_string(processed_frame, config=custom_config, output_type=Output.STRING)
            
            # Save the extracted card title to the file
            text_file.write(f"Frame {frame_number} (Card Title): {title_text}\n")
        
        frame_number += 1

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
