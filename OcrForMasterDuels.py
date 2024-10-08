import cv2
import pytesseract
from pytesseract import Output

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

        # Optional: Process every nth frame (to avoid too much processing)
        # if frame_number % 10 == 0:
        
        # Convert the frame to grayscale (pytesseract works better on grayscale images)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use pytesseract to do OCR on the frame
        text = pytesseract.image_to_string(gray_frame, output_type=Output.STRING)
        
        # Save the extracted text to the file
        text_file.write(f"Frame {frame_number}: {text}\n")
        
        frame_number += 1

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
