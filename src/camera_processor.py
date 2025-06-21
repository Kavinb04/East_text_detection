import cv2
import pytesseract
import numpy as np
import os
import sys
import time
# from src.tts_utils import speak_text

script_dir = os.path.dirname(__file__)
parent_dir = os.path.join(script_dir, os.pardir)
sys.path.append(parent_dir) # Add EAST_text_detection/ to path
sys.path.append(os.path.join(parent_dir, 'east_pretrained')) # Add east_pretrained/ to path

from src.east_utils import decode_east_predictions, east_preprocessing, apply_nms

# Path to the EAST model 
MODEL_PATH = os.path.join(parent_dir, 'east_pretrained', 'frozen_east_text_detection.pb')

# EAST model parameters
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
INPUT_SIZE = (640, 640)
DEBOUNCE_TIME = 2

# Initialize EAST Text Detector 
try:
    net = cv2.dnn.readNet(MODEL_PATH)
    # Define the output layers for the EAST model
    # First elem is the scores map and second is the geometry map.
    output_layers = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    print(f"EAST model loaded from: {MODEL_PATH}")
except Exception as e:
    print(f"Error loading EAST model from {MODEL_PATH}: {e}")
    print("Please ensure 'frozen_east_text_detection.pb' is in the 'east_pretrained/' directory.")
    sys.exit(1)

# Camera feed
def run_camera_detection():
    cap = cv2.VideoCapture(0) 

    if not cap.isOpened():
        print("Error: Could not open video stream. Check camera connection or index.")
        sys.exit(1)

    print("Press SPACE to read detected text aloud. Press 'q' to quit the camera feed.")

    last_spoken_text = ""
    last_spoken_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting.")
            break

        original_h, original_w = frame.shape[:2]

        # Preprocess the frame for the EAST model
        blob, rW, rH = east_preprocessing(frame.copy(), INPUT_SIZE)
        net.setInput(blob)
        (scores, geometry) = net.forward(output_layers)

        # Decode the EAST predictions
        rects, confidences = decode_east_predictions(scores, geometry, CONF_THRESHOLD)

        # Apply Non-Maximum Suppression
        indices = apply_nms(rects, confidences, NMS_THRESHOLD)

        detected_texts = []

        # Draw bounding boxes and recognize text
        if len(indices) > 0:
            for i in indices:
                # Get the box in (x, y, w, h) format
                x, y, w, h = rects[i]

                # Scale the bounding box coordinates back to the original frame size
                x = int(x * rW)
                y = int(y * rH)
                w = int(w * rW)
                h = int(h * rH)

                # Ensure coordinates are within frame boundaries
                x = max(0, x)
                y = max(0, y)
                w = min(original_w - x, w)
                h = min(original_h - y, h)

                # Draw the bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Extract the Region of Interest for Tesseract
                cropped_roi = frame[y:y+h, x:x+w]
                if cropped_roi.size > 0:
                    # Convert to grayscale for better OCR performance
                    gray_roi = cv2.cvtColor(cropped_roi, cv2.COLOR_BGR2GRAY)
                    _, thresh_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    # Use thresh_roi instead of gray_roi for Tesseract
                    text = pytesseract.image_to_string(thresh_roi, config='--psm 7').strip()
                    if text:
                        detected_texts.append(text)
                        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Combine all detected texts for TTS
        combined_text = " ".join(detected_texts)

        # Display the frame
        cv2.imshow("Live Text Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        now = time.time()

        # SPACE triggers TTS with debounce
        if key == ord(' '):
            if combined_text and (combined_text != last_spoken_text or now - last_spoken_time > DEBOUNCE_TIME):
                # speak_text(combined_text)
                last_spoken_text = combined_text
                last_spoken_time = now

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera feed stopped.")

if __name__ == '__main__':
    run_camera_detection()