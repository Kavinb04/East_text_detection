# main.py

import os
import sys

# IMPORTANT: Adjust Python Path for imports
# This ensures that modules within 'src' and 'east_pretrained' can be imported correctly
# when 'main.py' is run from the root directory 'EAST_text_detection/'.

# Get the directory where main.py resides (EAST_text_detection/)
project_root = os.path.dirname(os.path.abspath(__file__))

# Add 'EAST_text_detection/src' to the Python path
sys.path.append(os.path.join(project_root, 'src'))

# Add 'EAST_text_detection/east_pretrained' to the Python path
sys.path.append(os.path.join(project_root, 'east_pretrained'))

# Now, import your camera_text_detector.py script.
# We import it as 'camera_detector' to avoid naming conflicts and make calls clear.
try:
    import camera_text_detector as camera_detector
    print("Successfully imported camera_text_detector.")
except ImportError as e:
    print(f"Error importing camera_text_detector: {e}")
    print("Please ensure 'camera_text_detector.py' and 'east_utils.py' are in the 'src/' directory.")
    print("Also ensure your Python path setup in main.py is correct.")
    sys.exit(1)


def run_live_detection():
    """
    Function to initiate the live camera text detection.
    This calls the main execution logic defined in camera_text_detector.py.
    """
    print("\n--- Starting Live Camera Text Detection ---")
    camera_detector.run_camera_detection() # This calls the function in your camera_text_detector.py
    print("--- Live Camera Text Detection Finished ---")


if __name__ == "__main__":
    # This block ensures run_live_detection() is called only when main.py is executed directly
    run_live_detection()