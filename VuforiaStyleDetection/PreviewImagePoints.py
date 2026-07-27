import cv2
import os
import sys

def extract_features(image_path, max_features=1000):
    """
    Loads an image, converts it to grayscale, extracts ORB feature points,
    and saves the visualization in the same directory as this script.
    """
    # Get the directory where this script resides
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Normalize path (handles relative paths and removes surrounding quotes)
    image_path = os.path.abspath(image_path.strip("\"'"))

    # 1. Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        return

    original_img = cv2.imread(image_path)
    if original_img is None:
        print("Error: Could not read the image file. Check the file format.")
        return

    # 2. Convert to Grayscale
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # 3. Initialize ORB Feature Detector
    orb = cv2.ORB_create(nfeatures=max_features)

    # 4. Detect keypoints
    keypoints, descriptors = orb.detectAndCompute(gray_img, None)
    
    if not keypoints:
        print("No features found. Try an image with higher contrast or more detail.")
        return

    print(f"Success! Extracted {len(keypoints)} feature points from '{os.path.basename(image_path)}'.")

    # 5. Draw keypoints onto original image (Yellow markers)
    img_with_keypoints = cv2.drawKeypoints(
        original_img, 
        keypoints, 
        None, 
        color=(0, 255, 255), 
        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
    )

    # 6. Save output in the same folder where this script lives
    _, full_name = os.path.split(image_path)
    file_name, ext = os.path.splitext(full_name)
    output_filename = os.path.join(script_dir, f"{file_name}_features{ext}")
    
    cv2.imwrite(output_filename, img_with_keypoints)
    print(f"Saved visualization to: {output_filename}")

if __name__ == "__main__":
    # If a path was passed via command line 
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    else:
        # Otherwise, ask in the terminal
        target_path = input("Enter the path to the image: ")

    if target_path.strip():
        extract_features(target_path)