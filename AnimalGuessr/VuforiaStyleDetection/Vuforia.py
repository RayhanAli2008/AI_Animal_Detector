import cv2
import os

def extract_features(image_path, max_features=1000):
    """
    Simulates Step 1 of Vuforia's image detection: 
    Loads an image, converts it to grayscale, and extracts feature points.
    """
    # 1. Load the image
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        return

    original_img = cv2.imread(image_path)
    if original_img is None:
        print("Error: Could not read the image file. Check the format.")
        return

    # 2. Convert to Grayscale (Standard preparation for computer vision)
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # 3. Initialize the ORB Feature Detector
    # ORB is a fast, efficient alternative to SIFT/SURF
    orb = cv2.ORB_create(nfeatures=max_features)

    # 4. Detect keypoints (features) and compute their descriptors
    keypoints, descriptors = orb.detectAndCompute(gray_img, None)
    
    if keypoints:
        print(f"Success! Extracted {len(keypoints)} feature points.")
    else:
        print("No features found. Try an image with higher contrast or more detail.")
        return

    # 5. Draw the keypoints onto the original image for visualization
    # We'll use yellow markers (BGR: 0, 255, 255) to mimic Vuforia's dashboard
    img_with_keypoints = cv2.drawKeypoints(
        original_img, 
        keypoints, 
        None, 
        color=(0, 255, 255), 
        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
    )

    # 6. Save the output so you can inspect it
    output_filename = "feature_map_output.png"
    cv2.imwrite(output_filename, img_with_keypoints)
    print(f"Saved feature map visualization to: {output_filename}")

if __name__ == "__main__":
    # --- TEST THE CODE HERE ---
    # Put a sample image in the same folder as this script and change the name below
    sample_image = "test_target.png" 
    extract_features(sample_image)