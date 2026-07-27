# ==============================================================================
# OPENCV MULTI-OBJECT FEATURE TRACKING & FEATURE PREVIEW SYSTEM
# ==============================================================================
#
# A lightweight, real-time computer vision system built with Python, OpenCV,
# and ORB (Oriented FAST and Rotated BRIEF) features. This project allows you to
# build a local image database of physical objects (supporting multiple viewpoints
# per object) and track them live via webcam using perspective homography.
#
# ------------------------------------------------------------------------------
# FEATURES
# ------------------------------------------------------------------------------
# - Multi-Object Real-Time Tracking (Vuforia.py): Simultaneously detects and
#   tracks registered objects in a live camera feed.
# - Multi-View Data Points: Store multiple reference angles (e.g., front, back,
#   side views) for a single object within its dedicated folder.
# - Folder-Based Object Database: Objects are automatically loaded and labeled
#   based on subfolder names inside objects_db/.
# - Perspective Homography & Outlier Filtering: Uses RANSAC and Lowe's Ratio
#   Test to filter out ambiguous matches, ensuring stable bounding box projection.
# - Convexity Validation: Prevents distorted or self-intersecting false detections
#   using cv2.isContourConvex.
# - Feature Visualization Tool (PreviewImagePoints.py): Standalone script to inspect
#   extracted keypoints on any image before adding it to the database.
#
# ------------------------------------------------------------------------------
# PROJECT STRUCTURE
# ------------------------------------------------------------------------------
# .
# ├── Vuforia.py                # Main real-time webcam object tracking engine
# ├── PreviewImagePoints.py     # Tool to visualize extracted feature points
# ├── README.md                 # Project documentation
# └── objects_db/               # Auto-created folder for target object database
#     ├── Rubiks_Cube/          # Subfolder name becomes the object label
#     │   ├── front.jpg         # Viewpoint 1
#     │   └── back.jpg          # Viewpoint 2
#     └── Coffee_Mug/
#         └── side.png
#
# ------------------------------------------------------------------------------
# PREREQUISITES & INSTALLATION
# ------------------------------------------------------------------------------
# Requirements:
#   - Python 3.7+
#   - A working webcam / camera module
#
# Install Dependencies:
#   pip install opencv-python numpy
#
# ------------------------------------------------------------------------------
# USAGE GUIDE
# ------------------------------------------------------------------------------
# 1. Feature Visualization (PreviewImagePoints.py)
#
#    Run via Terminal/CLI:
#      python PreviewImagePoints.py path/to/your_image.png
#
#    Run Interactively:
#      python PreviewImagePoints.py
#      Prompt: Enter the path to the image: path/to/your_image.png
#
#    Output:
#      Saves a new image named <image_name>_features.<ext> in the same
#      directory as the script, marked with yellow feature points.
#
# 2. Setting Up the Object Database (objects_db)
#
#    - Run Vuforia.py once to auto-create the objects_db/ folder.
#    - Create subfolders inside objects_db/ named after your object
#      (e.g., objects_db/Book_Cover/).
#    - Add one or more target images inside that folder (e.g., front.png, back.jpg).
#
# 3. Real-Time Tracking (Vuforia.py)
#
#    Run:
#      python Vuforia.py
#
#    Controls:
#      Press 'q' or close the window to exit.
#
#    Visual Feedback:
#      - Searching... (Red text): Searching for objects registered in objects_db/.
#      - Green Polygon & Label: Object matched! Displays bounding box and folder name.
#
# ------------------------------------------------------------------------------
# HOW IT WORKS
# ------------------------------------------------------------------------------
# 1. ORB Keypoint Extraction: Detects up to 1000 keypoints per image view and
#    camera frame.
# 2. K-Nearest Neighbors (KNN) Matching: Finds candidate descriptor pairs between
#    camera frames and target images.
# 3. Lowe's Ratio Test: Retains only unique matches where top match distance is
#    significantly closer than the second nearest (m.distance < 0.75 * n.distance).
# 4. Homography Estimation: Computes the perspective matrix using RANSAC to handle
#    rotation, scale, and perspective changes.
# 5. Shape Validation: Verifies that projected 4-corner boundary forms a convex
#    polygon before drawing UI elements.
#
# ------------------------------------------------------------------------------
# CONFIGURATION & PERFORMANCE TUNING
# ------------------------------------------------------------------------------
# Inside Vuforia.py, you can fine-tune parameters to fit your environment:
#   - MIN_MATCH_COUNT = 15 : Lowering to 10 increases sensitivity; raising to 20
#                            prevents false positives.
#   - nfeatures = 1000     : Increase to 2000 in cv2.ORB_create() for fine textures.
#   - Ratio Threshold = 0.75: Lowering to 0.65 forces stricter match uniqueness.
#
# ------------------------------------------------------------------------------
# LICENSE
# ------------------------------------------------------------------------------
# This project is open-source and available under the MIT License.
# ==============================================================================
