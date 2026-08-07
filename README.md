
                   REAL-TIME ORB OBJECT TRACKER
===============================================================================

Overview
This script provides real-time multi-instance object recognition and tracking
from a live webcam feed using feature-based matching. It extracts keypoints and
descriptors using ORB (Oriented FAST and Rotated BRIEF), matches them against
a target database using Brute-Force Hamming distance matching, and estimates
bounding polygons via Homography with RANSAC filtering.

Directory Structure
The script automatically looks for an objects_db directory in the same path
where it is executed. Populate this directory with subfolders representing each
target object, containing reference images captured from various angles and lighting conditions:

project_directory/
├── object_tracker.py
└── objects_db/
    ├── soda_can/
    │   ├── front.jpg
    │   ├── back.jpg
    │   └── top.png
    └── cereal_box/
        ├── view1.jpg
        └── view2.jpg
Dependencies
Install the required packages before running:

pip install opencv-python numpy
Configuration Parameters
You can fine-tune performance and detection accuracy by editing the global constants
at the top of the file:

MAX_FEATURES (1500)      : Maximum ORB features to extract per image/frame.

MIN_MATCHES (15)         : Minimum matched features required to trigger homography.

RATIO_THRESHOLD (0.75)   : Ratio threshold for Lowe's match filtering.

MAX_HAMMING_DIST (60)    : Maximum allowed feature distance to reduce false positives.

MIN_AREA / MAX_AREA      : Pixel area limits (900 to 400,000 px) to filter out
glitched or warped homography polygons.

Usage & Controls
Place your target images inside objects_db/<object_name>/.

Run the script:
python object_tracker.py

To exit the video stream:
- Press 'q' on your keyboard while the window is focused.
- Click the window's close (X) button.
===============================================================================
