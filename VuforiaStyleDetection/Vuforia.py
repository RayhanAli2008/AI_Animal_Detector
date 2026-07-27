import cv2
import numpy as np
import os

# 1. SET UP DATABASE DIRECTORY (Relative to script's directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(SCRIPT_DIR, "objects_db")

if not os.path.exists(DB_DIR):
    os.makedirs(DB_DIR)
    print(f"Created '{DB_DIR}' folder. Add subfolders named after your objects with target images inside them.")
    exit()

orb = cv2.ORB_create(nfeatures=1000)

# Store loaded descriptors and shapes per object
# Structure: [{'name': 'ObjectName', 'views': [{'descriptors': ..., 'kp': ..., 'shape': ...}]}]
db_objects = []


# Scan all subdirectories in objects_db
for object_name in os.listdir(DB_DIR):
    object_path = os.path.join(DB_DIR, object_name)
    
    if os.path.isdir(object_path):
        views = []
        for img_name in os.listdir(object_path):
            img_path = os.path.join(object_path, img_name)
            
            # Load view image in grayscale
            target_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if target_img is None:
                continue
            
            kp_target, des_target = orb.detectAndCompute(target_img, None)
            
            if des_target is not None:
                views.append({
                    "descriptors": des_target,
                    "kp": kp_target,
                    "shape": target_img.shape
                })
                print(f"  └─ Loaded view: '{img_name}' -> Object: '{object_name}'")
        
        if views:
            db_objects.append({
                "name": object_name,
                "views": views
            })

if not db_objects:
    print(f"No valid images found inside subfolders of '{DB_DIR}'.")
    exit()

print(f"\nSuccessfully loaded {len(db_objects)} object(s) into database.\n")

# 2. INITIALIZE MATCHER & CAMERA
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
camera = cv2.VideoCapture(0)
window_name = "Multi-Object Feature Detector"

MIN_MATCH_COUNT = 15

while True:
    ret, frame = camera.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)

    any_target_found = False

    if des_frame is not None and len(kp_frame) >= MIN_MATCH_COUNT:
        
        # Iterate over each registered object in database
        for obj in db_objects:
            obj_detected = False
            
            # Check against every loaded view 
            for view in obj["views"]:
                des_target = view["descriptors"]
                kp_target = view["kp"]
                h, w = view["shape"]

                # KNN match for ratio test
                matches = bf.knnMatch(des_target, des_frame, k=2)

                # Lowe's Ratio Test
                good_matches = []
                for match in matches:
                    if len(match) == 2:
                        m, n = match
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)

                # Verify matches and calculate Homography
                if len(good_matches) >= MIN_MATCH_COUNT:
                    src_pts = np.float32([kp_target[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                    if M is not None:
                        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                        transformed_pts = cv2.perspectiveTransform(pts, M)
                        int_pts = np.int32(transformed_pts)

                        # Check for valid shape (convex polygon)
                        if cv2.isContourConvex(int_pts):
                            # Draw outline
                            cv2.polylines(frame, [int_pts], True, (0, 255, 0), 3, cv2.LINE_AA)
                            
                            # Label with Object Name near top-left of target
                            label_pos = (int_pts[0][0][0], max(30, int_pts[0][0][1] - 10))
                            cv2.putText(frame, obj['name'], label_pos, 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            any_target_found = True
                            obj_detected = True
                            break  # Move to next object once a matching view is found

    if not any_target_found:
        cv2.putText(frame, "Searching", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

camera.release()
cv2.destroyAllWindows()