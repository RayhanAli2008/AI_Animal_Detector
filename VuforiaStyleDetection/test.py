import os
import cv2
import numpy as np

# Tweak these if performance tanks
MIN_MATCHES = 15
RATIO_THRESHOLD = 0.75
MAX_FEATURES = 1500     
MAX_HAMMING_DIST = 60   # keep this tight to avoid false positives

# Homography limits
MIN_AREA = 900          # ~30x30 px 
MAX_AREA = 400000       # prevents massive screen-filling glitch polygons


def load_db(db_path):
    # Setup DB dir if it doesn't exist yet
    if not os.path.exists(db_path):
        os.makedirs(db_path)
        print(f"Created '{db_path}'. Drop some reference image folders in there with lots of pictures from different angles/lighting.")
        return []

    orb = cv2.ORB_create(nfeatures=MAX_FEATURES)
    database = []

    for folder in os.listdir(db_path):
        folder_path = os.path.join(db_path, folder)
        if not os.path.isdir(folder_path):
            continue

        views = []
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue

            kp, des = orb.detectAndCompute(img, None)
            if des is not None and len(kp) >= MIN_MATCHES:
                h, w = img.shape
                views.append({"kp": kp, "des": des, "size": (w, h)})

        if views:
            database.append({"name": folder, "views": views})

    return database


def find_instances(view, frame_kp, frame_des, matcher):
    target_des = view["des"]
    target_kp = view["kp"]
    w, h = view["size"]

    unmatched = list(range(len(frame_des)))
    found_boxes = []

    while len(unmatched) >= MIN_MATCHES:
        sub_des = frame_des[unmatched]
        matches = matcher.knnMatch(target_des, sub_des, k=2)

        good_matches = []
        matched_train_ids = []
        
        for m_pair in matches:
            if len(m_pair) == 2:
                m, n = m_pair
                # Ratio test + Hamming distance check
                if m.distance < MAX_HAMMING_DIST and m.distance < RATIO_THRESHOLD * n.distance:
                    good_matches.append(m)
                    matched_train_ids.append(m.trainIdx)

        if len(good_matches) < MIN_MATCHES:
            break

        src_pts = np.float32([target_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([frame_kp[unmatched[idx]].pt for idx in matched_train_ids]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if matrix is None or mask is None:
            break

        corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        projected = np.int32(cv2.perspectiveTransform(corners, matrix))

        area = cv2.contourArea(projected)
        if cv2.isContourConvex(projected) and MIN_AREA < area < MAX_AREA:
            found_boxes.append(projected)

        # Iteratively remove inliers to find duplicates in the same frame
        inliers = mask.ravel().tolist()
        used_train_ids = [matched_train_ids[i] for i, is_inlier in enumerate(inliers) if is_inlier]
        used_orig_indices = set(unmatched[i] for i in used_train_ids)
        unmatched = [idx for idx in unmatched if idx not in used_orig_indices]
        
        # Bail out if we get stuck in a weird homography loop
        if not used_train_ids:
            break

    return found_boxes


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, "objects_db")

    db = load_db(db_path)
    if not db:
        print("No object images found. Exiting.")
        return

    print(f"Loaded {len(db)} targets.")

    orb = cv2.ORB_create(nfeatures=MAX_FEATURES)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    cap = cv2.VideoCapture(0)
    
    # Force 720p. TODO: disable if this runs too slow on older webcams
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    win_title = "Tracker"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_kp, frame_des = orb.detectAndCompute(gray, None)

        target_found = False

        if frame_des is not None and len(frame_kp) >= MIN_MATCHES:
            for obj in db:
                for view in obj["views"]:
                    bounds_list = find_instances(view, frame_kp, frame_des, matcher)
                    
                    for bounds in bounds_list:
                        target_found = True
                        
                        # Draw bounding box
                        cv2.polylines(frame, [bounds], True, (0, 255, 0), 3, cv2.LINE_AA)
                        
                        label_x, label_y = bounds[0][0]
                        label_pos = (label_x, max(30, label_y - 10))
                        
                        # Text shadow trick so it doesn't blend into the background
                        cv2.putText(frame, obj["name"], label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                        cv2.putText(frame, obj["name"], label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if not target_found:
            cv2.putText(frame, "Searching...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow(win_title, frame)

        # Hit 'q' or close the window to quit
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty(win_title, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()