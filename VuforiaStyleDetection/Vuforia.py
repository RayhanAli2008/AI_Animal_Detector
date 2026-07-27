import os
import cv2
import numpy as np

# Detector configuration
MIN_MATCHES = 15
RATIO_THRESHOLD = 0.75
MAX_FEATURES = 1000


def load_database(db_path: str):
    """Loads feature descriptors and shapes for all target images in the database."""
    if not os.path.exists(db_path):
        os.makedirs(db_path)
        print(f"Created directory: '{db_path}'. Add object subfolders containing reference images.")
        return []

    orb = cv2.ORB_create(nfeatures=MAX_FEATURES)
    database = []

    for folder_name in os.listdir(db_path):
        folder_path = os.path.join(db_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        views = []
        for file_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            kp, des = orb.detectAndCompute(img, None)
            if des is not None:
                h, w = img.shape
                views.append({"kp": kp, "des": des, "size": (w, h)})

        if views:
            database.append({"name": folder_name, "views": views})

    return database


def find_instances(view: dict, frame_kp: list, frame_des: np.ndarray, matcher: cv2.BFMatcher):
    """
    Searches for all matching instances of a target view within frame descriptors.
    Removes matched inliers iteratively to support duplicate detection.
    """
    target_des = view["des"]
    target_kp = view["kp"]
    w, h = view["size"]

    unmatched_indices = list(range(len(frame_des)))
    detected_bounds = []

    while len(unmatched_indices) >= MIN_MATCHES:
        sub_des = frame_des[unmatched_indices]
        matches = matcher.knnMatch(target_des, sub_des, k=2)

        # Ratio test filtering
        good_matches = []
        matched_train_ids = []
        for m_pair in matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < RATIO_THRESHOLD * n.distance:
                    good_matches.append(m)
                    matched_train_ids.append(m.trainIdx)

        if len(good_matches) < MIN_MATCHES:
            break

        src_pts = np.float32([target_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([frame_kp[unmatched_indices[idx]].pt for idx in matched_train_ids]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if matrix is None or mask is None:
            break

        corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        projected = np.int32(cv2.perspectiveTransform(corners, matrix))

        if not cv2.isContourConvex(projected):
            break

        detected_bounds.append(projected)

        # Mask out used keypoints to continue searching for duplicate objects
        inliers = mask.ravel().tolist()
        used_train_ids = [matched_train_ids[i] for i, is_inlier in enumerate(inliers) if is_inlier]
        used_orig_indices = set(unmatched_indices[i] for i in used_train_ids)
        unmatched_indices = [idx for idx in unmatched_indices if idx not in used_orig_indices]

    return detected_bounds


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, "objects_db")

    database = load_database(db_path)
    if not database:
        print(f"No object images found in '{db_path}'. Exiting.")
        return

    print(f"Successfully loaded {len(database)} object category(s).")

    orb = cv2.ORB_create(nfeatures=MAX_FEATURES)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    cap = cv2.VideoCapture(0)

    win_title = "Multi-Object Real-Time Tracker"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_kp, frame_des = orb.detectAndCompute(gray, None)

        target_found = False

        if frame_des is not None and len(frame_kp) >= MIN_MATCHES:
            for obj in database:
                for view in obj["views"]:
                    bounds_list = find_instances(view, frame_kp, frame_des, matcher)
                    for bounds in bounds_list:
                        target_found = True
                        cv2.polylines(frame, [bounds], isClosed=True, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
                        
                        label_x, label_y = bounds[0][0]
                        label_pos = (label_x, max(30, label_y - 10))
                        cv2.putText(frame, obj["name"], label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if not target_found:
            cv2.putText(frame, "Searching...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow(win_title, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty(win_title, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()