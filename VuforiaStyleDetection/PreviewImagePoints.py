import sys
from pathlib import Path
import cv2


def extract_features(image_path: str, max_features: int = 1000):
    path = Path(image_path.strip("\"'")).resolve()

    if not path.is_file():
        print(f"Error: File not found -> {path}")
        return

    img = cv2.imread(str(path))
    if img is None:
        print(f"Error: Couldn't load or decode image at {path}")
        return

    # ORB operates on grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=max_features)
    keypoints, _ = orb.detectAndCompute(gray, None)

    if not keypoints:
        print("No feature points detected. Try an image with more texture/contrast.")
        return

    print(f"Found {len(keypoints)} keypoints in '{path.name}'.")

    # Render keypoints on the image in yellow
    vis = cv2.drawKeypoints(
        img,
        keypoints,
        outImage=None,
        color=(0, 255, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
    )

    # Save output to the same directory as this script
    script_dir = Path(__file__).resolve().parent
    output_path = script_dir / f"{path.stem}_features{path.suffix}"

    cv2.imwrite(str(output_path), vis)
    print(f"Saved preview to: {output_path}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else input("Enter image path: ")

    if target.strip():
        extract_features(target)