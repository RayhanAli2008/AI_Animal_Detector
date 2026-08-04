import sys
import cv2
from ultralytics import YOLO

def main():
    try:
        model = YOLO("yolov8n.pt")
    except Exception as e:
        print(f"Model error: {e}")
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera error")
        sys.exit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.45, stream=True, verbose=False)

        for r in results:
            frame = r.plot()

        cv2.imshow("Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty("Tracker", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()