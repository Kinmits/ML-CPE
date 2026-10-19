import argparse
import cv2
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Real-time YOLO Detection")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained weights")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (usually 0 for webcam)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    args = parser.parse_args()

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.camera)

    print(f"Starting real-time detection with camera index {args.camera}. Press 'Q' to exit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read from camera. Exiting...")
            break

        # Run YOLO detection on the frame
        results = model(frame, conf=args.conf, verbose=False)
        
        # Plot bounding boxes on the frame
        annotated_frame = results[0].plot()

        cv2.imshow("YOLOv8 Real-time Detection", annotated_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()