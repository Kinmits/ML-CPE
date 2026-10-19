import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Test YOLO on an Image/Video")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained weights")
    parser.add_argument("--source", type=str, required=True, help="Path to test image or folder of images")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    args = parser.parse_args()

    model = YOLO(args.weights)
    
    # Run prediction and save results
    print(f"Running inference on: {args.source}")
    results = model.predict(source=args.source, conf=args.conf, save=True, project="outputs", name="predict_results")
    
    print("\nInference completed! Annotated images are saved in the 'outputs/predict_results' folder.")

if __name__ == "__main__":
    main()