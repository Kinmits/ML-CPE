import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained YOLO Model")
    parser.add_argument("--weights", type=str, required=True, help="Path to the trained weights (e.g., outputs/train_e10_b16/weights/best.pt)")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Dataset split to evaluate on")
    args = parser.parse_args()

    # Load the trained model
    print(f"Loading trained weights: {args.weights}")
    model = YOLO(args.weights)

    # Evaluate model performance
    print(f"Evaluating model on '{args.split}' split...")
    metrics = model.val(data="dataset/data.yaml", split=args.split, project="outputs", plots=False)
    
    print("\n" + "="*40)
    print("         EVALUATION RESULTS")
    print("="*40)
    print(f"mAP50-95 : {metrics.box.map:.4f}")
    print(f"mAP50    : {metrics.box.map50:.4f}")
    print(f"mAP75    : {metrics.box.map75:.4f}")
    print("*" * 40)
    print("Detailed class metrics are saved in the runs folder.")

if __name__ == "__main__":
    main()