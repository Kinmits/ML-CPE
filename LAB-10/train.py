import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLO Model for Lab 10")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained model (e.g., yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    args = parser.parse_args()

    # Load a pretrained YOLO model
    print(f"Loading pretrained model: {args.model}")
    model = YOLO(args.model)

    # Train the model (Data should be in dataset/data.yaml)
    print(f"Starting training for {args.epochs} epochs, batch size {args.batch}, image size {args.imgsz}...")
    results = model.train(
        data="dataset/data.yaml",
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project="outputs",    # Save results in 'outputs' folder
        name=f"train_e{args.epochs}_b{args.batch}" # Name folder based on config
    )
    
    print("\nTraining completed! Results, Loss graphs, and Weights are saved in the 'outputs' directory.")

if __name__ == "__main__":
    main()