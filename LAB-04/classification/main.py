from data_loader import load_classification_data
from knn_tf import KNNTensorFlow
from evaluate import evaluate_k_curve, save_confusion_matrix, export_predictions
from sklearn.metrics import classification_report

def main():
    print("=== Running Classification Module ===")
    data = load_classification_data("../fpl.csv")
    
    best_k, accuracies = evaluate_k_curve(
        data["X_train"], data["y_train"], 
        data["X_test"], data["y_test"], 
        max_k=15, 
        output_path="outputs/01_k_curve.png"
    )
    print(f"Optimal k = {best_k} (Accuracy: {max(accuracies)*100:.2f}%)")
    
    knn_final = KNNTensorFlow(k=best_k)
    knn_final.fit(data["X_train"], data["y_train"])
    y_pred = knn_final.predict(data["X_test"])
    
    print("\nClassification Report:")
    print(classification_report(data["y_test"], y_pred))
    
    save_confusion_matrix(data["y_test"], y_pred, output_path="outputs/02_confusion_matrix.png")
    export_predictions(data["names_test"], data["y_test"], y_pred, output_path="outputs/predictions.csv")
    print("Classification Outputs Saved Successfully!")

if __name__ == "__main__":
    main()