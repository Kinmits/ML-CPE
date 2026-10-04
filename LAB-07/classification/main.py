from data_loader import load_data
from preprocessing import preprocess_data
from split_data import split_dataset
from cnn_model import build_cnn_model, train_and_save_model
from evaluate import evaluate_performance
from test_cnn import test_random_samples

def main():
    print("=== LAB 7: 1D CNN CLASSIFICATION ON FPL DATASET ===")
    
    # 1. Load Data
    X_raw, y, class_names, features = load_data("../fpl.csv")
    
    # 2. Preprocess
    X_processed, scaler = preprocess_data(X_raw)
    
    # 3. Split (Train, Val, Test)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X_processed, y)
    
    # 4. Configs
    input_shape = (X_train.shape[1], 1)  # มิติคือ (18, 1)
    num_classes = len(class_names)
    EPOCHS = 100  # ข้อมูลตารางมีขนาดเล็ก จึงต้องรันหลายรอบกว่ารูปภาพ
    
    # 5. Train Config 1
    model1 = build_cnn_model(input_shape, num_classes, config_type=1)
    history1 = train_and_save_model(model1, X_train, y_train, X_val, y_val, EPOCHS, "model_config1")
    evaluate_performance(model1, history1, X_test, y_test, class_names, "Config_1")
    
    # 6. Train Config 2
    model2 = build_cnn_model(input_shape, num_classes, config_type=2)
    history2 = train_and_save_model(model2, X_train, y_train, X_val, y_val, EPOCHS, "model_config2")
    evaluate_performance(model2, history2, X_test, y_test, class_names, "Config_2")
    
    # 7. Test Model (สุ่มนักเตะ 4 คน)
    test_random_samples("model_config2.h5", X_test, y_test, class_names)

if __name__ == "__main__":
    main()