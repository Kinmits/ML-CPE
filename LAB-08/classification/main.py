import os
from data_loader import load_images_from_folder
from preprocessing import preprocess_images
from split_data import split_and_save_dataset
from vgg_model import build_vgg_model, train_and_save_model
from evaluate import evaluate_performance
from test_vgg import test_random_images

def main():
    print("=== LAB 8: DEEP CNN & TRANSFER LEARNING (VGG16) ===")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    DATASET_PATH = os.path.join(current_dir, "..", "PetImages")
    
    IMG_SIZE = (224, 224)
    EPOCHS = 10
    BATCH_SIZE = 32
    
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Dataset folder '{DATASET_PATH}' not found!")
        return
        
    # เช็คว่ามีโฟลเดอร์ลูกอยู่ข้างใน PetImages ไหม
    subfolders = [f.path for f in os.scandir(DATASET_PATH) if f.is_dir()]
    if not subfolders:
         print(f"[ERROR] No class folders (like 'bear', 'panda') found inside '{DATASET_PATH}'!")
         return
        
    # 1. Load Data
    raw_images, labels, class_names = load_images_from_folder(DATASET_PATH)
    
    # 2. Preprocess
    processed_images = preprocess_images(raw_images, target_size=IMG_SIZE)
    
    # 3. Split & Save .npy arrays
    X_train, X_val, X_test, y_train, y_val, y_test = split_and_save_dataset(processed_images, labels)
    
    # 4. Build Model
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
    num_classes = len(class_names)
    model = build_vgg_model(input_shape, num_classes)
    
    # 5. Train & Save
    history = train_and_save_model(model, X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # 6. Evaluate
    evaluate_performance(model, history, X_test, y_test)
    
    # 7. Test Random Samples
    test_random_images("outputs/vgg_model.keras")

if __name__ == "__main__":
    main()